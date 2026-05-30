"""Wiki compilation pipeline: LLM overview → concept plan → concept pages → backlinks → index."""
from __future__ import annotations

from pathlib import Path

from tqdm.asyncio import tqdm

from openindex.agents import AgentPool, run_structured, run_text
from openindex.utils.logger import get_logger
from openindex.wiki.models import ConceptPage, ConceptPlan
from openindex.wiki.prompts import (
    build_concept_create_prompt,
    build_concept_update_prompt,
    build_concepts_plan_prompt,
    build_overview_prompt,
)
from openindex.wiki.renderer import (
    add_related_link,
    backlink_concepts,
    backlink_summary,
    list_existing_targets,
    read_concept_briefs,
    sanitize_slug,
    strip_ghost_wikilinks,
    upsert_index_concept,
    upsert_index_doc,
    write_concept_page,
    write_sources_json,
    write_summary_md,
)

logger = get_logger(__name__)


def _format_known_targets(targets: set[str]) -> str:
    """Format the wikilink whitelist as a bullet list for prompt injection.

    Args:
        targets: Set of valid wikilink target strings.

    Returns:
        Newline-separated bullet list, or a no-links sentinel string.
    """
    if not targets:
        return "(none yet — do not use any [[wikilinks]])"
    return "\n".join(f"- {t}" for t in sorted(targets))


async def compile_wiki(
    result: dict,
    wiki_dir: Path,
    pool: AgentPool,
) -> None:
    """Run the full wiki compilation pipeline for one document.

    Takes a WikiIndex.build() result dict and writes/updates the wiki:
      1. Write sources JSON and summary Markdown (no LLM)
      2. Generate prose overview from the section tree summary
      3. Plan which concept pages to create/update/cross-link
      4. Generate concept pages concurrently
      5. Strip ghost wikilinks
      6. Write concept pages to disk
      7. Add related cross-reference links (code-only)
      8. Backlink summary ↔ concepts (code-only)
      9. Update index.md

    Args:
        result: Output of WikiIndex.build() — must have doc_name, nodes, pages, description.
        wiki_dir: Root wiki directory (wiki/ folder will be written here directly).
        pool: Shared AgentPool with wiki_planner, wiki_concept, summarizer agents.
    """
    doc_name = Path(result["doc_name"]).stem
    logger.info("compile_wiki_start", doc=doc_name)

    # --- Step 1: Write file artifacts (no LLM) ---
    write_sources_json(result, wiki_dir)
    summary_path = write_summary_md(result, wiki_dir)
    logger.info("wiki_files_written", doc=doc_name)

    # --- Step 2: Overview ---
    summary_md = summary_path.read_text(encoding="utf-8")
    overview = await run_text(pool.summarizer, build_overview_prompt(doc_name, summary_md), pool.sem)
    logger.info("overview_done", doc=doc_name)

    # --- Step 3: Read existing concepts + plan ---
    concept_briefs = read_concept_briefs(wiki_dir)
    plan: ConceptPlan = await run_structured(
        pool.wiki_planner,
        build_concepts_plan_prompt(overview, concept_briefs),
        pool.sem,
    )
    logger.info("concept_plan_done", create=len(plan.create), update=len(plan.update), related=len(plan.related))

    # --- Step 4: Build wikilink whitelist ---
    planned_slugs = {sanitize_slug(c.name) for c in plan.create + plan.update}
    known_targets: set[str] = (
        list_existing_targets(wiki_dir)
        | {f"concepts/{s}" for s in planned_slugs}
        | {f"summaries/{doc_name}"}
    )
    known_targets_str = _format_known_targets(known_targets)

    # --- Step 5: Generate concept pages concurrently ---
    async def _gen_create(item) -> tuple[str, ConceptPage, bool]:
        slug = sanitize_slug(item.name)
        page: ConceptPage = await run_structured(
            pool.wiki_concept,
            build_concept_create_prompt(item.title, doc_name, overview, known_targets_str),
            pool.sem,
        )
        return slug, page, False

    async def _gen_update(item) -> tuple[str, ConceptPage, bool]:
        slug = sanitize_slug(item.name)
        concept_path = wiki_dir / "concepts" / f"{slug}.md"
        if concept_path.exists():
            raw = concept_path.read_text(encoding="utf-8")
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                existing_content = parts[2].strip() if len(parts) >= 3 else raw
            else:
                existing_content = raw
        else:
            existing_content = "(page not found — create from scratch)"
        page: ConceptPage = await run_structured(
            pool.wiki_concept,
            build_concept_update_prompt(item.title, doc_name, overview, existing_content, known_targets_str),
            pool.sem,
        )
        return slug, page, True

    tasks = [_gen_create(c) for c in plan.create] + [_gen_update(c) for c in plan.update]

    results: list[tuple[str, ConceptPage, bool]] = []
    if tasks:
        gathered = await tqdm.gather(*tasks, desc="Generating concepts", unit="concept")
        results = [r for r in gathered if r is not None]

    logger.info("concepts_generated", count=len(results))

    # --- Step 6: Strip ghost wikilinks + write concept pages ---
    concept_slugs: list[str] = []
    concept_briefs_map: dict[str, str] = {}

    for slug, page, is_update in results:
        cleaned_content = strip_ghost_wikilinks(page.content, known_targets)
        write_concept_page(
            wiki_dir,
            slug=slug,
            content=cleaned_content,
            brief=page.brief,
            source_doc=doc_name,
            is_update=is_update,
        )
        concept_slugs.append(slug)
        concept_briefs_map[slug] = page.brief

    # --- Step 7: Related cross-references (code-only) ---
    source_file = f"summaries/{doc_name}.md"
    for slug in plan.related:
        sanitized = sanitize_slug(slug)
        add_related_link(wiki_dir, sanitized, doc_name, source_file)

    # --- Step 8: Backlinks (code-only) ---
    all_slugs = concept_slugs + [sanitize_slug(s) for s in plan.related]
    if all_slugs:
        backlink_summary(wiki_dir, doc_name, all_slugs)
        backlink_concepts(wiki_dir, doc_name, all_slugs)

    # --- Step 9: Update index.md ---
    upsert_index_doc(wiki_dir, doc_name, brief=result.get("description", ""))
    for slug in concept_slugs:
        upsert_index_concept(wiki_dir, slug, brief=concept_briefs_map.get(slug, ""))

    logger.info("compile_wiki_done", doc=doc_name, concepts=len(concept_slugs))

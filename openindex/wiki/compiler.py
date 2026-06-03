"""Wiki compilation pipeline: LLM overview → concept plan → concept pages → backlinks → index."""
from __future__ import annotations

from pathlib import Path

from tqdm.asyncio import tqdm

from openindex.agents import AgentPool, run_structured, run_text
from openindex.utils.logger import get_logger
from openindex.wiki.models import ConceptEntry, ConceptPage, ConceptPlan, SourcePage, WikiDict
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
    render_nodes,
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
    if not targets:
        return "(none yet — do not use any [[wikilinks]])"
    return "\n".join(f"- {t}" for t in sorted(targets))


async def compile_wiki(
    result: dict,
    pool: AgentPool,
    wiki_dir: Path | None = None,
) -> WikiDict:
    """Run the full wiki compilation pipeline for one document.

    When wiki_dir is provided, artifacts are written/updated on disk and the
    planner reads existing concepts for LLM-based concept merging.
    When wiki_dir is None, runs fully in-memory (no disk reads or writes).

    Steps:
      1. Build summary Markdown (write to disk if wiki_dir set)
      2. Generate prose overview
      3. Plan concept pages (reads existing on disk if wiki_dir set)
      4. Generate concept pages concurrently (LLM merge for updates when wiki_dir set)
      5. Strip ghost wikilinks
      6. Write concept pages + backlinks + index.md (only if wiki_dir set)

    Args:
        result: Output of WikiIndex.build() — must have doc_name, nodes, pages, description.
        pool: Shared AgentPool with wiki_planner, wiki_concept, summarizer agents.
        wiki_dir: Root wiki directory. None = in-memory only, no disk I/O.

    Returns:
        WikiDict with compiled artifacts for this document.
    """
    doc_name = Path(result["doc_name"]).stem
    logger.info("compile_wiki_start", doc=doc_name, disk=wiki_dir is not None)

    # --- Sources ---
    pages_dict: dict = result.get("pages", {})
    sources = [
        SourcePage(page=page_num, content=text)
        for page_num, text in sorted(pages_dict.items())
    ]

    # --- Summary markdown ---
    if wiki_dir is not None:
        wiki_dir.mkdir(parents=True, exist_ok=True)
        write_sources_json(result, wiki_dir)
        summary_path = write_summary_md(result, wiki_dir)
        summary_md = summary_path.read_text(encoding="utf-8")
        logger.info("wiki_files_written", doc=doc_name)
    else:
        frontmatter = f"---\ndoc_type: pageindex\nfull_text: sources/{doc_name}.json\n---\n\n"
        summary_md = frontmatter + render_nodes(result.get("nodes", []), depth=1)

    # --- Overview ---
    overview = await run_text(pool.summarizer, build_overview_prompt(doc_name, summary_md), pool.sem)
    logger.info("overview_done", doc=doc_name)

    # --- Plan ---
    concept_briefs = read_concept_briefs(wiki_dir) if wiki_dir is not None else "(none yet)"
    plan: ConceptPlan = await run_structured(
        pool.wiki_planner,
        build_concepts_plan_prompt(overview, concept_briefs),
        pool.sem,
    )
    logger.info("concept_plan_done", create=len(plan.create), update=len(plan.update), related=len(plan.related))

    # --- Wikilink whitelist ---
    planned_slugs = {sanitize_slug(c.name) for c in plan.create + plan.update}
    known_targets: set[str] = {f"concepts/{s}" for s in planned_slugs} | {f"summaries/{doc_name}"}
    if wiki_dir is not None:
        known_targets |= list_existing_targets(wiki_dir)
    known_targets_str = _format_known_targets(known_targets)

    # --- Generate concept pages concurrently ---
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
        existing_content = "(page not found — create from scratch)"
        if wiki_dir is not None:
            concept_path = wiki_dir / "concepts" / f"{slug}.md"
            if concept_path.exists():
                raw = concept_path.read_text(encoding="utf-8")
                if raw.startswith("---"):
                    parts = raw.split("---", 2)
                    existing_content = parts[2].strip() if len(parts) >= 3 else raw
                else:
                    existing_content = raw
        page: ConceptPage = await run_structured(
            pool.wiki_concept,
            build_concept_update_prompt(item.title, doc_name, overview, existing_content, known_targets_str),
            pool.sem,
        )
        return slug, page, True

    # In-memory mode: treat plan.update as creates (no existing content to merge)
    if wiki_dir is not None:
        tasks = [_gen_create(c) for c in plan.create] + [_gen_update(c) for c in plan.update]
    else:
        tasks = [_gen_create(c) for c in plan.create + plan.update]

    raw_results: list[tuple[str, ConceptPage, bool]] = []
    if tasks:
        gathered = await tqdm.gather(*tasks, desc="Generating concepts", unit="concept")
        raw_results = [r for r in gathered if r is not None]

    logger.info("concepts_generated", count=len(raw_results))

    # --- Strip ghost wikilinks + optionally write to disk ---
    concepts: dict[str, ConceptEntry] = {}
    for slug, page, is_update in raw_results:
        cleaned = strip_ghost_wikilinks(page.content, known_targets)
        concepts[slug] = ConceptEntry(brief=page.brief, content=cleaned)
        if wiki_dir is not None:
            write_concept_page(
                wiki_dir,
                slug=slug,
                content=cleaned,
                brief=page.brief,
                source_doc=doc_name,
                is_update=is_update,
            )

    concept_slugs = list(concepts.keys())
    related_slugs = [sanitize_slug(s) for s in plan.related]
    description = result.get("description", "")

    # --- Disk-only: backlinks + index.md ---
    if wiki_dir is not None:
        source_file = f"summaries/{doc_name}.md"
        for slug in related_slugs:
            add_related_link(wiki_dir, slug, doc_name, source_file)

        all_slugs = concept_slugs + [s for s in related_slugs if s not in concept_slugs]
        if all_slugs:
            backlink_summary(wiki_dir, doc_name, all_slugs)
            backlink_concepts(wiki_dir, doc_name, all_slugs)

        upsert_index_doc(wiki_dir, doc_name, brief=description)
        for slug in concept_slugs:
            upsert_index_concept(wiki_dir, slug, brief=concepts[slug].brief)

        index_md = (wiki_dir / "index.md").read_text(encoding="utf-8") if (wiki_dir / "index.md").exists() else ""
    else:
        index_lines = ["# Knowledge Base Index", "", "## Documents", ""]
        index_lines.append(f"- [[summaries/{doc_name}]] (pageindex) — {description}")
        index_lines += ["", "## Concepts", ""]
        for slug, entry in sorted(concepts.items()):
            line = f"- [[concepts/{slug}]]"
            if entry.brief:
                line += f" — {entry.brief}"
            index_lines.append(line)
        index_md = "\n".join(index_lines)

    logger.info("compile_wiki_done", doc=doc_name, concepts=len(concept_slugs), disk=wiki_dir is not None)
    return WikiDict(
        doc_name=doc_name,
        description=description,
        summary=summary_md,
        sources=sources,
        concepts=concepts,
        related=related_slugs,
        index=index_md,
    )

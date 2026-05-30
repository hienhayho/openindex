"""Pure file I/O for wiki folder generation — no LLM calls."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Sources JSON
# ---------------------------------------------------------------------------

def write_sources_json(result: dict, wiki_dir: Path) -> Path:
    """Write per-page content as a JSON array to wiki/sources/<stem>.json.

    Converts result["pages"] (dict[int, str]) into the OpenKB-compatible
    array format: [{"page": N, "content": "...", "images": []}, ...].

    Args:
        result: PageIndex.build() output dict with "doc_name" and "pages" keys.
        wiki_dir: Root wiki directory.

    Returns:
        Path to the written JSON file.
    """
    stem = Path(result["doc_name"]).stem
    sources_dir = wiki_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    pages_dict: dict[int, str] = result.get("pages", {})
    pages_array = [
        {"page": page_num, "content": text, "images": []}
        for page_num, text in sorted(pages_dict.items())
    ]

    out_path = sources_dir / f"{stem}.json"
    out_path.write_text(json.dumps(pages_array, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Summary Markdown
# ---------------------------------------------------------------------------

def _render_nodes(nodes: list[dict], depth: int = 1) -> str:
    """Recursively render section nodes as Markdown headings with summaries.

    Args:
        nodes: List of node dicts with title, start_index, end_index, summary, children.
        depth: Current heading depth (1 = #, 2 = ##, etc.).

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    prefix = "#" * min(depth, 6)
    for node in nodes:
        title = node.get("title", "")
        start = node.get("start_index", "")
        end = node.get("end_index", "")
        summary = node.get("summary", "")
        children = node.get("children", [])

        lines.append(f"{prefix} {title} (pages {start}–{end})\n")
        if summary:
            lines.append(f"Summary: {summary}\n")
        if children:
            lines.append(_render_nodes(children, depth + 1))

    return "\n".join(lines)


def write_summary_md(result: dict, wiki_dir: Path) -> Path:
    """Write the section tree as a Markdown summary page with YAML frontmatter.

    Frontmatter: doc_type=pageindex, full_text=sources/<stem>.json

    Args:
        result: PageIndex.build() output with "doc_name", "nodes" keys.
        wiki_dir: Root wiki directory.

    Returns:
        Path to the written Markdown file.
    """
    stem = Path(result["doc_name"]).stem
    summaries_dir = wiki_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    frontmatter = f"---\ndoc_type: pageindex\nfull_text: sources/{stem}.json\n---\n\n"
    body = _render_nodes(result.get("nodes", []), depth=1)

    out_path = summaries_dir / f"{stem}.md"
    out_path.write_text(frontmatter + body, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# index.md management
# ---------------------------------------------------------------------------

def ensure_index(wiki_dir: Path) -> None:
    """Create wiki/index.md with stub sections if it does not exist.

    Args:
        wiki_dir: Root wiki directory.
    """
    index_path = wiki_dir / "index.md"
    if not index_path.exists():
        wiki_dir.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            "# Knowledge Base Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )


def _get_section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    """Return [start, end) line range for an H2 section.

    Args:
        lines: All lines of the file.
        heading: Exact H2 heading string, e.g. "## Documents".

    Returns:
        (start, end) tuple where start is the line after the heading and end
        is the line of the next H2 (or EOF). None if heading not found.
    """
    h2s = [(i, line.rstrip()) for i, line in enumerate(lines) if line.startswith("## ")]
    for k, (idx, normalized) in enumerate(h2s):
        if normalized == heading:
            start = idx + 1
            end = h2s[k + 1][0] if k + 1 < len(h2s) else len(lines)
            return start, end
    return None


def _ensure_h2_section(lines: list[str], heading: str) -> None:
    """Append an H2 section to lines if it is missing.

    Args:
        lines: File lines (mutated in place).
        heading: H2 heading to ensure, e.g. "## Related Concepts".
    """
    if _get_section_bounds(lines, heading) is not None:
        return
    while lines and lines[-1] == "":
        lines.pop()
    if lines:
        lines.append("")
    lines.append(heading)
    lines.append("")


def _section_contains_link(lines: list[str], heading: str, link: str) -> bool:
    """Check if a bullet entry starting with `- {link}` exists in the section.

    Args:
        lines: File lines.
        heading: H2 section heading.
        link: Wikilink string, e.g. "[[summaries/paper]]".

    Returns:
        True if a matching bullet exists in the section.
    """
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return False
    start, end = bounds
    prefix = f"- {link}"
    return any(line.startswith(prefix) for line in lines[start:end])


def _insert_section_entry(lines: list[str], heading: str, entry: str) -> None:
    """Prepend a bullet entry to the top of an H2 section.

    Args:
        lines: File lines (mutated in place).
        heading: H2 section heading.
        entry: Full bullet line, e.g. "- [[summaries/paper]] (pageindex) — brief".
    """
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return
    start, _ = bounds
    lines.insert(start, entry)


def _replace_section_entry(lines: list[str], heading: str, link: str, entry: str) -> bool:
    """Replace an existing bullet entry within an H2 section.

    Args:
        lines: File lines (mutated in place).
        heading: H2 section heading.
        link: Wikilink prefix to match, e.g. "[[summaries/paper]]".
        entry: New full bullet line.

    Returns:
        True if a matching entry was found and replaced.
    """
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return False
    start, end = bounds
    prefix = f"- {link}"
    for i in range(start, end):
        if lines[i].startswith(prefix):
            lines[i] = entry
            return True
    return False


def upsert_index_doc(wiki_dir: Path, doc_name: str, brief: str = "") -> None:
    """Insert or update a document entry in wiki/index.md under ## Documents.

    Entry format: `- [[summaries/<doc_name>]] (pageindex) — brief`

    Args:
        wiki_dir: Root wiki directory.
        doc_name: Document stem (no extension).
        brief: One-line description shown in the index.
    """
    ensure_index(wiki_dir)
    index_path = wiki_dir / "index.md"
    lines = index_path.read_text(encoding="utf-8").split("\n")

    _ensure_h2_section(lines, "## Documents")
    link = f"[[summaries/{doc_name}]]"
    entry = f"- {link} (pageindex)"
    if brief:
        entry += f" — {brief}"

    if _section_contains_link(lines, "## Documents", link):
        _replace_section_entry(lines, "## Documents", link, entry)
    else:
        _insert_section_entry(lines, "## Documents", entry)

    index_path.write_text("\n".join(lines), encoding="utf-8")


def upsert_index_concept(wiki_dir: Path, slug: str, brief: str = "") -> None:
    """Insert or update a concept entry in wiki/index.md under ## Concepts.

    Entry format: `- [[concepts/<slug>]] — brief`

    Args:
        wiki_dir: Root wiki directory.
        slug: Sanitized concept slug.
        brief: One-line definition shown in the index.
    """
    ensure_index(wiki_dir)
    index_path = wiki_dir / "index.md"
    lines = index_path.read_text(encoding="utf-8").split("\n")

    _ensure_h2_section(lines, "## Concepts")
    link = f"[[concepts/{slug}]]"
    entry = f"- {link}"
    if brief:
        entry += f" — {brief}"

    if _section_contains_link(lines, "## Concepts", link):
        _replace_section_entry(lines, "## Concepts", link, entry)
    else:
        _insert_section_entry(lines, "## Concepts", entry)

    index_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Concept page helpers
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^\w\-]")


def sanitize_slug(name: str) -> str:
    """Convert a concept name to a safe filename slug.

    Args:
        name: Raw concept name from LLM output.

    Returns:
        Kebab-case, ASCII-only slug safe for filenames.
    """
    name = unicodedata.normalize("NFKC", name)
    return _SAFE_NAME_RE.sub("-", name).strip("-") or "unnamed-concept"


def read_concept_briefs(wiki_dir: Path) -> str:
    """Read existing concept pages and return compact one-line summaries.

    Reads `brief:` from YAML frontmatter if present; otherwise truncates
    the first 150 chars of the body.

    Args:
        wiki_dir: Root wiki directory.

    Returns:
        Bullet list string like "- slug: brief text\\n..." or "(none yet)".
    """
    concepts_dir = wiki_dir / "concepts"
    if not concepts_dir.exists():
        return "(none yet)"

    md_files = sorted(concepts_dir.glob("*.md"))
    if not md_files:
        return "(none yet)"

    lines: list[str] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        brief = ""
        body = text
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                fm_text = text[3:end].strip("\n")
                body = text[end + 3:]
                try:
                    fm = yaml.safe_load(fm_text)
                except yaml.YAMLError:
                    fm = None
                if isinstance(fm, dict) and isinstance(fm.get("brief"), str):
                    brief = fm["brief"].strip()
        if not brief:
            brief = body.strip().replace("\n", " ")[:150]
        if brief:
            lines.append(f"- {path.stem}: {brief}")

    return "\n".join(lines) or "(none yet)"


def list_existing_targets(wiki_dir: Path) -> set[str]:
    """Return all valid wikilink targets currently on disk.

    Scans summaries/ and concepts/ for .md files and returns their
    relative paths without extension, e.g. {"summaries/paper", "concepts/attention"}.

    Args:
        wiki_dir: Root wiki directory.

    Returns:
        Set of target strings usable in [[wikilinks]].
    """
    targets: set[str] = set()
    for subdir in ("summaries", "concepts"):
        d = wiki_dir / subdir
        if d.exists():
            for p in d.glob("*.md"):
                targets.add(f"{subdir}/{p.stem}")
    return targets


def strip_ghost_wikilinks(text: str, valid_targets: set[str]) -> str:
    """Remove [[wikilinks]] whose target is not in valid_targets.

    Replaces `[[target]]` with plain `target` when target is not in the
    whitelist, preventing broken links in the wiki.

    Args:
        text: Markdown text potentially containing [[wikilinks]].
        valid_targets: Set of valid target strings (without [[ ]]).

    Returns:
        Text with invalid wikilinks converted to plain text.
    """
    def _replace(m: re.Match) -> str:
        target = m.group(1)
        return m.group(0) if target in valid_targets else target

    return re.sub(r"\[\[([^\]]+)\]\]", _replace, text)


def write_concept_page(
    wiki_dir: Path,
    slug: str,
    content: str,
    brief: str,
    source_doc: str,
    is_update: bool,
) -> None:
    """Write or update a concept page with YAML frontmatter.

    For new pages: creates frontmatter with sources list.
    For updates: prepends source_doc to existing sources list if not present,
    then replaces body with new content.

    Args:
        wiki_dir: Root wiki directory.
        slug: Sanitized concept slug.
        content: Markdown body (no frontmatter).
        brief: One-sentence definition.
        source_doc: Document stem name, used as "summaries/<source_doc>.md" in sources list.
        is_update: True to update existing page, False to create new.
    """
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    path = concepts_dir / f"{slug}.md"

    source_file = f"summaries/{source_doc}.md"

    if is_update and path.exists():
        existing = path.read_text(encoding="utf-8")
        # Prepend source to frontmatter if missing
        if source_file not in existing:
            existing = _prepend_source(existing, source_file)
        # Replace body, keep frontmatter
        if existing.startswith("---"):
            end = existing.find("---", 3)
            if end != -1:
                fm = existing[:end + 3]
                # Update brief in frontmatter
                brief_line = f'brief: {json.dumps(brief, ensure_ascii=False)}'
                if "brief:" in fm:
                    fm = re.sub(r"brief:.*", lambda _: brief_line, fm)
                else:
                    fm = fm.replace("---\n", f"---\n{brief_line}\n", 1)
                existing = fm + "\n\n" + content
            else:
                existing = content
        path.write_text(existing, encoding="utf-8")
    else:
        sources_line = f"sources: {json.dumps([source_file], ensure_ascii=False)}"
        brief_line = f"brief: {json.dumps(brief, ensure_ascii=False)}"
        frontmatter = f"---\n{sources_line}\n{brief_line}\n---\n\n"
        path.write_text(frontmatter + content, encoding="utf-8")


def _prepend_source(text: str, source_file: str) -> str:
    """Prepend source_file to the sources: list in YAML frontmatter.

    Args:
        text: Full concept page text.
        source_file: Source path to prepend, e.g. "summaries/paper.md".

    Returns:
        Updated text with source_file at the front of sources list.
    """
    if not text.startswith("---"):
        sources_line = f"sources: {json.dumps([source_file], ensure_ascii=False)}"
        return f"---\n{sources_line}\n---\n\n" + text

    fm_end = text.find("---", 3)
    if fm_end == -1:
        return text

    fm_block = text[:fm_end]
    body = text[fm_end:]
    fm_lines = fm_block.split("\n")

    for i, line in enumerate(fm_lines):
        if not line.lstrip().startswith("sources:"):
            continue
        colon = line.find(":")
        try:
            items: list = yaml.safe_load(line[colon + 1:]) or []
        except yaml.YAMLError:
            return text
        if source_file in items:
            return text
        items.insert(0, source_file)
        fm_lines[i] = f"sources: {json.dumps(items, ensure_ascii=False)}"
        return "\n".join(fm_lines) + body

    fm_lines.insert(1, f"sources: {json.dumps([source_file], ensure_ascii=False)}")
    return "\n".join(fm_lines) + body


def backlink_summary(wiki_dir: Path, doc_name: str, concept_slugs: list[str]) -> None:
    """Append missing concept wikilinks to the summary page.

    Ensures bidirectional links: summary → concepts.

    Args:
        wiki_dir: Root wiki directory.
        doc_name: Document stem name.
        concept_slugs: Sanitized concept slugs to link.
    """
    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not summary_path.exists():
        return

    text = summary_path.read_text(encoding="utf-8")
    missing = [s for s in concept_slugs if f"[[concepts/{s}]]" not in text]
    if not missing:
        return

    lines = text.split("\n")
    _ensure_h2_section(lines, "## Related Concepts")
    for slug in reversed(missing):
        _insert_section_entry(lines, "## Related Concepts", f"- [[concepts/{slug}]]")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def backlink_concepts(wiki_dir: Path, doc_name: str, concept_slugs: list[str]) -> None:
    """Append missing summary wikilink to each concept page.

    Ensures bidirectional links: concepts → summary.

    Args:
        wiki_dir: Root wiki directory.
        doc_name: Document stem name.
        concept_slugs: Sanitized concept slugs to update.
    """
    link = f"[[summaries/{doc_name}]]"
    concepts_dir = wiki_dir / "concepts"

    for slug in concept_slugs:
        path = concepts_dir / f"{slug}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if link in text:
            continue
        lines = text.split("\n")
        _ensure_h2_section(lines, "## Related Documents")
        _insert_section_entry(lines, "## Related Documents", f"- {link}")
        path.write_text("\n".join(lines), encoding="utf-8")


def add_related_link(wiki_dir: Path, concept_slug: str, doc_name: str, source_file: str) -> None:
    """Add a cross-reference link to an existing concept page (no LLM).

    Appends "See also: [[summaries/<doc_name>]]" if not already present.

    Args:
        wiki_dir: Root wiki directory.
        concept_slug: Sanitized concept slug.
        doc_name: Document stem name.
        source_file: "summaries/<doc_name>.md" for frontmatter tracking.
    """
    path = wiki_dir / "concepts" / f"{concept_slug}.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    link = f"[[summaries/{doc_name}]]"
    if link in text:
        return

    if source_file not in text:
        text = _prepend_source(text, source_file)

    text += f"\n\nSee also: {link}"
    path.write_text(text, encoding="utf-8")

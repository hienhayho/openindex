from __future__ import annotations


def build_overview_prompt(doc_name: str, summary_md: str) -> str:
    """Build prompt to condense a WikiIndex tree summary into a prose overview.

    Args:
        doc_name: Document stem name (no extension).
        summary_md: Markdown summary with section headings and summaries.

    Returns:
        Prompt string for the summarizer agent.
    """
    return f"""This is a structured section summary for document "{doc_name}":

{summary_md}

Write a concise prose overview (3-5 sentences) capturing the document's key themes, \
findings, and contributions. Do not list sections — synthesize the content.
Return only the overview text, no headings or labels."""


def build_concepts_plan_prompt(overview: str, concept_briefs: str) -> str:
    """Build prompt for the wiki planner to decide which concept pages to create/update.

    Args:
        overview: Prose overview of the document being indexed.
        concept_briefs: Bullet list of existing concept slugs and their briefs.

    Returns:
        Prompt string for the wiki_planner agent (returns ConceptPlan).
    """
    return f"""You are maintaining a knowledge base wiki. A new document has been indexed.

Document overview:
{overview}

Existing concept pages:
{concept_briefs}

Decide how to update the wiki concept pages:

1. "create" — new concepts introduced by this document not covered by any existing page.
   Each item: {{"name": "slug-format", "title": "Human Readable Title"}}
   - Use kebab-case slugs
   - Create at most 3-5 foundational concepts
   - Do NOT create a concept that overlaps an existing one — use "update" instead

2. "update" — existing concepts that have significant new information worth integrating.
   Each item: {{"name": "existing-slug", "title": "Existing Title"}}

3. "related" — existing concept slugs tangentially related to this document.
   These get a cross-reference link only, no content rewrite.
   Array of slug strings.

Return a JSON object with keys "create", "update", "related"."""


def build_concept_create_prompt(
    title: str,
    doc_name: str,
    overview: str,
    known_targets: str,
) -> str:
    """Build prompt to generate a new concept page.

    Args:
        title: Human-readable concept title.
        doc_name: Source document stem name.
        overview: Prose overview of the source document.
        known_targets: Newline-separated list of valid [[wikilink]] targets.

    Returns:
        Prompt string for the wiki_concept agent (returns ConceptPage).
    """
    return f"""Write a concept page for: {title}

This concept appears in the document "{doc_name}".

Document overview:
{overview}

Valid [[wikilink]] targets (ONLY use these — do not invent other wikilinks):
{known_targets}

Return a JSON object with:
- "brief": One sentence (<100 chars) defining this concept
- "content": Full Markdown explanation with [[wikilinks]] to related concepts \
and [[summaries/{doc_name}]] as the source reference

Do not include YAML frontmatter in content."""


def build_concept_update_prompt(
    title: str,
    doc_name: str,
    overview: str,
    existing_content: str,
    known_targets: str,
) -> str:
    """Build prompt to rewrite an existing concept page with new information.

    Args:
        title: Human-readable concept title.
        doc_name: Source document stem name.
        overview: Prose overview of the source document.
        existing_content: Current Markdown body of the concept page (no frontmatter).
        known_targets: Newline-separated list of valid [[wikilink]] targets.

    Returns:
        Prompt string for the wiki_concept agent (returns ConceptPage).
    """
    return f"""Update the concept page for: {title}

New information comes from document "{doc_name}".

Document overview:
{overview}

Current page content:
{existing_content}

Valid [[wikilink]] targets (ONLY use these — do not invent other wikilinks):
{known_targets}

Rewrite the full page, integrating new information naturally. Preserve the existing \
structure and intent. Do not just append — merge into a coherent page.

Return a JSON object with:
- "brief": One sentence (<100 chars) defining this concept (may differ from before)
- "content": Full rewritten Markdown body

Do not include YAML frontmatter in content."""

from __future__ import annotations

from pydantic import BaseModel, Field
from openindex.models import SectionNode


class SourcePage(BaseModel):
    """A single page entry in the sources array of a WikiDict.

    Attributes:
        page: 1-based page number.
        content: Full page text.
        images: Reserved for future image support; always empty.
    """

    page: int
    content: str
    images: list = Field(default_factory=list)


class ConceptEntry(BaseModel):
    """A single generated concept page stored in a WikiDict.

    Attributes:
        brief: One-sentence definition.
        content: Full Markdown body.
    """

    brief: str
    content: str


class WikiDict(BaseModel):
    """In-memory wiki artifacts for a single document.

    Output of compile_wiki_to_dict(). Pass to save_wiki_dicts_to_dir()
    to write to disk, or build_unified_index() to generate a unified index.

    Attributes:
        doc_name: Document stem, e.g. "paper".
        description: One-paragraph document summary.
        summary: Full section tree as Markdown (summaries/<doc>.md content).
        sources: Per-page content array (sources/<doc>.json content).
        concepts: Generated concept pages keyed by slug.
        related: Concept slugs related to this doc (from LLM planner).
        index: index.md content for this document alone.
    """

    doc_name: str
    description: str = ""
    summary: str = ""
    sources: list[SourcePage] = Field(default_factory=list)
    concepts: dict[str, ConceptEntry] = Field(default_factory=dict)
    related: list[str] = Field(default_factory=list)
    index: str = ""


class BuildResult(BaseModel):
    """Result of WikiIndex.build_wiki() / build_wiki_sync().

    Attributes:
        title: document stem, e.g. "paper".
        doc_name: document filename, e.g. "paper.pdf".
        description: accumulated document description.
        nodes: nested section tree.
        pages: mapping of 1-based page index to page text.
        wiki: compiled wiki artifacts (always present after build_wiki).
    """

    title: str
    doc_name: str
    description: str = ""
    nodes: list[SectionNode] = Field(default_factory=list)
    pages: dict[int, str] = Field(default_factory=dict)
    wiki: WikiDict | None = None


class ConceptItem(BaseModel):
    """A single concept entry in the plan returned by the wiki planner.

    Attributes:
        name: URL-safe slug, e.g. 'attention-mechanism'.
        title: Human-readable label, e.g. 'Attention Mechanism'.
    """

    name: str = Field(description="URL-safe slug for the concept page filename.")
    title: str = Field(description="Human-readable concept title.")


class ConceptPlan(BaseModel):
    """Structured plan for updating wiki concept pages after indexing a document.

    Attributes:
        create: New concept pages to create (not covered by existing pages).
        update: Existing concept pages to rewrite with new information.
        related: Existing concept slugs to cross-link without content changes.
    """

    create: list[ConceptItem] = Field(default=[], description="New concepts to create.")
    update: list[ConceptItem] = Field(default=[], description="Existing concepts to update.")
    related: list[str] = Field(default=[], description="Existing slugs to cross-reference (code-only, no LLM).")


class ConceptPage(BaseModel):
    """Structured output for a generated or updated concept page.

    Attributes:
        brief: One sentence (<100 chars) defining this concept.
        content: Full Markdown body with [[wikilinks]].
    """

    brief: str = Field(description="One sentence definition, under 100 characters.")
    content: str = Field(description="Full Markdown body for the concept page.")

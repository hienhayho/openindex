from __future__ import annotations

from pydantic import BaseModel, Field


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

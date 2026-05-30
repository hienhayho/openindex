from __future__ import annotations

from pydantic import BaseModel, Field


class FlatSection(BaseModel):
    """A single section entry in the flat list produced by the LLM generator.

    The structure field uses dot notation to encode hierarchy (e.g. "1.2.3"),
    which is later used by tree_builder to reconstruct the nested tree.

    Attributes:
        structure: Hierarchical index in dot notation, e.g. '1', '1.2', '2.3.1'.
        title: Section heading as it appears in the document.
        physical_index: 1-based page number where this section starts.
    """

    structure: str = Field(description="Hierarchical index in dot notation, e.g. '1', '1.2', '2.3.1'.")
    title: str = Field(description="Section heading as it appears in the document.")
    physical_index: int = Field(description="1-based page number where this section starts.")


class FlatSectionList(BaseModel):
    """Structured output schema for the generator agent.

    Wraps a list of FlatSection so agno can return a validated object.

    Attributes:
        sections: Ordered list of detected sections.
    """

    sections: list[FlatSection] = Field(default=[])


class VerifyResult(BaseModel):
    """Structured output for the verifier agent.

    Used to confirm whether a section title actually appears on its stated page.

    Attributes:
        is_correct: True if the section title appears on the stated page.
        reason: Optional explanation for the decision.
    """

    is_correct: bool = Field(description="True if the section title appears on the stated page.")
    reason: str | None = Field(default=None)


class LocateResult(BaseModel):
    """Structured output for the locator agent.

    Used during the fix pass to find the correct page for a misplaced section.

    Attributes:
        is_correct: False if the title was not found in any of the searched pages.
        physical_index: Page where the section actually starts; 0 if not found.
    """

    is_correct: bool = Field(description="False if the title was not found in any of the searched pages.")
    physical_index: int = Field(default=0, description="Page where the section actually starts; 0 if not found.")


class SectionNode(BaseModel):
    """A node in the final hierarchical section tree.

    start_index and end_index are 1-based page numbers (inclusive).
    depth=0 for top-level sections, depth=1 for their children, etc.
    summary is populated by the summarizer phase.

    Attributes:
        title: Section heading.
        start_index: 1-based page number where this section starts (inclusive).
        end_index: 1-based page number where this section ends (inclusive).
        depth: Nesting depth; 0 = top-level, 1 = child, etc.
        summary: LLM-generated summary of this section's content.
        children: Nested child sections.
    """

    title: str
    start_index: int
    end_index: int
    depth: int = 0
    summary: str = ""
    children: list[SectionNode] = Field(default=[])

    def to_dict(self) -> dict:
        """Serialize node to a plain dict for JSON output.

        Returns:
            dict with title, start_index, end_index, depth, children,
            and summary (only if non-empty).
        """
        d = {
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children],
        }
        if self.summary:
            d["summary"] = self.summary
        return d

    def to_tree(self, indent: int = 0) -> str:
        """Render node and its descendants as an indented text tree.

        Args:
            indent: current indentation level (spaces = indent * 2).

        Returns:
            Multi-line string representation of the subtree.
        """
        prefix = "  " * indent
        line = f"{prefix}{'#' * (self.depth + 1)} {self.title} (pages {self.start_index}–{self.end_index})"
        lines = [line]
        for child in self.children:
            lines.append(child.to_tree(indent + 1))
        return "\n".join(lines)


SectionNode.model_rebuild()

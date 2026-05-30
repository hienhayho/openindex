from openindex.models import FlatSection, SectionNode
from openindex.utils.logger import get_logger

logger = get_logger(__name__)


def flat_to_tree(
    sections: list[FlatSection],
    total_pages: int,
    start_index: int = 1,
) -> list[SectionNode]:
    """Convert a flat ordered section list into a nested SectionNode tree.

    end_index for each section is derived from the next sibling's start minus one,
    then corrected upward to cover all descendants.

    Args:
        sections: flat list from generate_flat_sections, ordered by physical_index.
        total_pages: last page number in the document (used as end_index for the final section).
        start_index: 1-based page number of the first page.

    Returns:
        List of root SectionNode with children nested by dot-notation structure.
    """
    if not sections:
        return []

    # Compute initial end_index from next sibling's start
    items: list[dict] = []
    for i, s in enumerate(sections):
        end = sections[i + 1].physical_index - 1 if i + 1 < len(sections) else total_pages
        end = max(s.physical_index, end)
        items.append({
            "structure": s.structure,
            "title": s.title,
            "start_index": s.physical_index,
            "end_index": end,
        })

    # Build tree using dot-notation structure numbers
    nodes: dict[str, SectionNode] = {}
    roots: list[SectionNode] = []

    for item in items:
        structure = item["structure"]
        depth = len(structure.split(".")) - 1
        node = SectionNode(
            title=item["title"],
            start_index=item["start_index"],
            end_index=item["end_index"],
            depth=depth,
        )
        nodes[structure] = node

        parent_key = _parent_structure(structure)
        if parent_key and parent_key in nodes:
            nodes[parent_key].children.append(node)
        else:
            roots.append(node)

    _fix_end_indices(roots)
    return roots


def _fix_end_indices(nodes: list[SectionNode]) -> int:
    """Recursively expand each node's end_index to cover all its descendants.

    Args:
        nodes: list of sibling nodes at the current level.

    Returns:
        Maximum end_index seen across all nodes and their descendants.
    """
    max_end = 0
    for node in nodes:
        if node.children:
            child_max = _fix_end_indices(node.children)
            node.end_index = max(node.end_index, child_max)
        max_end = max(max_end, node.end_index)
    return max_end


def _parent_structure(structure: str) -> str | None:
    """Return the parent dot-notation key for a structure string, or None if root.

    Args:
        structure: e.g. "1.2.3"

    Returns:
        "1.2" for "1.2.3", None for "1".
    """
    parts = structure.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def add_preface_if_needed(
    sections: list[FlatSection],
    start_index: int = 1,
) -> list[FlatSection]:
    """Prepend a Preface node if the first section doesn't start on the first page.

    Args:
        sections: flat section list.
        start_index: 1-based first page number.

    Returns:
        sections unchanged, or with a Preface prepended.
    """
    if not sections:
        return sections
    if sections[0].physical_index > start_index:
        preface = FlatSection(structure="0", title="Preface", physical_index=start_index)
        return [preface] + sections
    return sections

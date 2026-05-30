from tqdm.asyncio import tqdm

from openindex.agents import AgentPool
from openindex.config import TreeConfig
from openindex.models import SectionNode
from openindex.pipeline.generator import generate_flat_sections
from openindex.pipeline.tree_builder import flat_to_tree
from openindex.pipeline.verifier import verify_and_fix
from openindex.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_DEPTH = 5  # hard recursion cap to prevent runaway expansion


async def expand_large_nodes(
    nodes: list[SectionNode],
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int = 1,
    depth: int = 1,
) -> list[SectionNode]:
    """Recursively expand nodes that exceed the page/token size thresholds.

    Only leaf nodes are considered for expansion. After expanding a level,
    recurses into all children (newly expanded or pre-existing).

    Args:
        nodes: current level of nodes to inspect.
        page_texts: full document page texts.
        pool: agent pool.
        config: controls max_pages_per_node and max_tokens_per_node thresholds.
        start_index: 1-based first page number.
        depth: current recursion depth (stops at _MAX_DEPTH).

    Returns:
        nodes with large leaf nodes replaced by expanded versions.
    """
    if depth > _MAX_DEPTH:
        return nodes

    large = [(i, n) for i, n in enumerate(nodes) if _is_large(n, page_texts, config, pool, start_index)]
    if not large:
        return nodes

    expanded = await tqdm.gather(
        *[_expand_node(n, page_texts, pool, config, start_index, depth) for _, n in large],
        desc=f"Expanding large nodes [depth={depth}]",
        unit="node",
    )

    for (i, _), new_node in zip(large, expanded):
        nodes[i] = new_node

    # Recurse into all children at next depth
    for node in nodes:
        if node.children:
            node.children = await expand_large_nodes(
                node.children, page_texts, pool, config, start_index, depth + 1
            )

    return nodes


async def _expand_node(
    node: SectionNode,
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int,
    depth: int,
) -> SectionNode:
    """Run the full generate→verify→tree pipeline on a single large node's page range.

    Args:
        node: the node to expand.
        page_texts: full document page texts.
        pool: agent pool.
        config: tree config.
        start_index: 1-based first page number.
        depth: current depth (for logging).

    Returns:
        node with children populated, or the original node if expansion yields nothing useful.
    """
    s = node.start_index - start_index
    e = node.end_index - start_index + 1
    scoped_texts = page_texts[s:e]
    if not scoped_texts:
        return node

    sections = await generate_flat_sections(scoped_texts, pool, config, start_index=node.start_index)
    if not sections:
        return node

    sections = await verify_and_fix(sections, page_texts, pool, config, start_index)
    children = flat_to_tree(sections, node.end_index, start_index)

    # Discard expansion if it only mirrors the node itself
    if not children or (len(children) == 1 and children[0].title == node.title):
        return node

    node.children = children
    return node


def _is_large(
    node: SectionNode,
    page_texts: list[str],
    config: TreeConfig,
    pool: AgentPool,
    start_index: int,
) -> bool:
    """Return True if a leaf node exceeds both page and token thresholds.

    Args:
        node: node to evaluate.
        page_texts: full document page texts.
        config: thresholds.
        start_index: 1-based first page number.

    Returns:
        True if the node should be expanded.
    """
    if node.children:
        return False
    page_span = node.end_index - node.start_index + 1
    if page_span <= config.max_pages_per_node:
        return False
    s = node.start_index - start_index
    e = node.end_index - start_index + 1
    try:
        import litellm
        litellm_model = f"openai/{pool.model_name}"
        tokens = sum(litellm.token_counter(model=litellm_model, text=t) for t in page_texts[s:e])
    except Exception:
        tokens = sum(len(t.split()) * 2 for t in page_texts[s:e])
    return tokens >= config.max_tokens_per_node

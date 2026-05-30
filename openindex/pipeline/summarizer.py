from tqdm.asyncio import tqdm

from openindex.agents import AgentPool, run_text
from openindex.config import TreeConfig
from openindex.models import SectionNode
from openindex.prompts.generator import build_accumulate_description_prompt, build_summary_prompt
from openindex.utils.logger import get_logger

logger = get_logger(__name__)


def _collect_nodes(nodes: list[SectionNode]) -> list[SectionNode]:
    """Flatten a nested node tree into a single list (depth-first).

    Args:
        nodes: top-level nodes of the tree.

    Returns:
        All nodes including descendants, in depth-first order.
    """
    result = []
    for node in nodes:
        result.append(node)
        if node.children:
            result.extend(_collect_nodes(node.children))
    return result


async def add_summaries(
    nodes: list[SectionNode],
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int = 1,
) -> list[SectionNode]:
    """Generate and attach summaries to all nodes concurrently.

    Each node's summary is produced from its page text slice (capped at 8000 chars).
    Summaries are written directly onto node.summary in place.

    Args:
        nodes: root nodes of the section tree.
        page_texts: raw text per page.
        pool: agent pool with summarizer agent.
        config: tree config (unused directly, kept for interface consistency).
        start_index: 1-based page number of the first page.

    Returns:
        nodes with summary fields populated.
    """
    all_nodes = _collect_nodes(nodes)

    async def summarize(node: SectionNode) -> None:
        s = node.start_index - start_index
        e = node.end_index - start_index + 1
        text = " ".join(page_texts[s:e])
        prompt = build_summary_prompt(node.title, text[:8000])
        node.summary = await run_text(pool.summarizer, prompt, pool.sem)

    await tqdm.gather(
        *[summarize(n) for n in all_nodes],
        desc="Generating summaries",
        unit="node",
    )
    return nodes


async def generate_doc_description(
    nodes: list[SectionNode],
    pool: AgentPool,
) -> str:
    """Build a cumulative document description by chaining top-level node summaries.

    Each call sees the accumulated description so far plus the new section summary,
    allowing the LLM to progressively refine the description as it reads the document.

    Args:
        nodes: top-level root nodes (only these are used, not children).
        pool: agent pool with summarizer agent.

    Returns:
        Final accumulated description string (up to 5 sentences).
    """
    description = ""
    for node in nodes:
        if not node.summary:
            continue
        prompt = build_accumulate_description_prompt(description, node.title, node.summary)
        description = await run_text(pool.summarizer, prompt, pool.sem)
    return description

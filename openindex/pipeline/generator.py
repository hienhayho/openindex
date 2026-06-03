import math

from tqdm.asyncio import tqdm

from openindex.agents import AgentPool, run_structured
from openindex.config import TreeConfig
from openindex.models import FlatSection, FlatSectionList
from openindex.prompts.generator import build_generate_prompt
from openindex.utils.logger import get_logger

logger = get_logger(__name__)


def tag_pages(page_texts: list[str], start_index: int = 1) -> list[str]:
    """Wrap each page in physical_index tags for LLM page-number grounding.

    Args:
        page_texts: raw text of each page.
        start_index: 1-based page number of the first page.

    Returns:
        List of strings with format `<physical_index_N>\\ntext\\n<physical_index_N>`.
    """
    tagged = []
    for i, text in enumerate(page_texts):
        n = start_index + i
        tagged.append(f"<physical_index_{n}>\n{text}\n<physical_index_{n}>")
    return tagged


def group_pages(
    tagged_pages: list[str],
    token_counts: list[int],
    max_tokens: int,
    overlap: int = 1,
) -> list[str]:
    """Split tagged pages into token-budget groups with overlap for context continuity.

    If total tokens fit within max_tokens, returns a single group.
    Otherwise splits into roughly equal parts, each starting with `overlap`
    pages from the end of the previous group.

    Args:
        tagged_pages: output of tag_pages().
        token_counts: token count per page (same length as tagged_pages).
        max_tokens: maximum tokens per group.
        overlap: number of pages to repeat at the start of each new group.

    Returns:
        List of joined page strings, one per group.
    """
    total = sum(token_counts)
    if total <= max_tokens:
        return ["\n\n".join(tagged_pages)]

    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    expected_parts = math.ceil(total / max_tokens)
    target = math.ceil((total / expected_parts + max_tokens) / 2)

    for i, (page, tokens) in enumerate(zip(tagged_pages, token_counts)):
        if current_tokens + tokens > target and current:
            groups.append("\n\n".join(current))
            overlap_start = max(i - overlap, 0)
            current = tagged_pages[overlap_start:i]
            current_tokens = sum(token_counts[overlap_start:i])
        current.append(page)
        current_tokens += tokens

    if current:
        groups.append("\n\n".join(current))

    return groups


async def generate_flat_sections(
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int = 1,
    previous_sections: list[dict] | None = None,
) -> list[FlatSection]:
    """Generate a flat ordered list of sections by scanning all pages.

    Pages are tagged, grouped by token budget, then each group is sent to
    the generator LLM. Results are accumulated in order, with each group
    receiving the previous sections as context to avoid duplication.

    Args:
        page_texts: raw text per page.
        pool: shared agent pool with generator agent and semaphore.
        config: controls max_tokens_per_group and page_overlap.
        start_index: 1-based page number of the first page.
        previous_sections: sections already detected before this range (for continuation).

    Returns:
        Ordered list of FlatSection with structure, title, and physical_index.
    """
    try:
        import litellm
        litellm_model = f"openai/{pool.model_name}"
        token_counts = [litellm.token_counter(model=litellm_model, text=t) for t in page_texts]
    except Exception:
        token_counts = [len(t.split()) * 2 for t in page_texts]

    tagged = tag_pages(page_texts, start_index)
    groups = group_pages(tagged, token_counts, config.max_tokens_per_group, config.page_overlap)

    num_groups = len(groups)
    per_group = math.ceil(config.max_sections / num_groups) if config.max_sections > 0 else 0

    all_sections: list[FlatSection] = []
    prev = list(previous_sections) if previous_sections else None

    for group_text in await tqdm.gather(
        *[_generate_group(group, pool, prev, per_group) for group in groups],
        desc="Generating structure",
        unit="group",
    ):
        all_sections.extend(group_text)
        if group_text:
            prev = [
                {"structure": s.structure, "title": s.title, "physical_index": s.physical_index}
                for s in all_sections
            ]

    return all_sections


async def _generate_group(
    group_text: str,
    pool: AgentPool,
    previous_sections: list[dict] | None,
    max_sections: int = 0,
) -> list[FlatSection]:
    """Send one page group to the generator LLM and return detected sections.

    Args:
        group_text: joined tagged pages for this group.
        pool: agent pool.
        previous_sections: already-detected sections passed as context.
        max_sections: per-group section cap passed to the prompt. 0 = unlimited.

    Returns:
        List of FlatSection detected in this group.
    """
    prompt = build_generate_prompt(group_text, previous_sections, max_sections)
    result: FlatSectionList = await run_structured(pool.generator, prompt, pool.sem)
    return result.sections if result else []

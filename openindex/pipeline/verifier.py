import re

from tqdm.asyncio import tqdm

from openindex.agents import AgentPool, run_structured
from openindex.config import TreeConfig
from openindex.models import FlatSection, LocateResult, VerifyResult
from openindex.prompts.generator import build_locate_prompt, build_verify_prompt
from openindex.utils.logger import get_logger

logger = get_logger(__name__)

async def verify_and_fix(
    sections: list[FlatSection],
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int = 1,
) -> list[FlatSection]:
    """Verify all sections appear on their stated page; fix incorrect ones iteratively.

    Uses LLM to verify, then focused LLM locate calls to fix, then fast string
    check to confirm. Repeats up to config.max_fix_attempts times.

    Args:
        sections: flat section list with physical_index assignments.
        page_texts: raw text per page.
        pool: agent pool with verifier and locator agents.
        config: tree config (controls max_fix_attempts and fix_search_radius).
        start_index: 1-based page number of the first page.

    Returns:
        sections with corrected physical_index where possible.
    """
    incorrect = await _verify_all(sections, page_texts, pool, start_index)
    if not incorrect:
        return sections

    logger.info("verify_incorrect", count=len(incorrect))
    for attempt in range(config.max_fix_attempts):
        if not incorrect:
            break
        sections, incorrect = await _fix_pass(sections, incorrect, page_texts, pool, config, start_index)
        logger.info("fix_attempt", attempt=attempt + 1, remaining=len(incorrect))

    return sections


async def _verify_all(
    sections: list[FlatSection],
    page_texts: list[str],
    pool: AgentPool,
    start_index: int,
) -> list[int]:
    """Run LLM verification on all sections concurrently.

    Args:
        sections: sections to verify.
        page_texts: raw text per page.
        pool: agent pool.
        start_index: 1-based first page number.

    Returns:
        List of indices into sections that failed verification.
    """
    async def check(i: int) -> int | None:
        s = sections[i]
        page_idx = s.physical_index - start_index
        if page_idx < 0 or page_idx >= len(page_texts):
            return i
        prompt = build_verify_prompt(s.title, page_texts[page_idx])
        result: VerifyResult = await run_structured(pool.verifier, prompt, pool.sem)
        return None if (result and result.is_correct) else i

    results = await tqdm.gather(
        *[check(i) for i in range(len(sections))],
        desc="Verifying sections",
        unit="section",
    )
    return [i for i in results if i is not None]


async def _fix_pass(
    sections: list[FlatSection],
    incorrect_indices: list[int],
    page_texts: list[str],
    pool: AgentPool,
    config: TreeConfig,
    start_index: int,
) -> tuple[list[FlatSection], list[int]]:
    """Attempt to relocate incorrect sections using focused LLM locate calls.

    Search window is bounded by the nearest correct neighbours on each side,
    further clamped to ±config.fix_search_radius pages from the stated page.
    After fixing, re-verifies using fast string check (no LLM).

    Args:
        sections: full section list (mutated in place for fixed entries).
        incorrect_indices: indices of sections that failed verification.
        page_texts: raw text per page.
        pool: agent pool with locator agent.
        config: tree config (provides fix_search_radius).
        start_index: 1-based first page number.

    Returns:
        Tuple of (updated sections, indices still incorrect after fix).
    """
    from openindex.pipeline.generator import tag_pages

    incorrect_set = set(incorrect_indices)

    async def fix_one(i: int) -> FlatSection:
        s = sections[i]

        # Search window: between nearest correct neighbours, clamped by radius
        prev_page = next(
            (sections[j].physical_index for j in range(i - 1, -1, -1) if j not in incorrect_set),
            start_index,
        )
        next_page = next(
            (sections[j].physical_index for j in range(i + 1, len(sections)) if j not in incorrect_set),
            len(page_texts) + start_index - 1,
        )
        search_start = max(prev_page, s.physical_index - config.fix_search_radius, start_index)
        search_end = min(next_page, s.physical_index + config.fix_search_radius, len(page_texts) + start_index - 1)

        search_pages = page_texts[search_start - start_index: search_end - start_index + 1]
        if not search_pages:
            return s

        tagged = "\n\n".join(tag_pages(search_pages, search_start))
        prompt = build_locate_prompt(s.title, tagged)
        result: LocateResult = await run_structured(pool.locator, prompt, pool.sem)

        if result and result.is_correct and result.physical_index > 0:
            logger.info("fix_relocated", title=s.title, old=s.physical_index, new=result.physical_index)
            return FlatSection(structure=s.structure, title=s.title, physical_index=result.physical_index)
        return s

    fixed = await tqdm.gather(
        *[fix_one(i) for i in incorrect_indices],
        desc="Fixing sections",
        unit="section",
    )
    for idx, new_section in zip(incorrect_indices, fixed):
        sections[idx] = new_section

    # Re-verify with fast string check — no LLM round
    still_wrong = [
        incorrect_indices[i]
        for i, idx in enumerate(incorrect_indices)
        if not _string_check(sections[idx], page_texts, start_index)
    ]
    return sections, still_wrong


def _string_check(s: FlatSection, page_texts: list[str], start_index: int) -> bool:
    """Check if a section title appears (normalized) in its stated page text.

    Args:
        s: section to check.
        page_texts: raw text per page.
        start_index: 1-based first page number.

    Returns:
        True if title found in page text after whitespace normalization.
    """
    page_idx = s.physical_index - start_index
    if page_idx < 0 or page_idx >= len(page_texts):
        return False
    return _normalize(s.title) in _normalize(page_texts[page_idx])


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for fuzzy string matching.

    Args:
        text: input string.

    Returns:
        Normalized string with single spaces, stripped, lowercased.
    """
    return re.sub(r"\s+", " ", text).strip().lower()

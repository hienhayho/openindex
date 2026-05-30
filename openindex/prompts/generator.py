def build_generate_prompt(tagged_text: str, previous_sections: list[dict] | None = None) -> str:
    """Build the prompt for the section structure generator.

    Args:
        tagged_text: page text wrapped in <physical_index_X> tags.
        previous_sections: already-detected sections from prior groups,
            passed as context to avoid duplication.

    Returns:
        Prompt string for the generator agent.
    """
    prev_block = ""
    if previous_sections:
        import json
        prev_block = f"""Previously detected sections (do NOT repeat these, only continue from where they left off):
{json.dumps(previous_sections, ensure_ascii=False, indent=2)}

"""

    return f"""{prev_block}--- DOCUMENT PAGES ---
{tagged_text}

---

You are extracting the hierarchical section structure of this document.

Pages are wrapped in <physical_index_X>...</physical_index_X> tags showing their page number.

Extract ALL sections and subsections. For each section:
- structure: hierarchical index using dot notation ("1", "1.1", "1.2", "2", "2.1.3", etc.)
- title: exact heading text from the document (fix spacing inconsistencies only)
- physical_index: the page number where this section STARTS (use the <physical_index_X> number)

Rules:
- Detect ALL levels of headings: chapters, sections, subsections, numbered items that introduce distinct topics
- Use the <physical_index_X> tags to determine the correct page number for each section start
- For documents without explicit headings, derive a short title (2-6 words) from topic shifts
- Always include conclusion sections, summaries, and appendices
- If this is a continuation (previous sections provided), start structure numbering after the last previous section

Return an empty sections list only if the text contains no section boundaries at all."""


def build_verify_prompt(title: str, page_text: str) -> str:
    """Build the prompt for checking if a section starts on a given page.

    Args:
        title: section heading to look for.
        page_text: raw text of the page to check.

    Returns:
        Prompt string for the verifier agent.
    """
    return f"""Does the section titled "{title}" start on this page?

Page content:
{page_text}

Check if this section title (or very close variant) appears on the page and marks the beginning of that section.
Use fuzzy matching — ignore minor spacing or punctuation differences."""


def build_locate_prompt(title: str, tagged_text: str) -> str:
    """Build the prompt for locating which page a section actually starts on.

    Used during the fix pass when verification fails.

    Args:
        title: section heading to find.
        tagged_text: pages wrapped in <physical_index_X> tags to search within.

    Returns:
        Prompt string for the locator agent.
    """
    return f"""Find the page where the section "{title}" starts.

--- DOCUMENT PAGES ---
{tagged_text}

---

Pages are wrapped in <physical_index_X>...</physical_index_X> tags.

Look for the heading "{title}" (fuzzy match — ignore minor spacing/punctuation differences).
Return physical_index = the page number where this section heading appears.
If not found on any of these pages, return is_correct = false."""


def build_accumulate_description_prompt(current_description: str, section_title: str, section_summary: str) -> str:
    """Build the prompt for updating the cumulative document description.

    On the first call (no prior description), produces an initial description.
    On subsequent calls, refines the existing description with the new section.

    Args:
        current_description: description accumulated so far (empty on first call).
        section_title: title of the current section being incorporated.
        section_summary: summary of the current section.

    Returns:
        Prompt string for the summarizer agent.
    """
    if current_description:
        return f"""You are building a cumulative description of a document by reading it section by section.

Current description so far:
{current_description}

New section: "{section_title}"
Section summary:
{section_summary}

Update the description to incorporate this new section. Keep it concise (up to 5 sentences) and capture the document's overall scope so far.
Return only the updated description, nothing else."""
    else:
        return f"""You are building a cumulative description of a document by reading it section by section.

First section: "{section_title}"
Section summary:
{section_summary}

Write a concise description (up to 5 sentences) of what this document covers based on this first section.
Return only the description, nothing else."""


def build_summary_prompt(title: str, page_text: str) -> str:
    """Build the prompt for summarizing a single section.

    Args:
        title: section heading.
        page_text: text content of the section's pages.

    Returns:
        Prompt string for the summarizer agent.
    """
    return f"""Summarize the main points of the section "{title}":

{page_text}

Write a concise summary (2-4 sentences) of what this section covers. Return only the summary text."""

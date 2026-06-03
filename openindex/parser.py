from __future__ import annotations

from liteparse import LiteParse


def load_pages(pdf_path: str) -> list:
    """Parse a PDF and return its page objects.

    Args:
        pdf_path: path to the PDF file.

    Returns:
        List of liteparse page objects (each has .text or .text_items).
    """
    return LiteParse().parse(pdf_path).pages


def page_to_text(page) -> str:
    """Extract plain text from a liteparse page object.

    Handles both the .text shortcut and the .text_items fallback
    for pages where text is stored as a list of items.

    Args:
        page: liteparse page object.

    Returns:
        Full text of the page as a single string.
    """
    return page.text if hasattr(page, "text") else " ".join(item.text for item in page.text_items)


def split_text_to_pages(text: str, words_per_page: int = 500) -> list[str]:
    """Split a plain text or Markdown string into page-sized chunks.

    Splits on word count but always extends the boundary forward to the next
    newline, so sentences and lines are never broken mid-way.

    Args:
        text: input text (plain text or Markdown).
        words_per_page: target word count per chunk. Actual size may be slightly
            larger when the next newline falls after the word boundary.

    Returns:
        List of non-empty text chunks, one per "page".
    """
    pages: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        # find approximate end by word count
        words = 0
        pos = start
        while pos < n and words < words_per_page:
            if text[pos].isspace():
                while pos < n and text[pos].isspace():
                    pos += 1
            else:
                while pos < n and not text[pos].isspace():
                    pos += 1
                words += 1

        if pos >= n:
            # reached end of text
            chunk = text[start:].strip()
            if chunk:
                pages.append(chunk)
            break

        # extend forward to next newline so we don't break mid-sentence
        newline_pos = text.find("\n", pos)
        end = (newline_pos + 1) if newline_pos != -1 else n

        chunk = text[start:end].strip()
        if chunk:
            pages.append(chunk)
        start = end

    return pages


def load_page_images(pdf_path: str) -> list[bytes]:
    """Render each PDF page as a PNG screenshot and return image bytes.

    Args:
        pdf_path: path to the PDF file.

    Returns:
        List of PNG bytes, one per page, in page order.
    """
    screenshots = LiteParse().screenshot(pdf_path)
    return [s.image_bytes for s in screenshots]

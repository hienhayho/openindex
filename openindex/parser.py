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


def load_page_images(pdf_path: str) -> list[bytes]:
    """Render each PDF page as a PNG screenshot and return image bytes.

    Args:
        pdf_path: path to the PDF file.

    Returns:
        List of PNG bytes, one per page, in page order.
    """
    screenshots = LiteParse().screenshot(pdf_path)
    return [s.image_bytes for s in screenshots]

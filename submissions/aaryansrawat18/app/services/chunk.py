"""Cut page text into overlapping windows.

Each window is 800 characters. The next window starts 680 characters later,
so the two windows share 120 characters. Pages are cut separately. A chunk
keeps the page number it came from, which is what a citation shows later.
"""

WINDOW_SIZE = 800
OVERLAP_SIZE = 120


def split_text_into_windows(text: str) -> list[str]:
    """Split one page. Short text stays one window. Empty text returns nothing."""
    if text == "":
        return []
    if len(text) <= WINDOW_SIZE:
        return [text]

    # How far to move the start of the next window. Must stay positive.
    step = WINDOW_SIZE - OVERLAP_SIZE
    if step <= 0:
        raise ValueError("Overlap must be smaller than the window size.")

    windows = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + WINDOW_SIZE])
        # The last window already reaches the end of the page.
        if start + WINDOW_SIZE >= len(text):
            break
        start += step
    return windows


def build_chunks_for_document(
    pages: list[dict],
    document_name: str,
    user_id: str,
    document_id: str,
) -> list[dict]:
    """Build chunk dicts for every page, in order.

    `chunk_index` starts at 0 and increases across the whole document.
    `page` stays the page that window was cut from.
    """
    chunks = []
    next_chunk_index = 0
    for page in pages:
        page_number = page["page"]
        for content in split_text_into_windows(page["text"]):
            chunks.append(
                {
                    "document_name": document_name,
                    "page": page_number,
                    "chunk_index": next_chunk_index,
                    "content": content,
                    "user_id": user_id,
                    "document_id": document_id,
                }
            )
            next_chunk_index += 1
    return chunks

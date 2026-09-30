"""Checks for the 800-character windows with 120-character overlap."""

from app.services.chunk import (
    OVERLAP_SIZE,
    WINDOW_SIZE,
    build_chunks_for_document,
    split_text_into_windows,
)


def test_long_text_overlaps_and_indexes_increase():
    page_text = "policy " * 200
    assert len(page_text) > WINDOW_SIZE

    windows = split_text_into_windows(page_text)
    assert len(windows) > 1
    # The start of the next window repeats the end of the previous one.
    assert windows[1][:OVERLAP_SIZE] == windows[0][-OVERLAP_SIZE:]

    chunks = build_chunks_for_document(
        pages=[{"page": 3, "text": page_text}],
        document_name="handbook.pdf",
        user_id="user-1",
        document_id="doc-1",
    )
    chunk_indexes = [chunk["chunk_index"] for chunk in chunks]
    assert chunk_indexes == list(range(len(chunks)))
    assert all(chunk["page"] == 3 for chunk in chunks)


def test_short_text_is_one_chunk():
    chunks = build_chunks_for_document(
        pages=[{"page": 1, "text": "Short policy."}],
        document_name="note.txt",
        user_id="user-1",
        document_id="doc-1",
    )
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short policy."
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["page"] == 1


def test_page_numbers_stay_with_their_page():
    chunks = build_chunks_for_document(
        pages=[
            {"page": 1, "text": "Page one text."},
            {"page": 4, "text": "Page four text."},
        ],
        document_name="handbook.pdf",
        user_id="user-1",
        document_id="doc-1",
    )
    assert [(chunk["page"], chunk["content"], chunk["chunk_index"]) for chunk in chunks] == [
        (1, "Page one text.", 0),
        (4, "Page four text.", 1),
    ]

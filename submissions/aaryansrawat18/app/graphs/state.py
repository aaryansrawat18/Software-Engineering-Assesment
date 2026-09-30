"""In-memory state for one ingest run.

This is not a database table. It lives only while `graph.invoke` is running.
"""

from typing import TypedDict


class PageText(TypedDict):
    """Text taken from one page. Text files use page 1."""

    page: int
    text: str


class ChunkToSave(TypedDict, total=False):
    """One window of a page. The embed step adds `embedding`."""

    document_name: str
    page: int
    chunk_index: int
    content: str
    user_id: str
    document_id: str
    embedding: list[float]


class IngestState(TypedDict, total=False):
    """Values passed from one graph step to the next.

    Each step returns only the keys it changes. `error` is either None,
    `extract_failed`, or `embed_failed`.
    """

    document_id: str
    user_id: str
    filename: str
    pages: list[PageText]
    chunks: list[ChunkToSave]
    error: str | None

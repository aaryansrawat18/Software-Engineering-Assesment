"""In-memory state for one graph run.

This is not a database table. It lives only while `graph.invoke` is running.
The ingest graph and the answer graph each have their own state type.
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


class RetrievedChunk(TypedDict):
    """One passage brought back for a question, plus how close it was."""

    chunk_id: str
    content: str
    document_name: str
    page: int
    chunk_index: int
    similarity: float


class Citation(TypedDict):
    """A passage the answer is allowed to name. The chunk id is not shown to the client."""

    document_name: str
    passage: str
    page: int


class AnswerState(TypedDict, total=False):
    """Values passed from one answer-graph step to the next.

    The router fills `user_id`, `question`, `document_ids`, and `started_at`.
    Each later step returns only the keys it changes.
    """

    user_id: str
    question: str
    document_ids: list[str]
    started_at: float
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    refused: bool
    model_chunk_ids: list[str]
    citations: list[Citation]
    input_tokens: int
    output_tokens: int
    tokens: int
    estimated_cost_usd: float
    latency_ms: int
    question_id: str

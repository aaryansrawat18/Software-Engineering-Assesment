"""Find the closest ready passages for one question.

The question is embedded with the same MiniLM model used when the document
was saved. Postgres then orders this user's chunks by cosine distance.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.services.embed import VECTOR_SIZE, embed_texts

# The answer step sees this many passages, not the whole file.
MAX_PASSAGES = 4


def embed_question(question: str) -> list[float]:
    """Turn the question into one 384-number vector."""
    vectors = embed_texts([question])
    if len(vectors) != 1 or len(vectors[0]) != VECTOR_SIZE:
        raise ValueError("Question embedding must be one vector of length 384.")
    return vectors[0]


def search_ready_chunks(
    session: Session,
    user_id: uuid.UUID,
    query_vector: list[float],
    document_ids: list[uuid.UUID],
) -> list[dict]:
    """Return up to 4 passages from this user's ready documents.

    `document_ids` is the scope already checked by the router. An empty list
    means there is nothing to search, so this returns no rows.

    Similarity is `1 - cosine distance`. MiniLM vectors are normalized, so
    1.0 is the same meaning and 0.0 is unrelated. The nearest passage is first.
    """
    if not document_ids:
        return []

    # `<=>` is pgvector cosine distance. Smaller distance means a closer passage.
    cosine_distance = Chunk.embedding.cosine_distance(query_vector)
    similarity = (1 - cosine_distance).label("similarity")
    statement = (
        select(
            Chunk.id,
            Chunk.content,
            Chunk.document_name,
            Chunk.page,
            Chunk.chunk_index,
            similarity,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.user_id == user_id)
        .where(Document.status == "ready")
        .where(Document.id.in_(document_ids))
        .order_by(cosine_distance)
        .limit(MAX_PASSAGES)
    )
    passages = []
    for row in session.execute(statement):
        passages.append(
            {
                "chunk_id": str(row.id),
                "content": row.content,
                "document_name": row.document_name,
                "page": row.page,
                "chunk_index": row.chunk_index,
                "similarity": float(row.similarity),
            }
        )
    return passages

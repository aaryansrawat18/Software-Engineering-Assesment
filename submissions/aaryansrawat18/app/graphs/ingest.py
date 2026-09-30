"""Ingest graph: extract, chunk, embed, then save or mark failed.

The worker calls `graph.invoke({"document_id": ...})` once. There is no
checkpointer. The steps are a fixed path, not an agent picking tools.

    extract --pages--> chunk --> embed --vectors--> persistReady --> END
    extract --extract_failed--> markFailed --> END
    embed --embed_failed--> markFailed --> END
"""

import logging
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.graphs.state import IngestState
from app.models import Chunk, Document
from app.services.chunk import build_chunks_for_document
from app.services.embed import VECTOR_SIZE, embed_texts
from app.services.extract import ExtractFailed, read_pages
from app.services.storage import document_file_path

logger = logging.getLogger(__name__)

EXTRACT_FAILED = "extract_failed"
EMBED_FAILED = "embed_failed"


def _save_failure(session: Session, document: Document, reason: str) -> None:
    """Store a short reason only. Never store a traceback or file text."""
    if reason not in {EXTRACT_FAILED, EMBED_FAILED}:
        reason = EXTRACT_FAILED
    document.status = "failed"
    document.error = reason
    session.commit()


def extract_node(state: IngestState) -> dict:
    """Read the file for this document id and pull out page text."""
    document_id = state["document_id"]
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        return {"error": EXTRACT_FAILED}

    with SessionLocal() as session:
        document = session.get(Document, parsed_id)
        if document is None:
            return {"error": EXTRACT_FAILED}
        filename = document.filename
        media_type = document.media_type
        user_id = str(document.user_id)

    file_path = document_file_path(document_id)
    try:
        file_bytes = file_path.read_bytes()
        pages = read_pages(file_bytes, media_type)
    except (OSError, ExtractFailed):
        logger.info("extract failed", extra={"document_id": document_id})
        return {"error": EXTRACT_FAILED}

    if not pages:
        logger.info("extract failed", extra={"document_id": document_id})
        return {"error": EXTRACT_FAILED}

    return {
        "filename": filename,
        "user_id": user_id,
        "pages": pages,
        "error": None,
    }


def chunk_node(state: IngestState) -> dict:
    """Cut the extracted pages into 800-character windows with 120 overlap."""
    chunks = build_chunks_for_document(
        pages=state.get("pages", []),
        document_name=state.get("filename", ""),
        user_id=state.get("user_id", ""),
        document_id=state["document_id"],
    )
    return {"chunks": chunks, "error": None}


def route_after_extract(state: IngestState) -> Literal["chunk", "markFailed"]:
    """Leave extract toward chunk, or toward markFailed when reading failed."""
    if state.get("error") == EXTRACT_FAILED:
        return "markFailed"
    return "chunk"


def route_after_embed(state: IngestState) -> Literal["persistReady", "markFailed"]:
    """Leave embed toward persistReady, or toward markFailed when vectors failed."""
    if state.get("error") == EMBED_FAILED:
        return "markFailed"
    return "persistReady"


def persist_ready_node(state: IngestState) -> dict:
    """Replace this document's chunks and set status to ready, in one transaction.

    Status stays off `ready` until the commit succeeds. Re-running is safe
    because the old chunks for this document are deleted first.
    """
    document_id = uuid.UUID(state["document_id"])
    with SessionLocal() as session:
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        for chunk in state.get("chunks", []):
            session.add(
                Chunk(
                    document_id=document_id,
                    user_id=uuid.UUID(chunk["user_id"]),
                    document_name=chunk["document_name"],
                    page=chunk["page"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=chunk["embedding"],
                )
            )
        document = session.get(Document, document_id)
        if document is None:
            session.rollback()
            return {"error": EXTRACT_FAILED}
        document.status = "ready"
        document.error = None
        session.commit()
    return {"error": None}


def mark_failed_node(state: IngestState) -> dict:
    """Set status to failed and keep the short reason already on the state."""
    document_id_text = state.get("document_id", "")
    try:
        document_id = uuid.UUID(document_id_text)
    except ValueError:
        return {}
    reason = state.get("error") or EXTRACT_FAILED
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        if document is None:
            return {}
        _save_failure(session, document, reason)
    logger.info("ingest failed", extra={"document_id": document_id_text})
    return {}


def build_ingest_graph(embed_texts_fn=None):
    """Compile the ingest graph. Pass a fake embedder in tests.

    The real model is not loaded here. It loads on the first real embed call.
    """
    if embed_texts_fn is None:
        embed_texts_fn = embed_texts

    def embed_node(state: IngestState) -> dict:
        """Turn each chunk into a 384-number vector."""
        chunks = state.get("chunks") or []
        passages = [chunk["content"] for chunk in chunks]
        if not passages:
            return {"error": EMBED_FAILED}
        try:
            vectors = embed_texts_fn(passages)
        except Exception:
            logger.info("embed failed", extra={"document_id": state.get("document_id")})
            return {"error": EMBED_FAILED}
        if len(vectors) != len(passages):
            return {"error": EMBED_FAILED}

        chunks_with_vectors = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != VECTOR_SIZE:
                return {"error": EMBED_FAILED}
            chunks_with_vectors.append({**chunk, "embedding": vector})
        return {"chunks": chunks_with_vectors, "error": None}

    builder = StateGraph(IngestState)
    builder.add_node("extract", extract_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("embed", embed_node)
    builder.add_node("persistReady", persist_ready_node)
    builder.add_node("markFailed", mark_failed_node)

    builder.add_edge(START, "extract")
    builder.add_conditional_edges(
        "extract",
        route_after_extract,
        {"chunk": "chunk", "markFailed": "markFailed"},
    )
    builder.add_edge("chunk", "embed")
    builder.add_conditional_edges(
        "embed",
        route_after_embed,
        {"persistReady": "persistReady", "markFailed": "markFailed"},
    )
    builder.add_edge("persistReady", END)
    builder.add_edge("markFailed", END)
    return builder.compile()


# Compiled once at import. The worker calls this object. No checkpointer.
graph = build_ingest_graph()

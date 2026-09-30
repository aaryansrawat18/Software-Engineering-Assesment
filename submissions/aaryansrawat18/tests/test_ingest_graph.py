"""The ingest graph, with a fake embedder so the real model is not loaded."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.auth import hash_password
from app.db import SessionLocal
from app.graphs.ingest import build_ingest_graph
from app.models import Chunk, Document, User
from app.services.embed import VECTOR_SIZE
from app.worker import ingest, requeue_stuck_documents


def add_document(filename: str, media_type: str, file_bytes: bytes, upload_dir) -> uuid.UUID:
    """Insert one user and one document, and write the file where the graph reads it."""
    with SessionLocal() as session:
        user = User(
            username=f"user-{uuid.uuid4().hex[:8]}",
            email=f"{uuid.uuid4().hex[:8]}@example.com",
            password_hash=hash_password("at-least-8"),
        )
        session.add(user)
        session.flush()
        document = Document(
            user_id=user.id,
            filename=filename,
            media_type=media_type,
            storage_key="",
            status="queued",
            byte_size=len(file_bytes),
            attempt_count=0,
        )
        session.add(document)
        session.flush()
        document.storage_key = str(document.id)
        session.commit()
        document_id = document.id
    (upload_dir / str(document_id)).write_bytes(file_bytes)
    return document_id


def visited_node_names(graph, document_id: uuid.UUID) -> list[str]:
    names = []
    for step in graph.stream({"document_id": str(document_id)}):
        names.extend(step.keys())
    return names


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1] * VECTOR_SIZE for _ in texts]


def test_good_file_ends_on_persist_ready(upload_dir):
    document_id = add_document(
        "handbook.txt",
        "text/plain",
        b"Employees receive 15 days of annual leave.",
        upload_dir,
    )
    graph = build_ingest_graph(fake_embed)
    names = visited_node_names(graph, document_id)

    assert names[-1] == "persistReady"
    assert "markFailed" not in names

    with SessionLocal() as session:
        document = session.get(Document, document_id)
        assert document is not None
        assert document.status == "ready"
        assert document.error is None
        chunks = session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()
        assert len(chunks) == 1
        assert chunks[0].content == "Employees receive 15 days of annual leave."
        assert chunks[0].page == 1
        assert chunks[0].document_name == "handbook.txt"
        assert len(chunks[0].embedding) == VECTOR_SIZE

    # Running the graph again replaces chunks instead of adding a second copy.
    visited_node_names(graph, document_id)
    with SessionLocal() as session:
        chunk_count = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        assert chunk_count == 1


def test_broken_pdf_ends_on_mark_failed(upload_dir):
    document_id = add_document(
        "broken.pdf",
        "application/pdf",
        b"%PDF-1.4\ngarbage",
        upload_dir,
    )
    graph = build_ingest_graph(fake_embed)
    names = visited_node_names(graph, document_id)

    assert names[-1] == "markFailed"
    assert "persistReady" not in names

    with SessionLocal() as session:
        document = session.get(Document, document_id)
        assert document is not None
        assert document.status == "failed"
        assert document.error == "extract_failed"
        leftover = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        assert leftover == 0


def test_embed_failure_does_not_store_the_exception_text(upload_dir):
    def broken_embed(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model exploded with secret file text")

    document_id = add_document("notes.txt", "text/plain", b"A short note.", upload_dir)
    graph = build_ingest_graph(broken_embed)
    names = visited_node_names(graph, document_id)

    assert names[-1] == "markFailed"
    assert "persistReady" not in names

    with SessionLocal() as session:
        document = session.get(Document, document_id)
        assert document is not None
        assert document.status == "failed"
        assert document.error == "embed_failed"
        leftover = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        assert leftover == 0


def test_ingest_job_marks_processing_before_the_graph(monkeypatch):
    document_id = None
    with SessionLocal() as session:
        user = User(
            username="ada-worker",
            email="ada-worker@example.com",
            password_hash=hash_password("at-least-8"),
        )
        session.add(user)
        session.flush()
        document = Document(
            user_id=user.id,
            filename="notes.txt",
            media_type="text/plain",
            storage_key="pending",
            status="queued",
            byte_size=4,
            attempt_count=0,
        )
        session.add(document)
        session.flush()
        document.storage_key = str(document.id)
        session.commit()
        document_id = document.id

    seen = {}

    def fake_invoke(payload):
        with SessionLocal() as session:
            document = session.get(Document, uuid.UUID(payload["document_id"]))
            assert document is not None
            seen["status"] = document.status
            seen["attempt_count"] = document.attempt_count
            seen["started"] = document.processing_started_at
        return {}

    monkeypatch.setattr("app.graphs.ingest.graph.invoke", fake_invoke)
    ingest(str(document_id))
    assert seen["status"] == "processing"
    assert seen["attempt_count"] == 1
    assert seen["started"] is not None

    ingest(str(uuid.uuid4()))


def test_stuck_processing_document_is_queued_again(monkeypatch):
    enqueued = []
    monkeypatch.setattr("app.worker.enqueue_ingest", lambda document_id: enqueued.append(document_id))

    with SessionLocal() as session:
        user = User(
            username="ada-stuck",
            email="ada-stuck@example.com",
            password_hash=hash_password("at-least-8"),
        )
        session.add(user)
        session.flush()
        stuck = Document(
            user_id=user.id,
            filename="old.txt",
            media_type="text/plain",
            storage_key="pending",
            status="processing",
            byte_size=1,
            attempt_count=1,
            processing_started_at=datetime.now(timezone.utc) - timedelta(minutes=11),
        )
        fresh = Document(
            user_id=user.id,
            filename="new.txt",
            media_type="text/plain",
            storage_key="pending",
            status="processing",
            byte_size=1,
            attempt_count=1,
            processing_started_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        session.add(stuck)
        session.add(fresh)
        session.flush()
        stuck.storage_key = str(stuck.id)
        fresh.storage_key = str(fresh.id)
        session.commit()
        stuck_id = stuck.id
        fresh_id = fresh.id

    requeue_stuck_documents()
    assert enqueued == [str(stuck_id)]

    with SessionLocal() as session:
        stuck_again = session.get(Document, stuck_id)
        fresh_again = session.get(Document, fresh_id)
        assert stuck_again is not None and stuck_again.status == "queued"
        assert fresh_again is not None and fresh_again.status == "processing"

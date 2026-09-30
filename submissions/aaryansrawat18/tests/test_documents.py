"""Upload, list, and delete. The queue is mocked so the request does not embed."""

import uuid

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Chunk, Document


def signup_and_token(client, username: str) -> str:
    client.post(
        "/auth/signup",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": "at-least-8",
        },
    )
    login = client.post("/auth/login", json={"username": username, "password": "at-least-8"})
    assert login.status_code == 200
    return login.json()["access_token"]


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_upload_returns_202_and_does_not_embed(client, upload_dir, monkeypatch):
    queued_ids = []

    def capture_queue(document_id: str) -> None:
        queued_ids.append(document_id)

    def embed_must_not_run(*_args, **_kwargs):
        raise AssertionError("The upload request must not embed the file.")

    def graph_must_not_run(*_args, **_kwargs):
        raise AssertionError("The upload request must not run the ingest graph.")

    monkeypatch.setattr("app.routers.documents.enqueue_ingest", capture_queue)
    monkeypatch.setattr("app.services.embed.embed_texts", embed_must_not_run)
    monkeypatch.setattr("app.graphs.ingest.graph.invoke", graph_must_not_run)

    token = signup_and_token(client, "ada")
    file_bytes = b"Employees receive 15 days of annual leave."
    response = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("handbook.txt", file_bytes, "text/plain")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    document_id = body["id"]
    assert queued_ids == [document_id]

    saved_file = upload_dir / document_id
    assert saved_file.read_bytes() == file_bytes

    with SessionLocal() as session:
        document = session.get(Document, uuid.UUID(document_id))
        assert document is not None
        assert document.status == "queued"
        assert document.attempt_count == 0
        assert document.storage_key == document_id
        assert document.byte_size == len(file_bytes)
        assert document.filename == "handbook.txt"
        assert document.media_type == "text/plain"


def test_upload_accepts_pdf_magic_and_markdown(client, monkeypatch):
    monkeypatch.setattr("app.routers.documents.enqueue_ingest", lambda document_id: None)
    token = signup_and_token(client, "ada")

    pdf = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("scan.bin", b"%PDF-1.4\n", "application/octet-stream")},
    )
    assert pdf.status_code == 202
    with SessionLocal() as session:
        document = session.get(Document, uuid.UUID(pdf.json()["id"]))
        assert document is not None
        assert document.media_type == "application/pdf"

    markdown = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("guide.md", b"# Leave policy\n", "application/octet-stream")},
    )
    assert markdown.status_code == 202
    with SessionLocal() as session:
        document = session.get(Document, uuid.UUID(markdown.json()["id"]))
        assert document is not None
        assert document.media_type == "text/markdown"


def test_upload_rejects_oversize_and_binary(client, monkeypatch):
    monkeypatch.setattr("app.routers.documents.enqueue_ingest", lambda document_id: None)
    token = signup_and_token(client, "ada")

    too_big = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("big.txt", b"a" * (10 * 1024 * 1024 + 1), "text/plain")},
    )
    assert too_big.status_code == 413

    binary = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("bad.bin", b"\xff\xfe\x00\x01", "application/octet-stream")},
    )
    assert binary.status_code == 415

    missing_file = client.post("/documents", headers=auth_header(token))
    assert missing_file.status_code == 422
    assert client.get("/documents").status_code == 401

    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(Document))
        assert count == 0


def test_list_is_only_the_callers_documents(client, monkeypatch):
    monkeypatch.setattr("app.routers.documents.enqueue_ingest", lambda document_id: None)
    ada_token = signup_and_token(client, "ada")
    grace_token = signup_and_token(client, "grace")
    uploaded = client.post(
        "/documents",
        headers=auth_header(ada_token),
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert uploaded.status_code == 202

    ada_list = client.get("/documents", headers=auth_header(ada_token))
    grace_list = client.get("/documents", headers=auth_header(grace_token))
    assert ada_list.status_code == 200
    assert grace_list.status_code == 200
    assert len(ada_list.json()) == 1
    assert ada_list.json()[0]["filename"] == "notes.txt"
    assert grace_list.json() == []


def test_other_user_get_and_delete_are_404(client, upload_dir, monkeypatch):
    monkeypatch.setattr("app.routers.documents.enqueue_ingest", lambda document_id: None)
    ada_token = signup_and_token(client, "ada")
    grace_token = signup_and_token(client, "grace")
    uploaded = client.post(
        "/documents",
        headers=auth_header(ada_token),
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    document_id = uploaded.json()["id"]

    foreign_get = client.get(f"/documents/{document_id}", headers=auth_header(grace_token))
    foreign_delete = client.delete(f"/documents/{document_id}", headers=auth_header(grace_token))
    assert foreign_get.status_code == 404
    assert foreign_delete.status_code == 404

    missing = client.get(f"/documents/{uuid.uuid4()}", headers=auth_header(ada_token))
    assert missing.status_code == 404

    owner_get = client.get(f"/documents/{document_id}", headers=auth_header(ada_token))
    assert owner_get.status_code == 200
    assert owner_get.json()["id"] == document_id
    assert (upload_dir / document_id).exists()


def test_delete_removes_file_and_chunks(client, upload_dir, monkeypatch):
    monkeypatch.setattr("app.routers.documents.enqueue_ingest", lambda document_id: None)
    token = signup_and_token(client, "ada")
    uploaded = client.post(
        "/documents",
        headers=auth_header(token),
        files={"file": ("notes.txt", b"hello notes", "text/plain")},
    )
    document_id = uuid.UUID(uploaded.json()["id"])

    with SessionLocal() as session:
        document = session.get(Document, document_id)
        assert document is not None
        session.add(
            Chunk(
                document_id=document.id,
                user_id=document.user_id,
                document_name=document.filename,
                page=1,
                chunk_index=0,
                content="hello notes",
                embedding=[0.0] * 384,
            )
        )
        session.commit()

    deleted = client.delete(f"/documents/{document_id}", headers=auth_header(token))
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert not (upload_dir / str(document_id)).exists()

    with SessionLocal() as session:
        assert session.get(Document, document_id) is None
        leftover = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        assert leftover == 0

    gone = client.get(f"/documents/{document_id}", headers=auth_header(token))
    assert gone.status_code == 404

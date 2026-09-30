"""Questions, citations, and refusals. Gemini and MiniLM are mocked."""

import json
import uuid
from pathlib import Path

from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal
from app.graphs.answer import NOT_FOUND_ANSWER
from app.models import Chunk, Document, Question, User
from app.services.embed import VECTOR_SIZE
from app.services.generate import SYSTEM_PROMPT, build_user_message
from tests.test_documents import auth_header, signup_and_token

PASSAGE = "Employees receive 15 days of annual leave."
INJECTION_FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "prompt-injection.md"


def unit_vector(hot_index: int) -> list[float]:
    """One 384-number vector with a single 1. Two different indexes are unrelated."""
    vector = [0.0] * VECTOR_SIZE
    vector[hot_index] = 1.0
    return vector


def user_id_for(username: str) -> uuid.UUID:
    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.username == username))
        assert user is not None
        return user.id


def add_ready_passage(
    owner_id: uuid.UUID,
    content: str,
    embedding: list[float],
    filename: str = "handbook.txt",
    status: str = "ready",
    page: int = 2,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert a document and one chunk. `status` defaults to ready."""
    with SessionLocal() as session:
        document = Document(
            user_id=owner_id,
            filename=filename,
            media_type="text/plain",
            storage_key=str(uuid.uuid4()),
            status=status,
            byte_size=len(content.encode()),
            attempt_count=1,
        )
        session.add(document)
        session.flush()
        chunk = Chunk(
            document_id=document.id,
            user_id=owner_id,
            document_name=filename,
            page=page,
            chunk_index=0,
            content=content,
            embedding=embedding,
        )
        session.add(chunk)
        session.commit()
        return document.id, chunk.id


class FakeResponse:
    """The small part of an HTTP response `generate` actually reads."""

    def __init__(self, status_code: int, payload: dict | None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeGemini:
    """Stand-in for `httpx.Client`. Records each call and returns scripted replies."""

    def __init__(self, replies: list[tuple[int, dict | None]]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        fake = self

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json=None, headers=None):
                fake.calls.append({"url": url, "json": json, "headers": headers})
                status_code, payload = fake.replies.pop(0)
                return FakeResponse(status_code, payload)

        return Client()


def gemini_payload(text: str, prompt_tokens: int = 10, candidate_tokens: int = 5, thought_tokens: int = 2) -> dict:
    """A Gemini body. Output tokens are candidate tokens plus thinking tokens."""
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidate_tokens,
            "thoughtsTokenCount": thought_tokens,
        },
    }


def model_json(chunk_ids: list[str], answer: str = "15 days of annual leave.", refused: bool = False) -> str:
    return json.dumps({"answer": answer, "chunk_ids": chunk_ids, "refused": refused})


def use_gemini(monkeypatch, replies: list[tuple[int, dict | None]]) -> FakeGemini:
    monkeypatch.setattr(get_settings(), "gemini_api_key", "test-key")
    monkeypatch.setattr("app.services.generate.RETRY_WAIT_SECONDS", 0)
    fake = FakeGemini(replies)
    monkeypatch.setattr("app.services.generate.httpx.Client", fake)
    return fake


def use_matching_embedding(monkeypatch) -> None:
    """The question vector matches a chunk stored with `unit_vector(0)`."""
    monkeypatch.setattr("app.services.retrieve.embed_texts", lambda texts: [unit_vector(0)])


def track_answer_nodes(monkeypatch) -> list[str]:
    """Run the real graph, but remember which nodes ran."""
    from app.graphs import answer as answer_module

    visited: list[str] = []

    def tracking_invoke(inputs, *args, **kwargs):
        state = dict(inputs)
        for step in answer_module.graph.stream(inputs):
            for node_name, update in step.items():
                visited.append(node_name)
                if isinstance(update, dict):
                    state.update(update)
        return state

    monkeypatch.setattr("app.routers.questions.answer_graph.invoke", tracking_invoke)
    return visited


def test_happy_path_cites_the_retrieved_passage(client, monkeypatch):
    use_matching_embedding(monkeypatch)
    token = signup_and_token(client, "ada")
    owner_id = user_id_for("ada")
    document_id, chunk_id = add_ready_passage(owner_id, PASSAGE, unit_vector(0))
    queued_id, _queued_chunk = add_ready_passage(
        owner_id,
        "THIS QUEUED TEXT MUST NOT BE SENT",
        unit_vector(0),
        filename="draft.txt",
        status="queued",
        page=9,
    )
    fake = use_gemini(monkeypatch, [(200, gemini_payload(model_json([str(chunk_id)])))])
    visited = track_answer_nodes(monkeypatch)

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "How many days of leave?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"] == "15 days of annual leave."
    assert body["citations"] == [{"document_name": "handbook.txt", "passage": PASSAGE, "page": 2}]
    # 10 input + 5 candidate + 2 thinking. Output price includes thinking tokens.
    assert body["usage"]["tokens"] == 17
    assert body["usage"]["estimated_cost_usd"] == 0.0000205
    assert body["usage"]["latency_ms"] >= 0
    assert visited == ["retrieve", "generate", "filterCitations", "persist"]

    user_text = fake.calls[0]["json"]["contents"][0]["parts"][0]["text"]
    system_text = fake.calls[0]["json"]["systemInstruction"]["parts"][0]["text"]
    assert system_text == SYSTEM_PROMPT
    assert PASSAGE in user_text
    assert "THIS QUEUED TEXT MUST NOT BE SENT" not in user_text
    assert SYSTEM_PROMPT not in user_text
    assert fake.calls[0]["headers"]["x-goog-api-key"] == "test-key"
    assert queued_id is not None

    history = client.get("/questions", headers=auth_header(token))
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["id"] == body["id"]
    assert history.json()[0]["document_ids"] == [str(document_id)]
    assert history.json()[0]["refused"] is False

    other = signup_and_token(client, "grace")
    assert client.get("/questions", headers=auth_header(other)).json() == []


def test_other_users_document_id_is_404_and_skips_the_graph(client, monkeypatch):
    ada = signup_and_token(client, "ada")
    grace = signup_and_token(client, "grace")
    ada_document, _chunk = add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(0))
    grace_document, _grace_chunk = add_ready_passage(
        user_id_for("grace"), "Grace's notes.", unit_vector(0), filename="grace.txt"
    )

    def graph_must_not_run(*args, **kwargs):
        raise AssertionError("The answer graph must not run for a foreign document id.")

    monkeypatch.setattr("app.routers.questions.answer_graph.invoke", graph_must_not_run)
    response = client.post(
        "/questions",
        headers=auth_header(ada),
        json={"question": "What does Grace know?", "document_ids": [str(ada_document), str(grace_document)]},
    )
    assert response.status_code == 404
    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(Question))
        assert count == 0


def test_low_similarity_refuses_without_calling_gemini(client, monkeypatch):
    # Question vector is unrelated to the stored passage, so the gate skips Gemini.
    monkeypatch.setattr("app.services.retrieve.embed_texts", lambda texts: [unit_vector(0)])
    fake = use_gemini(monkeypatch, [(200, gemini_payload(model_json(["should-not-be-used"])))])
    visited = track_answer_nodes(monkeypatch)
    token = signup_and_token(client, "ada")
    add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(1))

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "What is the office wifi password?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == []
    assert body["usage"]["tokens"] == 0
    assert body["usage"]["estimated_cost_usd"] == 0.0
    assert fake.calls == []
    assert visited == ["retrieve", "refuse", "persist"]


def test_filter_drops_a_chunk_id_the_model_invented(client, monkeypatch):
    use_matching_embedding(monkeypatch)
    token = signup_and_token(client, "ada")
    _document_id, chunk_id = add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(0))
    invented_id = str(uuid.uuid4())
    use_gemini(monkeypatch, [(200, gemini_payload(model_json([str(chunk_id), invented_id])))])
    visited = track_answer_nodes(monkeypatch)

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "How many days of leave?"},
    )
    assert response.status_code == 200
    assert response.json()["citations"] == [
        {"document_name": "handbook.txt", "passage": PASSAGE, "page": 2}
    ]
    assert invented_id not in response.text
    assert visited == ["retrieve", "generate", "filterCitations", "persist"]


def test_deleted_document_is_not_retrieved(client, monkeypatch):
    use_matching_embedding(monkeypatch)
    fake = use_gemini(monkeypatch, [(200, gemini_payload(model_json(["should-not-be-used"])))])
    token = signup_and_token(client, "ada")
    document_id, chunk_id = add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(0))

    deleted = client.delete(f"/documents/{document_id}", headers=auth_header(token))
    assert deleted.status_code == 204
    with SessionLocal() as session:
        remaining = session.scalar(select(func.count()).select_from(Chunk).where(Chunk.id == chunk_id))
        assert remaining == 0

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "How many days of leave?"},
    )
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["citations"] == []
    assert PASSAGE not in response.json()["answer"]
    assert fake.calls == []


def test_bad_model_json_is_503_and_saves_nothing(client, monkeypatch):
    use_matching_embedding(monkeypatch)
    use_gemini(monkeypatch, [(200, gemini_payload("this is not json"))])
    token = signup_and_token(client, "ada")
    add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(0))

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "How many days of leave?"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "The answer service is unavailable. Try again."
    assert "this is not json" not in response.text
    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(Question))
        assert count == 0


def test_gemini_is_retried_once_after_a_server_error(client, monkeypatch):
    use_matching_embedding(monkeypatch)
    token = signup_and_token(client, "ada")
    _document_id, chunk_id = add_ready_passage(user_id_for("ada"), PASSAGE, unit_vector(0))
    fake = use_gemini(
        monkeypatch,
        [
            (503, {"error": "provider body must not leak"}),
            (200, gemini_payload(model_json([str(chunk_id)]))),
        ],
    )

    response = client.post(
        "/questions",
        headers=auth_header(token),
        json={"question": "How many days of leave?"},
    )
    assert response.status_code == 200
    assert len(fake.calls) == 2
    assert "provider body must not leak" not in response.text
    assert response.json()["citations"][0]["passage"] == PASSAGE


def test_blank_question_and_missing_token_are_rejected(client):
    token = signup_and_token(client, "ada")
    blank = client.post("/questions", headers=auth_header(token), json={"question": "   "})
    assert blank.status_code == 422
    assert client.post("/questions", json={"question": "How many days of leave?"}).status_code == 401
    assert client.get("/questions").status_code == 401


def test_injection_fixture_stays_in_the_user_message():
    """The hostile file is a passage. It is not copied into the system prompt."""
    passage = INJECTION_FIXTURE.read_text(encoding="utf-8")
    user_message = build_user_message(
        "What is the secret password?",
        [
            {
                "chunk_id": "chunk-1",
                "document_name": "prompt-injection.md",
                "page": 1,
                "content": passage,
            }
        ],
    )
    assert "Ignore all previous instructions" in passage
    assert "Ignore all previous instructions" in user_message
    assert "Ignore all previous instructions" not in SYSTEM_PROMPT
    assert passage not in SYSTEM_PROMPT
    assert SYSTEM_PROMPT not in user_message
    assert "chunk_id: chunk-1" in user_message

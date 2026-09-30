"""POST /questions allows 10 asks per user per minute. Upload is not limited."""

import uuid

from app.services.rate_limit import QUESTIONS_PER_MINUTE
from tests.test_documents import auth_header, signup_and_token


def fake_answer(_state):
    """A finished graph result. These tests do not call Gemini."""
    return {
        "question_id": str(uuid.uuid4()),
        "answer": "15 days.",
        "citations": [],
        "tokens": 0,
        "latency_ms": 1,
        "estimated_cost_usd": 0.0,
        "refused": False,
    }


def test_eleventh_question_is_429_and_does_not_run_the_graph(client, monkeypatch, memory_redis):
    calls = []

    def record_and_answer(state):
        calls.append(state)
        return fake_answer(state)

    monkeypatch.setattr("app.routers.questions.answer_graph.invoke", record_and_answer)
    token = signup_and_token(client, "rate-limit-user")
    headers = auth_header(token)
    body = {"question": "How many days of leave?"}

    for _ in range(QUESTIONS_PER_MINUTE):
        response = client.post("/questions", headers=headers, json=body)
        assert response.status_code == 200

    blocked = client.post("/questions", headers=headers, json=body)
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "Too many questions. You can ask 10 per minute."
    assert len(calls) == QUESTIONS_PER_MINUTE
    rate_keys = [key for key in memory_redis.values if key.startswith("rl:")]
    assert len(rate_keys) == 1
    assert memory_redis.values[rate_keys[0]] == QUESTIONS_PER_MINUTE + 1
    assert memory_redis.ttls[rate_keys[0]] == 60


def test_upload_is_not_rate_limited(client, upload_dir, monkeypatch):
    monkeypatch.setattr(
        "app.routers.documents.enqueue_ingest",
        lambda document_id, request_id=None: None,
    )
    token = signup_and_token(client, "upload-user")
    headers = auth_header(token)
    for _ in range(QUESTIONS_PER_MINUTE + 1):
        response = client.post(
            "/documents",
            headers=headers,
            files={"file": ("note.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 202

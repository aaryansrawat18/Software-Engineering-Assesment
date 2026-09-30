"""Health routes. Live stays up when Redis or the worker is down."""

from datetime import datetime, timedelta, timezone

from redis.exceptions import RedisError
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Document, User
from app.services.heartbeat import SWEEP_TIMEOUT_MINUTES, WORKER_HEARTBEAT_KEY
from tests.test_documents import signup_and_token


def test_live_is_ok_even_when_the_worker_heartbeat_is_missing(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_names_the_worker_when_the_heartbeat_is_missing(client):
    for path in ("/health", "/health/ready"):
        response = client.get(path)
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "down"
        assert body["checks"]["database"] == "ok"
        assert body["checks"]["vector"] == "ok"
        assert body["checks"]["redis"] == "ok"
        assert body["checks"]["worker"] == "down"
        assert "postgres" not in response.text
        assert "redis://" not in response.text


def test_ready_is_ok_when_the_worker_heartbeat_is_present(client, memory_redis):
    memory_redis.set(WORKER_HEARTBEAT_KEY, "alive", ex=15)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == {
        "database": "ok",
        "vector": "ok",
        "redis": "ok",
        "worker": "ok",
    }


def test_old_queued_document_marks_the_worker_down(client, memory_redis):
    memory_redis.set(WORKER_HEARTBEAT_KEY, "alive", ex=15)
    signup_and_token(client, "queue-owner")
    with SessionLocal() as session:
        owner = session.scalar(select(User).where(User.username == "queue-owner"))
        assert owner is not None
        session.add(
            Document(
                user_id=owner.id,
                filename="old.txt",
                media_type="text/plain",
                storage_key="old-file",
                status="queued",
                byte_size=4,
                attempt_count=0,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=SWEEP_TIMEOUT_MINUTES + 1),
            )
        )
        session.commit()

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"]["worker"] == "down"


def test_redis_down_is_named(client, monkeypatch):
    def redis_is_down():
        raise RedisError("unavailable")

    monkeypatch.setattr("app.services.redis_client.get_redis", redis_is_down)
    response = client.get("/health")
    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["redis"] == "down"
    assert checks["worker"] == "down"
    assert checks["database"] == "ok"

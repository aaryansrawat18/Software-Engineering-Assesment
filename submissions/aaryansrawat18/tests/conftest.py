import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/documind",
)
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-32b")

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.auth import current_user
from app.config import get_settings
from app.db import engine, init_db
from app.main import app
from app.models import User


@app.get("/protected-placeholder")
def protected_placeholder(user: User = Depends(current_user)) -> dict[str, str]:
    return {"id": str(user.id)}


def _truncate() -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE chunks, questions, documents, users CASCADE"))


@pytest.fixture(scope="session", autouse=True)
def database():
    init_db()
    yield


@pytest.fixture(autouse=True)
def clean_tables(database):
    _truncate()
    yield
    _truncate()


class MemoryRedis:
    """The Redis commands this app uses, stored in a dict so tests need no server.

    `incr`, `expire`, `ping`, `set`, and `exists` match the real client closely
    enough for the rate limit, the heartbeat, and the health check.
    """

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.ttls: dict[str, int] = {}

    def incr(self, key: str) -> int:
        count = int(self.values.get(key, 0)) + 1
        self.values[key] = count
        return count

    def expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: object, ex: int | None = None) -> bool:
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    def exists(self, key: str) -> int:
        return 1 if key in self.values else 0


@pytest.fixture(autouse=True)
def memory_redis(monkeypatch):
    """Point every Redis call at one in-memory client for this test."""
    fake_redis = MemoryRedis()
    monkeypatch.setattr("app.services.redis_client.get_redis", lambda: fake_redis)
    return fake_redis


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Store uploaded files in a temporary folder so tests do not touch real uploads."""
    monkeypatch.setattr(get_settings(), "upload_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(upload_dir):
    with TestClient(app) as test_client:
        yield test_client

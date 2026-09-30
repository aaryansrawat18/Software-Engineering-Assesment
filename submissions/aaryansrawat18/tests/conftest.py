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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

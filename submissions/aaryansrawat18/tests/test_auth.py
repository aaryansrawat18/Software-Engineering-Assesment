import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from app.auth import get_owned_document, verify_password
from app.config import Settings
from app.db import SessionLocal
from app.models import Document, User


def test_settings_reject_blank_secret(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/documind")
    monkeypatch.setenv("JWT_SECRET", " ")
    with pytest.raises(ValidationError):
        Settings()


def test_signup_hashes_password(client):
    password = "at-least-8"
    response = client.post(
        "/auth/signup",
        json={"username": "ada", "email": "ada@example.com", "password": password},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "ada"
    assert body["email"] == "ada@example.com"
    assert "password" not in body
    assert "password_hash" not in body

    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.username == "ada"))
        assert user is not None
        assert user.password_hash != password
        assert user.password_hash.startswith("$argon2")
        assert verify_password(password, user.password_hash)


def test_login_rejects_wrong_password(client):
    client.post(
        "/auth/signup",
        json={"username": "ada", "email": "ada@example.com", "password": "at-least-8"},
    )
    wrong = client.post("/auth/login", json={"username": "ada", "password": "wrong-password"})
    unknown = client.post("/auth/login", json={"username": "nobody", "password": "at-least-8"})
    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert wrong.json()["detail"] == "Incorrect username or password."
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_protected_route_requires_token(client):
    missing = client.get("/protected-placeholder")
    assert missing.status_code == 401

    client.post(
        "/auth/signup",
        json={"username": "ada", "email": "ada@example.com", "password": "at-least-8"},
    )
    login = client.post("/auth/login", json={"username": "ada", "password": "at-least-8"})
    assert login.status_code == 200
    token_body = login.json()
    assert token_body["token_type"] == "bearer"
    assert token_body["expires_in"] == 86400
    by_email = client.post(
        "/auth/login",
        json={"username": "Ada@Example.com", "password": "at-least-8"},
    )
    assert by_email.status_code == 200
    ok = client.get(
        "/protected-placeholder",
        headers={"Authorization": f"Bearer {token_body['access_token']}"},
    )
    assert ok.status_code == 200


def test_duplicate_username_or_email_is_409(client):
    payload = {"username": "ada", "email": "ada@example.com", "password": "at-least-8"}
    assert client.post("/auth/signup", json=payload).status_code == 201
    again = client.post(
        "/auth/signup",
        json={"username": "ada", "email": "other@example.com", "password": "at-least-8"},
    )
    assert again.status_code == 409
    email = client.post(
        "/auth/signup",
        json={"username": "other", "email": "ada@example.com", "password": "at-least-8"},
    )
    assert email.status_code == 409


def test_other_users_document_is_404(client):
    client.post(
        "/auth/signup",
        json={"username": "ada", "email": "ada@example.com", "password": "at-least-8"},
    )
    client.post(
        "/auth/signup",
        json={"username": "grace", "email": "grace@example.com", "password": "at-least-8"},
    )
    with SessionLocal() as session:
        ada = session.scalar(select(User).where(User.username == "ada"))
        grace = session.scalar(select(User).where(User.username == "grace"))
        assert ada is not None and grace is not None
        document = Document(
            id=uuid.uuid4(),
            user_id=ada.id,
            filename="policy.txt",
            media_type="text/plain",
            storage_key="pending",
            status="queued",
            byte_size=12,
            attempt_count=0,
        )
        document.storage_key = str(document.id)
        session.add(document)
        session.commit()

        assert get_owned_document(session, ada.id, document.id).id == document.id
        with pytest.raises(HTTPException) as foreign:
            get_owned_document(session, grace.id, document.id)
        assert foreign.value.status_code == 404
        with pytest.raises(HTTPException) as missing:
            get_owned_document(session, ada.id, uuid.uuid4())
        assert missing.value.status_code == 404

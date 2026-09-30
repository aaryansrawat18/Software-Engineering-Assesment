import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.models import Document, User

ACCESS_TOKEN_TTL_SECONDS = 86400
_ALGORITHM = "HS256"
_hasher = PasswordHash.recommended()
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password, password_hash)
    except UnknownHashError:
        return False


def create_token(user_id: uuid.UUID) -> str:
    expires = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
    payload = {"sub": str(user_id), "exp": expires}
    return jwt.encode(payload, get_settings().jwt_secret, algorithm=_ALGORITHM)


def read_user_id_from_token(token: str) -> str | None:
    """Return the user id inside a token, or None when the token is bad.

    Logs use this. The route still rejects a bad token with 401.
    The token string is not returned.
    """
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=[_ALGORITHM])
        return str(uuid.UUID(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_session),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        payload = jwt.decode(credentials.credentials, get_settings().jwt_secret, algorithms=[_ALGORITHM])
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Not authenticated.") from None
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


def get_owned_document(session: Session, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    document = session.get(Document, document_id)
    if document is None or document.user_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


def find_user_by_login(session: Session, username_or_email: str) -> User | None:
    user = session.scalar(select(User).where(User.username == username_or_email))
    if user is not None:
        return user
    return session.scalar(select(User).where(User.email == username_or_email.lower()))

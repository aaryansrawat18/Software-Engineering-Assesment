import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import (
    ACCESS_TOKEN_TTL_SECONDS,
    create_token,
    find_user_by_login,
    hash_password,
    verify_password,
)
from app.db import get_session
from app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class SignupRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username is required")
        return value

    @field_validator("email")
    @classmethod
    def lower_email(cls, value: str) -> str:
        return value.lower()


class UserResponse(BaseModel):
    id: uuid.UUID
    username: str
    email: str


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username is required")
        return value


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


@router.post("/signup", response_model=UserResponse, status_code=201)
def signup(body: SignupRequest, session: Session = Depends(get_session)) -> UserResponse:
    existing = session.scalar(
        select(User.id).where(or_(User.username == body.username, User.email == body.email))
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username or email already exists.")
    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists.") from None
    session.refresh(user)
    return UserResponse(id=user.id, username=user.username, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)) -> TokenResponse:
    user = find_user_by_login(session, body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return TokenResponse(
        access_token=create_token(user.id),
        token_type="bearer",
        expires_in=ACCESS_TOKEN_TTL_SECONDS,
    )

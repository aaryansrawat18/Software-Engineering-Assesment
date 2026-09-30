"""Ask a question and read this user's question history.

`POST /questions` checks the token and ownership, then runs the answer graph
once. The router does not call Gemini. `GET /questions` only reads rows.
"""

import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user, get_owned_document
from app.db import get_session
from app.graphs.answer import AnswerFailed, graph as answer_graph
from app.models import Document, Question, User

router = APIRouter(prefix="/questions", tags=["questions"])

ANSWER_UNAVAILABLE = "The answer service is unavailable. Try again."


class AskQuestionRequest(BaseModel):
    """A question, and an optional set of document ids to search."""

    question: str = Field(min_length=1)
    document_ids: list[uuid.UUID] | None = None

    @field_validator("question")
    @classmethod
    def question_must_have_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question is required.")
        return cleaned


class CitationResponse(BaseModel):
    """One cited passage. The chunk id stays internal."""

    document_name: str
    passage: str
    page: int


class UsageResponse(BaseModel):
    """Tokens, time, and estimated cost for this question."""

    tokens: int
    latency_ms: int
    estimated_cost_usd: float


class AskQuestionResponse(BaseModel):
    """What `POST /questions` returns. A refusal is still HTTP 200."""

    id: uuid.UUID
    answer: str
    citations: list[CitationResponse]
    usage: UsageResponse
    refused: bool


class QuestionHistoryItem(BaseModel):
    """One saved question in `GET /questions`."""

    id: uuid.UUID
    question: str
    answer: str
    citations: list[CitationResponse]
    document_ids: list[uuid.UUID]
    usage: UsageResponse
    refused: bool
    created_at: datetime


def _documents_in_scope(
    session: Session,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None,
) -> list[uuid.UUID]:
    """Decide which documents this question may search.

    An omitted or empty list means every ready document this user owns.
    When ids are sent, each one must belong to this user. One missing or
    foreign id is 404 for the whole request.
    """
    if document_ids:
        for document_id in document_ids:
            get_owned_document(session, user_id, document_id)
        return list(document_ids)

    ready_ids = session.scalars(
        select(Document.id).where(Document.user_id == user_id, Document.status == "ready")
    ).all()
    return list(ready_ids)


def _usage_from_state(state: dict) -> UsageResponse:
    return UsageResponse(
        tokens=int(state.get("tokens") or 0),
        latency_ms=int(state.get("latency_ms") or 0),
        estimated_cost_usd=float(state.get("estimated_cost_usd") or 0.0),
    )


def _to_history_item(row: Question) -> QuestionHistoryItem:
    return QuestionHistoryItem(
        id=row.id,
        question=row.question,
        answer=row.answer,
        citations=row.citations or [],
        document_ids=list(row.document_ids or []),
        usage=UsageResponse(
            tokens=row.tokens,
            latency_ms=row.latency_ms,
            estimated_cost_usd=float(row.estimated_cost),
        ),
        refused=row.refused,
        created_at=row.created_at,
    )


@router.post("", response_model=AskQuestionResponse)
def ask_question(
    body: AskQuestionRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> AskQuestionResponse:
    """Run the answer graph and return the saved answer or a not-found refusal."""
    document_ids = _documents_in_scope(session, user.id, body.document_ids)
    started_at = time.perf_counter()
    try:
        state = answer_graph.invoke(
            {
                "user_id": str(user.id),
                "question": body.question,
                "document_ids": [str(document_id) for document_id in document_ids],
                "started_at": started_at,
            }
        )
    except AnswerFailed:
        raise HTTPException(status_code=503, detail=ANSWER_UNAVAILABLE) from None

    question_id = state.get("question_id")
    if not question_id:
        raise HTTPException(status_code=503, detail=ANSWER_UNAVAILABLE)
    return AskQuestionResponse(
        id=uuid.UUID(question_id),
        answer=state.get("answer") or "",
        citations=state.get("citations") or [],
        usage=_usage_from_state(state),
        refused=bool(state.get("refused")),
    )


@router.get("", response_model=list[QuestionHistoryItem])
def list_questions(
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> list[QuestionHistoryItem]:
    """Return this user's questions, newest first. Does not run the answer graph."""
    rows = session.scalars(
        select(Question).where(Question.user_id == user.id).order_by(Question.created_at.desc())
    ).all()
    return [_to_history_item(row) for row in rows]

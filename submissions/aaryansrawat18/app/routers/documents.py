"""Upload, list, and delete documents.

`POST /documents` checks the token, saves the file, and queues the document id.
It does not extract, chunk, or embed. The worker does that in the background.
List and delete only see documents owned by the token's user. A missing or
foreign id is 404, so the response does not reveal that the id exists.
"""

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user, get_owned_document
from app.db import get_session
from app.models import Document, User
from app.services.extract import ExtractFailed, choose_media_type
from app.services.queue import enqueue_ingest
from app.services.storage import document_file_path

router = APIRouter(prefix="/documents", tags=["documents"])

# Stop reading once the file passes this size. 10 MB.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Read the upload in small pieces so a huge body is not held all at once.
READ_PIECE_BYTES = 64 * 1024
# Multipart wrappers sit around the file. Allow this much extra on Content-Length.
FORM_OVERHEAD_BYTES = 64 * 1024

FILE_TOO_LARGE = "File is larger than 10 MB."
UNSUPPORTED_FILE = "Only PDF and UTF-8 text or Markdown files are allowed."


class QueuedDocumentResponse(BaseModel):
    """What `POST /documents` returns right after the file is queued."""

    id: uuid.UUID
    status: str


class DocumentResponse(BaseModel):
    """One document in a list or a single get. No file bytes."""

    id: uuid.UUID
    filename: str
    media_type: str
    status: str
    error: str | None
    byte_size: int
    created_at: datetime


def _reject_if_content_length_is_too_big(request: Request) -> None:
    """Refuse before reading when the request header is already over the cap.

    A client can omit or lie about Content-Length. The read loop below is the
    check that actually stops a file past 10 MB.
    """
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        return
    try:
        content_length = int(raw_length)
    except ValueError:
        return
    if content_length > MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES:
        raise HTTPException(status_code=413, detail=FILE_TOO_LARGE)


async def _read_upload_with_size_limit(upload: UploadFile) -> bytes:
    """Read the file in pieces. Raise 413 as soon as the size passes 10 MB."""
    pieces: list[bytes] = []
    total_bytes = 0
    while True:
        piece = await upload.read(READ_PIECE_BYTES)
        if not piece:
            break
        if total_bytes + len(piece) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=FILE_TOO_LARGE)
        pieces.append(piece)
        total_bytes += len(piece)
    return b"".join(pieces)


def _original_filename(upload: UploadFile) -> str:
    """Keep the base name only, so a path in the filename cannot escape the folder."""
    filename = Path(upload.filename or "upload").name
    if not filename or filename in {".", ".."}:
        return "upload"
    return filename


def _to_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        media_type=document.media_type,
        status=document.status,
        error=document.error,
        byte_size=document.byte_size,
        created_at=document.created_at,
    )


@router.post("", status_code=202, response_model=QueuedDocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> QueuedDocumentResponse:
    """Save the file and enqueue its id. Return 202 without waiting for the worker."""
    _reject_if_content_length_is_too_big(request)
    file_bytes = await _read_upload_with_size_limit(file)
    filename = _original_filename(file)
    try:
        media_type = choose_media_type(file_bytes, filename, file.content_type)
    except ExtractFailed:
        raise HTTPException(status_code=415, detail=UNSUPPORTED_FILE) from None

    document_id = uuid.uuid4()
    file_path = document_file_path(str(document_id))
    file_path.write_bytes(file_bytes)

    document = Document(
        id=document_id,
        user_id=user.id,
        filename=filename,
        media_type=media_type,
        storage_key=str(document_id),
        status="queued",
        error=None,
        byte_size=len(file_bytes),
        attempt_count=0,
    )
    session.add(document)
    try:
        session.commit()
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    try:
        # Queue the id string only. The worker opens the file we just wrote.
        # Pass the request id so the worker log matches this HTTP request.
        enqueue_ingest(str(document_id), request_id=getattr(request.state, "request_id", None))
    except Exception:
        session.delete(document)
        session.commit()
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not queue the document. Try the upload again.") from None

    return QueuedDocumentResponse(id=document_id, status="queued")


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> list[DocumentResponse]:
    """Return this user's documents, newest first."""
    documents = session.scalars(
        select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())
    ).all()
    return [_to_document_response(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: uuid.UUID,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> DocumentResponse:
    """Return one document. Someone else's id, or a missing id, is 404."""
    document = get_owned_document(session, user.id, document_id)
    return _to_document_response(document)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: uuid.UUID,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Response:
    """Delete the file and the row. The database removes chunks via ON DELETE CASCADE."""
    document = get_owned_document(session, user.id, document_id)
    file_path = document_file_path(document.storage_key)
    session.delete(document)
    session.commit()
    file_path.unlink(missing_ok=True)
    return Response(status_code=204)

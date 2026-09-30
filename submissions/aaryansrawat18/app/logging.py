"""JSON logs for the API.

Each request gets an id. The log line can include the path, status, user id,
document id, and latency. It never includes the document text, the password,
the token, or the Gemini API key. Those values are not in the field list below,
so a caller cannot add them by accident through `extra=`.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request

from app.auth import read_user_id_from_token

# Only these extra fields are copied into the JSON line.
_FIELDS = ("request_id", "path", "status", "user_id", "document_id", "latency_ms")

# A caller-supplied id longer than this is ignored. It is not safe to log a huge header.
_MAX_REQUEST_ID_LENGTH = 200

logger = logging.getLogger("app.request")


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Unknown fields are left out."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in _FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def configure_logging() -> None:
    """Send every log line to stdout as JSON."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def choose_request_id(header_value: str | None) -> str:
    """Use the incoming `X-Request-ID` when it is safe. Otherwise make a new id.

    A value with a newline is rejected so it cannot break the JSON log line.
    """
    if header_value:
        cleaned = header_value.strip()
        if (
            cleaned
            and len(cleaned) <= _MAX_REQUEST_ID_LENGTH
            and "\n" not in cleaned
            and "\r" not in cleaned
        ):
            return cleaned
    return str(uuid.uuid4())


def document_id_from_path(path: str) -> str | None:
    """Return the document id when the path is `/documents/{id}`."""
    prefix = "/documents/"
    if not path.startswith(prefix):
        return None
    raw_id = path[len(prefix) :].split("/", 1)[0]
    try:
        return str(uuid.UUID(raw_id))
    except ValueError:
        return None


def user_id_from_authorization_header(authorization: str | None) -> str | None:
    """Read the user id from a Bearer token. Return None if the header is missing or bad.

    The token itself is not returned and is not logged.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return read_user_id_from_token(token.strip())


def add_request_logging(app: FastAPI) -> None:
    """Log each request and return `X-Request-ID` on the response.

    The id is also stored on `request.state.request_id` so the upload route
    can put it on the queue job.
    """

    @app.middleware("http")
    async def log_each_request(request: Request, call_next):
        request_id = choose_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            extra: dict[str, object] = {
                "request_id": request_id,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": latency_ms,
            }
            user_id = user_id_from_authorization_header(request.headers.get("authorization"))
            if user_id:
                extra["user_id"] = user_id
            document_id = document_id_from_path(request.url.path)
            if document_id:
                extra["document_id"] = document_id
            # The message stays the word "request". The body is not logged.
            logger.info("request", extra=extra)

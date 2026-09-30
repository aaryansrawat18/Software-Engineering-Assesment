"""Health routes. No token. No connection strings in the response.

`GET /health/live` only checks that this process can answer. The host uses
it as the restart probe, so a Redis blip must not fail this route.

`GET /health` and `GET /health/ready` check Postgres, the vector extension,
Redis, and the worker. Any failed check is HTTP 503 and is named in JSON.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import func, select, text

from app.db import SessionLocal, engine
from app.models import Document
from app.services import redis_client
from app.services.heartbeat import SWEEP_TIMEOUT_MINUTES, WORKER_HEARTBEAT_KEY

router = APIRouter(tags=["health"])

CHECK_OK = "ok"
CHECK_DOWN = "down"


def _database_check() -> str:
    """Postgres answers `SELECT 1`."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return CHECK_DOWN
    return CHECK_OK


def _vector_check() -> str:
    """The `vector` extension is installed. This is the vector store check."""
    try:
        with engine.connect() as connection:
            extension_row = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
    except Exception:
        return CHECK_DOWN
    if extension_row is None:
        return CHECK_DOWN
    return CHECK_OK


def _redis_check() -> str:
    """Redis answers `PING`."""
    try:
        if redis_client.get_redis().ping():
            return CHECK_OK
    except RedisError:
        return CHECK_DOWN
    return CHECK_DOWN


def _oldest_queued_document_is_stuck() -> bool:
    """True when a document has been `queued` longer than the sweep window.

    A live worker should pick that document up. An old queued row means the
    queue is not being drained.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SWEEP_TIMEOUT_MINUTES)
    with SessionLocal() as session:
        oldest_created_at = session.scalar(
            select(func.min(Document.created_at)).where(Document.status == "queued")
        )
    if oldest_created_at is None:
        return False
    if oldest_created_at.tzinfo is None:
        oldest_created_at = oldest_created_at.replace(tzinfo=timezone.utc)
    return oldest_created_at < cutoff


def _worker_check() -> str:
    """The heartbeat key exists, and no queued document is past the sweep timeout."""
    try:
        heartbeat_is_present = bool(redis_client.get_redis().exists(WORKER_HEARTBEAT_KEY))
    except RedisError:
        return CHECK_DOWN
    if not heartbeat_is_present:
        return CHECK_DOWN
    try:
        if _oldest_queued_document_is_stuck():
            return CHECK_DOWN
    except Exception:
        return CHECK_DOWN
    return CHECK_OK


def readiness_body() -> tuple[int, dict]:
    """Run every dependency check. Return the HTTP status and the JSON body.

    The body only contains `ok` or `down`. It never contains a URL or a password.
    """
    checks = {
        "database": _database_check(),
        "vector": _vector_check(),
        "redis": _redis_check(),
        "worker": _worker_check(),
    }
    everything_ok = all(result == CHECK_OK for result in checks.values())
    status_code = 200 if everything_ok else 503
    status_name = CHECK_OK if everything_ok else CHECK_DOWN
    return status_code, {"status": status_name, "checks": checks}


@router.get("/health/live")
def live() -> dict[str, str]:
    """Process is up. This route does not call Postgres, Redis, or the worker."""
    return {"status": CHECK_OK}


@router.get("/health")
@router.get("/health/ready")
def ready() -> JSONResponse:
    """Same checks for `/health` and `/health/ready`."""
    status_code, body = readiness_body()
    return JSONResponse(status_code=status_code, content=body)

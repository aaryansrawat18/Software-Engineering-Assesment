"""Background worker that turns an uploaded file into chunks.

The API enqueues the document id only. This process loads that row, marks it
processing, and runs the ingest graph. While it runs, it writes a heartbeat
key so `/health` can see that a worker is alive.

Start it with the same REDIS_URL as the API:

    python -m app.worker

The rq command works too, if you pass this class so the sweep and heartbeat run:

    rq worker --url $REDIS_URL --worker-class app.worker.IngestWorker
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone

from rq import Worker, get_current_job
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Document
from app.services import redis_client
from app.services.heartbeat import SWEEP_TIMEOUT_MINUTES, write_heartbeat_until_stopped
from app.services.queue import enqueue_ingest

logger = logging.getLogger(__name__)

# A document left in `processing` longer than this is put back on the queue.
STUCK_PROCESSING_MINUTES = SWEEP_TIMEOUT_MINUTES


def _log_fields(document_id: str) -> dict[str, str]:
    """Fields for one worker log line. Always the document id.

    The request id is included only when the API stored it on the RQ job.
    """
    fields = {"document_id": document_id}
    job = get_current_job()
    if job is None:
        return fields
    request_id = job.meta.get("request_id")
    if request_id:
        fields["request_id"] = str(request_id)
    return fields


def ingest(document_id: str) -> None:
    """RQ job. `document_id` is a string. The file is read from disk, not from Redis."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        return

    with SessionLocal() as session:
        document = session.get(Document, parsed_id)
        if document is None:
            return
        document.status = "processing"
        document.attempt_count = (document.attempt_count or 0) + 1
        document.processing_started_at = datetime.now(timezone.utc)
        session.commit()

    logger.info("ingest started", extra=_log_fields(document_id))
    # Imported here so the API process does not load the graph just to enqueue.
    from app.graphs.ingest import graph

    graph.invoke({"document_id": document_id})


def requeue_stuck_documents() -> None:
    """Put documents stuck in `processing` back to `queued` and enqueue their ids.

    Running the graph again is safe: the ready step replaces chunks for that id.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_PROCESSING_MINUTES)
    with SessionLocal() as session:
        stuck_documents = session.scalars(
            select(Document).where(
                Document.status == "processing",
                Document.processing_started_at.is_not(None),
                Document.processing_started_at < cutoff,
            )
        ).all()
        document_ids = []
        for document in stuck_documents:
            document.status = "queued"
            document_ids.append(str(document.id))
        session.commit()

    for document_id in document_ids:
        logger.info("requeue stuck document", extra={"document_id": document_id})
        enqueue_ingest(document_id)


def _start_heartbeat() -> threading.Event:
    """Start the heartbeat thread and return the event that stops it."""
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=write_heartbeat_until_stopped,
        args=(stop_event,),
        name="worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    return stop_event


class IngestWorker(Worker):
    """RQ worker that requeues stuck documents once, then writes a heartbeat."""

    def work(self, *args, **kwargs):
        # Same JSON logs as the API. Imported here so tests that only call
        # `ingest` do not reset the logging setup.
        from app.logging import configure_logging

        configure_logging()
        requeue_stuck_documents()
        stop_heartbeat = _start_heartbeat()
        try:
            return super().work(*args, **kwargs)
        finally:
            stop_heartbeat.set()


def main() -> None:
    """Run the startup sweep, then listen on the default Redis queue."""
    worker = IngestWorker(["default"], connection=redis_client.get_redis())
    worker.work()


if __name__ == "__main__":
    main()

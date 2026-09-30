"""Background worker that turns an uploaded file into chunks.

The API enqueues the document id only. This process loads that row, marks it
processing, and runs the ingest graph.

Start it with the same REDIS_URL as the API:

    python -m app.worker

The rq command works too, if you pass this class so the startup sweep runs:

    rq worker --worker-class app.worker.IngestWorker
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from redis import Redis
from rq import Worker
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Document
from app.services.queue import enqueue_ingest

logger = logging.getLogger(__name__)

# A document left in `processing` longer than this is put back on the queue.
STUCK_PROCESSING_MINUTES = 10


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

    logger.info("ingest started", extra={"document_id": document_id})
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


class IngestWorker(Worker):
    """RQ worker that requeues stuck documents once, when the process starts."""

    def work(self, *args, **kwargs):
        requeue_stuck_documents()
        return super().work(*args, **kwargs)


def main() -> None:
    """Run the startup sweep, then listen on the default Redis queue."""
    redis_connection = Redis.from_url(get_settings().redis_url)
    worker = IngestWorker(["default"], connection=redis_connection)
    worker.work()


if __name__ == "__main__":
    main()

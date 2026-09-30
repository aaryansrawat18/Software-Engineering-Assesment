"""Put a document id on the Redis queue.

The file bytes stay on disk. The worker receives only the id string and
opens `{UPLOAD_DIR}/{document_id}` itself. The request id, when we have one,
is stored on the job so the worker can log the same id.
"""

import logging

from rq import Queue

from app.services import redis_client

logger = logging.getLogger(__name__)

# Dotted path of the job function. RQ imports it inside the worker process.
INGEST_JOB_PATH = "app.worker.ingest"


def enqueue_ingest(document_id: str, request_id: str | None = None) -> None:
    """Add one ingest job. `document_id` must be a string, not the file.

    `request_id` is optional. The worker logs it when it is present, and
    otherwise logs only the document id.
    """
    job_queue = Queue(connection=redis_client.get_redis())
    job = job_queue.enqueue(INGEST_JOB_PATH, document_id)
    if not request_id:
        return
    try:
        job.meta["request_id"] = request_id
        job.save_meta()
    except Exception:
        # The job is already queued. Missing the request id only affects the log line.
        logger.warning("could not store request id on the ingest job")

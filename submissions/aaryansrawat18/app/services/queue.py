"""Put a document id on the Redis queue.

The file bytes stay on disk. The worker receives only the id string and
opens `{UPLOAD_DIR}/{document_id}` itself.
"""

from redis import Redis
from rq import Queue

from app.config import get_settings

# Dotted path of the job function. RQ imports it inside the worker process.
INGEST_JOB_PATH = "app.worker.ingest"


def enqueue_ingest(document_id: str) -> None:
    """Add one ingest job. `document_id` must be a string, not the file."""
    redis_connection = Redis.from_url(get_settings().redis_url)
    job_queue = Queue(connection=redis_connection)
    job_queue.enqueue(INGEST_JOB_PATH, document_id)

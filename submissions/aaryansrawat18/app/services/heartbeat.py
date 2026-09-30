"""The worker proves it is alive by writing one Redis key.

`GET /health` looks for `worker:heartbeat`. If the key is missing, the
worker process is treated as down. The key expires on its own, so a crashed
worker disappears without an extra cleanup step.
"""

import logging
import threading

from redis.exceptions import RedisError

from app.services import redis_client

logger = logging.getLogger(__name__)

# Health checks and the worker must use this exact key name.
WORKER_HEARTBEAT_KEY = "worker:heartbeat"
# A missed write older than this means the worker stopped.
WORKER_HEARTBEAT_TTL_SECONDS = 15
# How often the worker refreshes the key. Shorter than the TTL, so a live
# worker never lets the key expire.
WORKER_HEARTBEAT_EVERY_SECONDS = 5
# Same window the worker uses when it requeues a stuck document.
SWEEP_TIMEOUT_MINUTES = 10


def write_heartbeat_until_stopped(stop_event: threading.Event) -> None:
    """Write `worker:heartbeat` until `stop_event` is set.

    The first write happens immediately, then again every few seconds.
    A Redis error is logged and the loop keeps trying.
    """
    connection = redis_client.get_redis()
    while True:
        try:
            connection.set(WORKER_HEARTBEAT_KEY, "alive", ex=WORKER_HEARTBEAT_TTL_SECONDS)
        except RedisError:
            logger.warning("could not write worker heartbeat")
        if stop_event.wait(WORKER_HEARTBEAT_EVERY_SECONDS):
            return

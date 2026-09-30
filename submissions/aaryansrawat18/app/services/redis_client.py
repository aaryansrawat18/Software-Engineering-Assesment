"""Open Redis from the `REDIS_URL` setting.

The question rate limit, the worker heartbeat, and the health check all use
this helper. Tests replace `get_redis` with a small in-memory stand-in.
"""

from redis import Redis

from app.config import get_settings


def get_redis() -> Redis:
    """Return a Redis connection. Callers share the URL, not one socket."""
    return Redis.from_url(get_settings().redis_url)

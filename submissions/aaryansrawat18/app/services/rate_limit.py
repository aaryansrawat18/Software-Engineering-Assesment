"""Stop one user from asking too many questions.

Only `POST /questions` uses this. Upload and status polling stay unlimited
so a reviewer can still watch a document become ready.
"""

from datetime import datetime, timezone
from uuid import UUID

from redis.exceptions import RedisError

from app.services import redis_client

# The brief allows 10 questions per user per minute.
QUESTIONS_PER_MINUTE = 10
# The Redis key expires after one minute so the next minute starts at zero.
RATE_LIMIT_WINDOW_SECONDS = 60


class RateLimitUnavailable(Exception):
    """Redis could not count the question. The route turns this into HTTP 503."""


def question_is_allowed(user_id: UUID) -> bool:
    """Count one question for this user. Return False when they are over the cap.

    The key is `rl:{user_id}:{current_minute}`. `INCR` adds one. The key
    expires after 60 seconds. The 11th question in that minute is refused.
    """
    current_minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    redis_key = f"rl:{user_id}:{current_minute}"
    try:
        connection = redis_client.get_redis()
        question_count = int(connection.incr(redis_key))
        connection.expire(redis_key, RATE_LIMIT_WINDOW_SECONDS)
    except RedisError as error:
        raise RateLimitUnavailable from error
    return question_count <= QUESTIONS_PER_MINUTE

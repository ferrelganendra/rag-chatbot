"""Retry and timeout utilities for RAG pipeline resilience."""

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Exponential backoff: 1s -> 2s -> 4s -> 8s, max 3 attempts
DEFAULT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def with_retry(func=None, *, max_attempts: int = 3):
    """Decorator: retry on transient errors with exponential backoff."""
    if func is not None:
        return DEFAULT_RETRY(func)

    def decorator(f):
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(f)

    return decorator

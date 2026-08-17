"""
Resilient Retry Utility with Exponential Backoff and Jitter for J.A.R.V.I.S.
Guards external LLM invocations, network scrapers, and API requests against transient failures.
"""

import functools
import logging
import random
import time
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger(__name__)
_jitter_random = random.SystemRandom()


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff and jitter upon catching specified exceptions.

    :param max_retries: Maximum number of retry attempts before giving up.
    :param initial_delay: Initial sleep delay in seconds.
    :param backoff_factor: Multiplier for exponential growth.
    :param max_delay: Upper cap on sleep delay in seconds.
    :param jitter: Whether to add random uniform jitter (±20%) to avoid thundering herd.
    :param retryable_exceptions: Tuple of exception classes that trigger a retry.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            current_delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_retries} retries. "
                            f"Final error: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    sleep_time = min(current_delay, max_delay)
                    if jitter:
                        sleep_time = sleep_time * (0.8 + 0.4 * _jitter_random.random())

                    logger.warning(
                        f"Transient failure in '{func.__name__}' (attempt {attempt}/{max_retries}): "
                        f"{type(e).__name__}: {str(e)}. Retrying in {sleep_time:.2f}s..."
                    )

                    time.sleep(sleep_time)
                    current_delay *= backoff_factor

        return wrapper

    return decorator

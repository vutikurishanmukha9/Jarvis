"""
Tests for exponential backoff and jittered retry utility in src/core/retry_utils.py.
"""

import pytest

from src.core.retry_utils import retry_with_backoff


def test_retry_success_on_first_attempt():
    """Verify function executes once and returns cleanly on immediate success."""
    calls = 0

    @retry_with_backoff(max_retries=3, initial_delay=0.01)
    def quick_task():
        nonlocal calls
        calls += 1
        return "SUCCESS"

    res = quick_task()
    assert res == "SUCCESS"
    assert calls == 1


def test_retry_recovers_after_transient_failure():
    """Verify function retries after failures and returns value once recovered."""
    calls = 0

    @retry_with_backoff(max_retries=3, initial_delay=0.01, retryable_exceptions=(ValueError,))
    def flaky_task():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("Temporary network glitch")
        return "RECOVERED"

    res = flaky_task()
    assert res == "RECOVERED"
    assert calls == 3


def test_retry_exhaustion_raises_final_exception():
    """Verify function raises exception once max retries are exhausted."""
    calls = 0

    @retry_with_backoff(max_retries=2, initial_delay=0.01, retryable_exceptions=(RuntimeError,))
    def failing_task():
        nonlocal calls
        calls += 1
        raise RuntimeError("Permanent API outage")

    with pytest.raises(RuntimeError) as exc_info:
        failing_task()

    assert "Permanent API outage" in str(exc_info.value)
    assert calls == 3  # initial + 2 retries

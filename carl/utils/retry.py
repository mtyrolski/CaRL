from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    attempts: int = 5
    delay_seconds: float = 10.0
    backoff: float = 1.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    on_retry: Callable[[int, BaseException], None] | None = None


def retry(config: RetryConfig) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator with optional backoff.

    The wrapped function is executed up to `attempts` times.
    After each failure (except the last), it sleeps for `delay_seconds` and
    multiplies the delay by `backoff`.
    """

    if config.attempts < 1:
        raise ValueError("RetryConfig.attempts must be >= 1")
    if config.delay_seconds < 0:
        raise ValueError("RetryConfig.delay_seconds must be >= 0")
    if config.backoff < 1:
        raise ValueError("RetryConfig.backoff must be >= 1")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = config.delay_seconds
            last_exc: BaseException | None = None
            for attempt in range(1, config.attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except config.retry_on as exc:
                    last_exc = exc
                    if attempt >= config.attempts:
                        raise
                    if config.on_retry is not None:
                        config.on_retry(attempt, exc)
                    if delay > 0:
                        time.sleep(delay)
                    delay *= config.backoff
            assert last_exc is not None
            raise last_exc

        return wrapped

    return decorator

import pytest

from carl.utils.retry import RetryConfig
from carl.utils.retry import retry


def test_retry_succeeds_after_transient_failures():
    attempts: list[int] = []
    callback_attempts: list[int] = []

    @retry(
        RetryConfig(
            attempts=4,
            delay_seconds=0.0,
            backoff=1.0,
            retry_on=(ValueError,),
            on_retry=lambda attempt, _exc: callback_attempts.append(attempt),
        )
    )
    def flaky() -> int:
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("temporary")
        return 7

    assert flaky() == 7
    assert len(attempts) == 3
    assert callback_attempts == [1, 2]


def test_retry_raises_after_exhausting_attempts():
    attempts: list[int] = []

    @retry(
        RetryConfig(
            attempts=3,
            delay_seconds=0.0,
            backoff=1.0,
            retry_on=(RuntimeError,),
        )
    )
    def always_fails() -> None:
        attempts.append(1)
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="still broken"):
        always_fails()
    assert len(attempts) == 3


def test_retry_uses_backoff_for_sleep(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr("carl.utils.retry.time.sleep", lambda value: sleep_calls.append(value))

    @retry(
        RetryConfig(
            attempts=3,
            delay_seconds=1.0,
            backoff=2.0,
            retry_on=(ValueError,),
        )
    )
    def always_fails() -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        always_fails()

    assert sleep_calls == [1.0, 2.0]


@pytest.mark.parametrize(
    "config",
    [
        RetryConfig(attempts=0),
        RetryConfig(delay_seconds=-0.1),
        RetryConfig(backoff=0.5),
    ],
)
def test_retry_rejects_invalid_configuration(config: RetryConfig):
    with pytest.raises(ValueError):
        retry(config)

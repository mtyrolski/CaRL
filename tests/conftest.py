"""Keep pre-existing fake transports isolated from the real campaign ledger."""
import pytest


class OfflineLedger:
    """No-op ledger for tests that do not exercise budget accounting."""

    def event(self, *_args, **_kwargs):
        pass

    def reserve(self, *_args, **_kwargs):
        return "offline-test"

    def settle(self, *_args, **_kwargs):
        return True

    def mark_unknown(self, *_args, **_kwargs):
        pass


@pytest.fixture(autouse=True)
def isolate_campaign_ledger(monkeypatch):
    """Never use the user's real ledger during existing mock-transport tests."""
    import carl.icl.api as api

    monkeypatch.setattr(api, "_campaign_ledger", OfflineLedger)


@pytest.fixture
def deterministic_token_estimates(monkeypatch):
    """Provide repeatable token counts when tokenisation itself is not under test."""
    from carl.icl import experiment_api

    class ByteEncoding:
        def encode(self, text, **_kwargs):
            return list(text.encode("utf-8"))

    monkeypatch.setattr(experiment_api.tiktoken, "get_encoding", lambda _name: ByteEncoding())

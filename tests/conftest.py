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

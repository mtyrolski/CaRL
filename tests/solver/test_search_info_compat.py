"""Compatibility tests for public and saved search metadata."""

import pickle
from dataclasses import asdict

from carl.planners.base import SearchInfo


def test_legacy_positional_counters() -> None:
    info = SearchInfo("solved", 5, 2, 1)

    assert (info.low_level_nodes_visited, info.high_level_nodes_valid, info.high_level_nodes_unreachable) == (5, 2, 1)
    assert info.generator_mode is None
    assert info.runtime_seconds is None


def test_restoring_legacy_search_info_initializes_missing_diagnostics() -> None:
    info = SearchInfo("solved", low_level_nodes_visited=5)
    names = ("realized_segment_lengths", "progress_per_segment", "proposal_events")
    for name in names:
        delattr(info, name)
    saved = pickle.dumps(info)

    first = pickle.loads(saved)
    second = pickle.loads(saved)
    assert first.low_level_nodes_visited == 5
    fields = asdict(first)
    assert all(fields[name] == [] for name in names)
    for name in names:
        assert getattr(first, name) is not getattr(second, name)

    first.proposal_events.append({"event_id": 0})
    assert pickle.loads(pickle.dumps(first)).proposal_events == [{"event_id": 0}]

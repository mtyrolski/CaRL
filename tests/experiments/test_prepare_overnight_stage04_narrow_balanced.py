"""Tests for matching narrow layout contributions without changing the sampler."""

from collections import Counter

import pytest

from scripts.prepare_overnight_stage04_narrow_balanced import QUOTAS
from scripts.prepare_overnight_stage04_narrow_balanced import select_exact


def test_exact_flow_matches_quotas_and_uses_28_or_29_examples_per_board():
    pools = {f'layout_{index}': {kind: [(i, 1, kind, {}) for i in range(256)] for kind in QUOTAS}
             for index in range(9)}
    selected, extra = select_exact(pools, QUOTAS)
    assert len(extra) == 4
    assert sorted(len(rows) for rows in selected.values()) == [28] * 5 + [29] * 4
    assert Counter(item[2] for rows in selected.values() for item in rows) == Counter(QUOTAS)
    assert select_exact(pools, QUOTAS) == (selected, extra)


def test_exact_flow_fails_if_any_board_lacks_available_examples():
    pools = {f'layout_{index}': {kind: [(i, 1, kind, {}) for i in range(256)] for kind in QUOTAS}
             for index in range(9)}
    pools['layout_0'] = {kind: [] for kind in QUOTAS}
    with pytest.raises(ValueError, match='No exact'):
        select_exact(pools, QUOTAS)

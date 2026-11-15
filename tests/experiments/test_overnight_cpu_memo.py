"""Check deterministic decision memoisation without running a model."""

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.data_sources.sokoban import environment
from scripts.figure1_cllp_matched import Pair
from scripts.figure1_cllp_matched import rollout_pairs
from scripts.overnight_cpu_memo import MemoPredictor


@pytest.fixture
def boards():
    a = from_rows(["#####", "#@--#", "#-$-#", "#-.-#", "#####"])
    env = environment(a)
    b = np.asarray(env.next_state(a, 3))
    c = np.asarray(env.next_state(a, 1))
    return a, b, c


def test_deduplicates_batch_and_across_steps_preserving_order(boards):
    a, b, c = boards
    calls = []

    def predict(states, targets):
        calls.append((len(states), [state_hash(target) for target in targets]))
        return [1 if state_hash(state) == state_hash(a) else 3 for state in states]

    memo = MemoPredictor(predict, conditioned=True, batch_size=1)
    assert memo([a, a, b, a], [c, c, c, c]) == [1, 1, 3, 1]
    assert len(calls) == 2 and memo.stats.unique_decisions == 2 and memo.stats.forward_batches == 2
    assert memo.stats.same_batch_hits == 2 and memo.stats.previous_cache_hits == 0
    assert memo([a, b, a], [c, c, c]) == [1, 3, 1]
    assert len(calls) == 2 and memo.stats.requests == 7 and memo.stats.previous_cache_hits == 3


def test_conditioned_keys_include_target_but_policy_keys_do_not(boards):
    a, b, c = boards
    goal_sensitive = MemoPredictor(lambda states, targets: [1 if state_hash(t) == state_hash(b) else 2 for t in targets],
                                   conditioned=True, batch_size=8)
    assert goal_sensitive([a, a], [b, c]) == [1, 2]
    assert goal_sensitive.stats.unique_decisions == 2
    ordinary = MemoPredictor(lambda states, _targets: [3] * len(states), conditioned=False, batch_size=8)
    assert ordinary([a, a], [b, c]) == [3, 3]
    assert ordinary.stats.unique_decisions == 1 and ordinary.stats.same_batch_hits == 1


def test_lru_eviction_keeps_results_for_current_batch(boards):
    a, b, c = boards
    memo = MemoPredictor(lambda states, _targets: list(range(len(states))), conditioned=True,
                         batch_size=8, max_entries=1)
    assert memo([a, b, c], [a, b, c]) == [0, 1, 2]
    assert len(memo.cache) == 1 and memo.stats.evictions == 2 and memo.stats.peak_entries == 1


def test_noop_rollout_still_counts_every_attempt(boards):
    source, target, _ = boards
    pair = Pair("p", canonical_layout_id(source), "t", 0, 0, 1, state_hash(source), state_hash(target), source, target)
    plain = rollout_pairs([pair], lambda states, _targets: [0] * len(states), "cllp8", cap=4)[0]
    memo = MemoPredictor(lambda states, _targets: [0] * len(states), conditioned=True, batch_size=4)
    cached = rollout_pairs([pair], memo, "cllp8", cap=4)[0]
    assert cached == plain
    assert cached["attempted_actions"] == cached["no_op_count"] == 4
    assert memo.stats.requests == 4 and memo.stats.unique_decisions == 1 and memo.stats.previous_cache_hits == 3


def test_invalid_inputs_and_model_actions_fail_closed(boards):
    a, b, _ = boards
    with pytest.raises(ValueError, match="positive"):
        MemoPredictor(lambda _s, _t: [], conditioned=True, batch_size=0)
    memo = MemoPredictor(lambda _s, _t: [4], conditioned=True, batch_size=1)
    with pytest.raises(ValueError, match="one target"):
        memo([a], [])
    with pytest.raises(ValueError, match="invalid actions"):
        memo([a], [b])
    assert len(memo.cache) == 0

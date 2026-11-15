"""Memoise deterministic CPU policy decisions without skipping simulator steps.

Create a fresh instance per frozen controller and board shard. CLLP cache keys
contain both complete states; ordinary-policy keys contain only the current
state because that checkpoint does not receive the target.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from carl.icl.boards import state_hash


@dataclass
class MemoStats:
    requests: int = 0
    previous_cache_hits: int = 0
    same_batch_hits: int = 0
    unique_decisions: int = 0
    forward_batches: int = 0
    peak_entries: int = 0
    evictions: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class MemoPredictor:
    """Cache greedy decisions within one board/model, preserving input order.

    The wrapped predictor is invoked in explicit sub-batches no larger than
    ``batch_size``. This records actual model-forward batches when the wrapped
    CaRL predictor performs one forward per call of at most that size.
    """

    def __init__(self, predictor: Callable[[list[np.ndarray], list[np.ndarray]], Sequence[int]], *,
                 conditioned: bool, batch_size: int, max_entries: int = 8192) -> None:
        if batch_size < 1 or max_entries < 1:
            raise ValueError("batch_size and max_entries must be positive")
        self.predictor = predictor
        self.conditioned = conditioned
        self.batch_size = batch_size
        self.max_entries = max_entries
        self.cache: OrderedDict[tuple[str, ...], int] = OrderedDict()
        self.stats = MemoStats()

    def _key(self, state: np.ndarray, target: np.ndarray) -> tuple[str, ...]:
        source = state_hash(state)
        return (source, state_hash(target)) if self.conditioned else (source,)

    def __call__(self, states: list[np.ndarray], targets: list[np.ndarray]) -> list[int]:
        if len(states) != len(targets):
            raise ValueError("Expected one target per current state")
        keys = [self._key(state, target) for state, target in zip(states, targets)]
        missing: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        resolved: dict[tuple[str, ...], int] = {}
        previous_hits = same_batch_hits = 0
        for key, state, target in zip(keys, states, targets):
            if key in self.cache:
                resolved[key] = self.cache[key]
                self.cache.move_to_end(key)
                previous_hits += 1
            elif key in missing:
                same_batch_hits += 1
            else:
                missing[key] = (state, target)
        requests = list(missing.items())
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start:start + self.batch_size]
            actions = list(self.predictor([entry[1][0] for entry in chunk], [entry[1][1] for entry in chunk]))
            self.stats.forward_batches += 1
            if len(actions) != len(chunk) or any(
                not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_))
                or int(action) not in range(4) for action in actions
            ):
                raise ValueError("Predictor returned invalid actions")
            for (key, _), action in zip(chunk, actions):
                value = int(action)
                resolved[key] = value
                self.cache[key] = value
                self.cache.move_to_end(key)
                if len(self.cache) > self.max_entries:
                    self.cache.popitem(last=False)
                    self.stats.evictions += 1
                self.stats.peak_entries = max(self.stats.peak_entries, len(self.cache))
        self.stats.requests += len(keys)
        self.stats.previous_cache_hits += previous_hits
        self.stats.same_batch_hits += same_batch_hits
        self.stats.unique_decisions += len(requests)
        return [resolved[key] for key in keys]

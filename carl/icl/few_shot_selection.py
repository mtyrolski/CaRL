"""Deterministic, simulator-verified policy demonstrations from training only.

The 8, 32 or 64 examples are drawn from training layouts and can be correlated.
Evaluator metadata stays outside the actual model payload.
"""
import hashlib
import json
from collections import Counter
from functools import lru_cache
from typing import TypedDict

import numpy as np

from carl.icl.boards import canonical_layout_id, from_rows, state_hash, tiles, to_rows, validate_subgoal
from carl.icl.data_sources.sokoban import environment

VERSION = "diverse_training_v2"
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OFFSETS = (1, 2, 4, 8, 12, 16)
SELECTION_SEED = 20261104
# Fixed round-robin ensures 8 is a prefix of 32, itself a prefix of 64.
CATEGORIES = tuple((action, push) for action in range(4) for push in (False, True))


class _Candidate(TypedDict):
    context_index: int
    layout: str
    anchor: int
    offset: int
    source_hash: str
    target_hash: str
    rank: str


def _push(state, action):
    board = tiles(state)
    player = np.argwhere(np.isin(board, [5, 6]))
    if len(player) != 1:
        raise ValueError("Training state must contain exactly one player")
    row, col = player[0]
    dr, dc = DELTAS[action]
    row, col = int(row + dr), int(col + dc)
    if not (0 <= row < board.shape[0] and 0 <= col < board.shape[1]):
        raise ValueError("Training action leaves the board")
    return int(board[row, col]) in (3, 4)


def _rank(*parts):
    return hashlib.sha256(json.dumps([VERSION, SELECTION_SEED, *parts], separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=8)
def _select(serialized, count):
    if count not in (8, 32, 64):
        raise ValueError("diverse_training_v2 requires exactly 8, 32 or 64 demonstrations")
    contexts = json.loads(serialized).get("contexts", [])
    if len(contexts) < 8:
        raise ValueError("diverse_training_v2 requires at least eight training contexts")
    if len({context["id"] for context in contexts}) != len(contexts):
        raise ValueError("Training context IDs must be unique")
    prepared = []
    for context in contexts:
        candidates = sorted(context["candidates"], key=lambda item: item["k"])
        if len(candidates) < 16 or [x["k"] for x in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError("Training trajectories require contiguous action offsets through 16")
        actions = candidates[-1]["actions"]
        if (len(actions) != len(candidates) or any(not isinstance(a, int) or isinstance(a, bool) or a not in range(4) for a in actions)
                or any(item["actions"] != actions[:item["k"]] for item in candidates)):
            raise ValueError("Training trajectory witness prefixes are inconsistent")
        states = [from_rows(context["state"]), *(from_rows(item["subgoal"]) for item in candidates)]
        # Replay all prefixes from the original state; a fabricated mid-trajectory
        # anchor cannot enter the demonstration pool.
        env = environment(states[0])
        current = states[0].copy()
        for index, action in enumerate(actions):
            if env.is_solved(current):
                raise ValueError("Training witness continues after task terminal")
            nxt = np.asarray(env.next_state(current, action))
            if np.array_equal(current, nxt) or not np.array_equal(nxt, states[index + 1]):
                raise ValueError("Training witness prefix fails exact simulator replay")
            current = nxt
        prepared.append((context, actions, states, canonical_layout_id(states[0])))

    if len({item[3] for item in prepared}) < 8:
        raise ValueError("diverse_training_v2 requires at least eight distinct canonical training layouts")

    pools: dict[tuple[int, bool], list[_Candidate]] = {category: [] for category in CATEGORIES}
    seen_pool = set()
    for ci, (context, actions, states, layout) in enumerate(prepared):
        for anchor, action in enumerate(actions):
            source = states[anchor]
            source_hash = state_hash(source)
            push = _push(source, action)
            for offset in OFFSETS:
                if anchor + offset >= len(states):
                    continue
                target_hash = state_hash(states[anchor + offset])
                pair = (source_hash, target_hash)
                if source_hash == target_hash or pair in seen_pool:
                    continue
                seen_pool.add(pair)
                pools[(action, push)].append({"context_index": ci, "layout": layout,
                    "anchor": anchor, "offset": offset, "source_hash": source_hash,
                    "target_hash": target_hash, "rank": _rank(context["id"], anchor, offset)})

    selected = []
    used_pairs, used_sources = set(), set()
    layout_counts: Counter[str] = Counter()
    horizon_counts: Counter[int] = Counter()
    category_layout_counts: Counter[tuple[tuple[int, bool], str]] = Counter()
    for index in range(count):
        category = CATEGORIES[index % len(CATEGORIES)]
        candidates = [item for item in pools[category] if (item["source_hash"], item["target_hash"]) not in used_pairs]
        if not candidates:
            raise ValueError(f"Insufficient unique training pairs for category {category} after {index} demonstrations")
        # Prefer previously unseen sources, then underrepresented horizons and
        # layouts; the final hash is a fixed seed tie-break, not outcome tuning.
        def candidate_rank(candidate: _Candidate, category: tuple[int, bool] = category) -> tuple[bool, int, int, int, str]:
            layout = candidate["layout"]
            return (
                candidate["source_hash"] in used_sources, horizon_counts[candidate["offset"]], layout_counts[layout],
                category_layout_counts[(category, layout)], candidate["rank"],
            )

        item = min(candidates, key=candidate_rank)
        selected.append((category, item))
        used_pairs.add((item["source_hash"], item["target_hash"]))
        used_sources.add(item["source_hash"])
        layout_counts[item["layout"]] += 1
        horizon_counts[item["offset"]] += 1
        category_layout_counts[(category, item["layout"])] += 1

    result = []
    for (action, push), item in selected:
        context, actions, states, layout = prepared[item["context_index"]]
        anchor, offset = item["anchor"], item["offset"]
        source, target = states[anchor], states[anchor + offset]
        validate_subgoal(source, target)
        # All preceding states have already been independently replayed above.
        probabilities = [0.0] * 4
        probabilities[action] = 1.0
        result.append({"question": {"state": to_rows(source), "subgoal": to_rows(target)},
                       "answer": {"probabilities": probabilities},
                       "metadata": {"selector": VERSION, "selection_seed": SELECTION_SEED,
                                    "trajectory_id": context["id"], "layout_id": layout,
                                    "anchor_index": anchor, "target_index": anchor + offset, "k_actions": offset,
                                    "source_hash": item["source_hash"], "target_hash": item["target_hash"],
                                    "first_action": ACTION_NAMES[action], "push": push}})
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def diverse_training_examples(training, count=32):
    """Return 8, 32 or 64 examples plus evaluator metadata in deterministic order."""
    key = json.dumps(training, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(_select(key, count))

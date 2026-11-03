"""Durable physical-action traces and witness-free replay verification."""
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.datasets import environment
from carl.icl.search import SearchResult
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.types import BoardEnvironment
from carl.utils.aliases import State


def verify_tree(tree: SearchTreeRecorder, initial: np.ndarray, budget: int) -> dict[str, Any]:
    """Replay recorded actions from actual source nodes; never access a witness."""
    if tree.result is None or not tree.nodes:
        raise ValueError('Missing search trace')
    env = environment(initial)
    if tree.nodes[0].state != to_rows(initial):
        raise ValueError('Trace root differs from initial board')
    actions = 0
    for node in tree.nodes[1:]:
        if node.parent_id is None or node.parent_id >= node.node_id:
            raise ValueError('Invalid trace parent')
        current = from_rows(tree.nodes[node.parent_id].state)
        for index, action in enumerate(node.actions):
            if action not in range(4) or env.is_solved(current):
                raise ValueError('Illegal trace action or post-terminal step')
            current = np.asarray(env.next_state(current, action))
            if env.is_solved(current) and index + 1 < len(node.actions):
                raise ValueError('Trace continued after solving')
        if to_rows(current) != node.state or bool(env.is_solved(current)) != node.solved:
            raise ValueError('Trace endpoint or solved flag does not replay')
        if node.proposal is not None and node.kind == 'subgoal' and node.reached != np.array_equal(current, from_rows(node.proposal)):
            raise ValueError('Trace exact-hit flag does not replay')
        actions += len(node.actions)
    if actions != tree.result.attempted_actions or actions > budget:
        raise ValueError('Trace actions disagree with accounting or budget')
    if tree.result.solved:
        current = initial.copy()
        for action in tree.result.action_path:
            if env.is_solved(current):
                raise ValueError('Solution path continues after terminal')
            current = np.asarray(env.next_state(current, action))
        if not env.is_solved(current) or len(tree.result.action_path) > actions:
            raise ValueError('Reported solution did not replay')
    return {'verified': True, 'trace_actions': actions, 'trace_edges': len(tree.nodes) - 1,
            'solution_replayed': bool(tree.result.solved)}

class DurableTree(SearchTreeRecorder):
    """Append and sync each observed transition so interruptions do not erase the trace."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def on_start(self, root: Any) -> None:
        super().on_start(root)
        self.append({'event': 'start', 'node': asdict(self.nodes[0])})

    def on_expansion(self, entry: Any, index: int, actions: int, queue_size: int) -> None:
        super().on_expansion(entry, index, actions, queue_size)
        self.append({'event': 'expansion', 'expansion': asdict(self.expansions[-1])})

    def on_transition(self, transition: Any) -> None:
        super().on_transition(transition)
        self.append({'event': 'transition', 'node': asdict(self.nodes[-1])})

    def on_finish(self, result: SearchResult) -> None:
        super().on_finish(result)
        self.append({'event': 'finish', 'outcome': asdict(result), 'solution_node': self.solution_node})

class DurableEnvironment:
    """Record each executed simulator action, including incomplete rollouts."""

    def __init__(self, delegate: BoardEnvironment, tree: DurableTree) -> None:
        self.delegate = delegate
        self.tree = tree

    def next_state(self, state: State, action: int, /) -> State:
        after = self.delegate.next_state(state, action)
        self.tree.append({'event': 'action', 'action': action,
                          'source_hash': state_hash(np.asarray(state)),
                          'endpoint_hash': state_hash(np.asarray(after)),
                          'solved': bool(self.delegate.is_solved(after))})
        return after

    def is_solved(self, state: State, /) -> bool:
        return self.delegate.is_solved(state)

def verify_actions(tree: SearchTreeRecorder, initial: np.ndarray, events: list[dict[str, Any]]) -> None:
    """Check the durable per-action log against replayed search transitions."""
    saved = [event for event in events if event['event'] == 'action']
    actual = []
    env = environment(initial)
    for node in tree.nodes[1:]:
        assert node.parent_id is not None
        current = from_rows(tree.nodes[node.parent_id].state)
        for action in node.actions:
            after = np.asarray(env.next_state(current, action))
            actual.append({'event': 'action', 'action': action, 'source_hash': state_hash(current),
                           'endpoint_hash': state_hash(after), 'solved': bool(env.is_solved(after))})
            current = after
    if actual != saved:
        raise ValueError('Durable action records disagree with the replayed search tree')

"""Verify saved shared-search traces without trained model inference."""
from typing import Any

import numpy as np

from carl.icl.boards import from_rows, to_rows
from carl.icl.types import Board, BoardEnvironment


def verify_saved(env: BoardEnvironment, initial: Board, saved: dict[str, Any],
                 action_budget: int, node_budget: int) -> dict[str, Any]:
    """Replay every recorded edge and any successful solution before accepting completion."""
    nodes, outcome = saved['nodes'], saved['outcome']
    if not nodes or nodes[0]['state'] != to_rows(initial):
        raise ValueError('Saved trace root differs from this board')
    actions = 0
    for index, node in enumerate(nodes[1:], start=1):
        parent = node['parent_id']
        if not isinstance(parent, int) or not 0 <= parent < index or node['node_id'] != index:
            raise ValueError('Invalid trace parent or node index')
        current = from_rows(nodes[parent]['state'])
        for action in node['actions']:
            if not isinstance(action, int) or isinstance(action, bool) or action not in range(4) or env.is_solved(current):
                raise ValueError('Invalid action or post-terminal trace')
            current = np.asarray(env.next_state(current, action))
        if to_rows(current) != node['state'] or bool(env.is_solved(current)) != node['solved']:
            raise ValueError('Trace endpoint or solved flag differs')
        if node['proposal'] is not None and node['kind'] == 'subgoal':
            if node['reached'] != np.array_equal(current, from_rows(node['proposal'])):
                raise ValueError('Exact target claim differs')
        actions += len(node['actions'])
    if actions != outcome['attempted_actions'] or not 0 <= actions <= action_budget:
        raise ValueError('Action accounting differs')
    if len(saved['expansions']) != outcome['expansions'] or not 0 <= outcome['expansions'] <= node_budget:
        raise ValueError('Expansion accounting differs')
    if outcome['solved']:
        current = initial.copy()
        for action in outcome['action_path']:
            if action not in range(4) or env.is_solved(current):
                raise ValueError('Invalid solution action or post-terminal path')
            current = np.asarray(env.next_state(current, action))
        if not env.is_solved(current) or len(outcome['action_path']) > actions:
            raise ValueError('Claimed solution does not replay')
    return {'verified': True, 'trace_actions': actions, 'solution_replayed': bool(outcome['solved'])}

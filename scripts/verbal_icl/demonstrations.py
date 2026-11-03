"""Convert verified training placement snippets into worded demonstrations."""
from typing import Any, TypedDict

import numpy as np

from carl.icl.boards import from_rows, to_rows, tiles
from carl.icl.datasets import environment
from carl.icl.types import BoardRows, TrainingData
from scripts.verbal_icl.components import describe_transition


class VerbalDemo(TypedDict):
    """One observed training action; no future witness is needed at inference."""

    source: BoardRows
    state: BoardRows
    after: BoardRows
    goal: str
    action: int
    reached: bool


def placement_goal(source: BoardRows, target: BoardRows) -> str:
    """Name only newly occupied goal squares in a verified training snippet."""
    before, after = tiles(from_rows(source)), tiles(from_rows(target))
    cells = np.argwhere((after == 3) & (before != 3))
    if not len(cells):
        raise ValueError('Placement demonstration must occupy a new goal')
    destinations = ', '.join(f'[{int(row)},{int(col)}]' for row, col in cells)
    return f'Place a box on each of these goal squares: {destinations}.'


def prepare(rows: list[dict[str, Any]]) -> tuple[TrainingData, list[VerbalDemo]]:
    """Use all training snippets for examples and three multi-step sources for calibration."""
    contexts = []
    demos: list[VerbalDemo] = []
    for row in rows:
        source = from_rows(row['state'])
        env, state = environment(source), source.copy()
        goal = placement_goal(row['state'], row['subgoal'])
        states = [state.copy()]
        for index, action in enumerate(row['actions']):
            after = np.asarray(env.next_state(state, action))
            if index in {0, len(row['actions']) - 1}:
                demos.append({'source': row['state'], 'state': to_rows(state), 'after': to_rows(after),
                              'goal': goal, 'action': action, 'reached': index == len(row['actions']) - 1})
            states.append(after.copy())
            state = after
        if to_rows(state) != row['subgoal']:
            raise ValueError('Training placement witness does not replay')
        if len(row['actions']) > 1 and len(contexts) < 3:
            candidates = []
            for k in (len(row['actions']) - 1, len(row['actions'])):
                candidates.append({'subgoal': to_rows(states[k]), 'k': k, 'actions': row['actions'][:k]})
            contexts.append({'id': row['id'], 'state': row['state'], 'remaining_actions': len(row['actions']),
                             'candidates': candidates, 'calibration_indices': [0, 1]})
    training: TrainingData = {'contexts': contexts, 'provenance': {'verbal_demonstrations': rows}}
    return training, demos


def training_goal(source: BoardRows, target: BoardRows) -> str:
    """Prefer placement language when the training endpoint occupies a new goal."""
    try:
        return placement_goal(source, target)
    except ValueError:
        return describe_transition(source, target)

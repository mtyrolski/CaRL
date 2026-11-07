"""Prompted textual generator, controller and judge using the existing API client."""
import json
from collections.abc import Callable
from typing import Any, Final, TypedDict, TYPE_CHECKING
from pathlib import Path
from dataclasses import asdict

import numpy as np

from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows, tiles, to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from carl.icl.luna_components.formatting import board_coordinates, task
from carl.icl.responses import request_options, text_message
from carl.icl.types import Board, BoardRows
from scripts.verbal_icl.types import Judgment
from scripts.verbal_icl.artifacts import append_jsonl

if TYPE_CHECKING:
    from scripts.verbal_icl.demonstrations import VerbalDemo

ACTION_NAMES: Final = ('UP', 'DOWN', 'LEFT', 'RIGHT')
RULES: Final = (
    'Sokoban uses # wall, - floor, . goal, $ box, * box on goal, @ player and + player on goal. '
    'A move changes the player position one cell UP, DOWN, LEFT or RIGHT. A player can push one box '
    'into a free cell, but cannot pull a box or push two boxes. All boxes on goals solves the task. '
    'Coordinates are zero-based [row,column], with rows increasing DOWN and columns increasing RIGHT.'
)


class Example(TypedDict):
    """A semantic goal with a percentile measured by this same controller and judge."""

    context: str
    target_index: int
    source: BoardRows
    goal: str
    u: float


def describe_transition(source: BoardRows, target: BoardRows) -> str:
    """Build a truthful short goal from a verified training transition, not from a test trajectory."""
    before, after = tiles(from_rows(source)), tiles(from_rows(target))
    before_boxes = {tuple(int(v) for v in rc) for rc in np.argwhere(np.isin(before, [3, 4]))}
    after_boxes = {tuple(int(v) for v in rc) for rc in np.argwhere(np.isin(after, [3, 4]))}
    new_boxes = sorted(after_boxes - before_boxes)
    if new_boxes:
        destinations = ', '.join(f'[{row},{col}]' for row, col in new_boxes)
        return f'Place boxes on these cells: {destinations}.'
    before_player = np.argwhere(np.isin(before, [5, 6]))
    after_player = np.argwhere(np.isin(after, [5, 6]))
    if len(before_player) != 1 or len(after_player) != 1 or np.array_equal(before_player, after_player):
        raise ValueError('Training target does not change the player or box positions')
    row, col = (int(v) for v in after_player[0])
    return f'Move the player to cell [{row},{col}].'


class VerbalComponents:
    """Do not infer a target board or query a simulator to validate a model judgment."""

    def __init__(self, config: ExperimentConfig, api: ExperimentAPI, reference: ICLComponents,
                 examples: list[Example], prompts: dict[str, str], audit_path: Path | None = None,
                 demonstrations: list["VerbalDemo"] | None = None, generator_examples: int | None = None) -> None:
        self.config, self.api, self.reference = config, api, reference
        self.examples, self.prompts = examples, prompts
        self.audit_path = audit_path
        self.demonstrations = demonstrations or []
        self.generator_examples = generator_examples
        if generator_examples is not None and generator_examples < 1:
            raise ValueError("generator_examples must be positive")
        self.calls: dict[str, int] = {'generator': 0, 'policy': 0, 'judge': 0, 'value': 0}

    def _call(self, role: str, prompt: str, examples: list[tuple[dict[str, Any], dict[str, Any]]],
              question: dict[str, Any], mock: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [text_message('developer', f'{RULES}\n{prompt}')]
        for q, answer in examples:
            messages.extend((text_message('user', json.dumps(q)), text_message('assistant', json.dumps(answer))))
        messages.append(text_message('user', json.dumps(question)))
        reasoning = self.config.policy_reasoning_effort if role == 'policy' else self.config.reasoning_effort
        limit = {'generator': self.config.generator_tokens, 'policy': self.config.policy_tokens,
                 'judge': self.config.value_tokens}[role]
        payload: dict[str, Any] = {
            **request_options(reasoning or 'none', {'type': 'text'}), 'model': self.config.model,
            'input': messages, 'max_output_tokens': limit, 'metadata': {'component': role},
        }
        for attempt in range(self.config.json_max_retries + 1):
            self.calls[role] += 1
            def offline() -> dict[str, Any]:
                return {'output_text': json.dumps(mock()), 'model': 'mock', 'usage': {}}
            if self.audit_path is not None:
                append_jsonl(self.audit_path, {'role': role, 'status': 'requested', 'payload': payload})
            response = (self.api.call('responses', payload, mock=offline) if attempt == 0 else
                        self.api.call_json_retry('responses', payload, mock=offline))
            if self.audit_path is not None:
                append_jsonl(self.audit_path, {'role': role, 'payload': payload, 'response': response})
            try:
                value = json.loads(response['output_text'])
                if not isinstance(value, dict):
                    raise ValueError('Expected an object')
                return value
            except (TypeError, ValueError):
                if attempt == self.config.json_max_retries:
                    raise RequestFailed(f'Invalid JSON in verbal {role} response') from None
        raise AssertionError('Unreachable retry loop')

    def _policy_examples(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if self.demonstrations:
            return [({'source': row['source'], 'state': row['state'], 'subgoal': row['goal']},
                     {'action': ACTION_NAMES[row['action']]}) for row in self.demonstrations]
        pairs = []
        for question, answer in self.reference.policy_examples():
            goal = describe_transition(question['state'], question['subgoal'])
            probabilities = answer['probabilities']
            pairs.append(({'source': question['state'], 'state': question['state'], 'subgoal': goal},
                          {'action': ACTION_NAMES[int(np.argmax(probabilities))]}))
        return pairs

    def propose(self, state: Board, scale: float) -> list[str]:
        if not self.examples:
            raise ValueError('Semantic calibration must be completed before generating goals')
        examples = [({'state': row['source'], 'u': row['u']}, {'subgoals': [row['goal']]})
                    for row in self.examples[:self.generator_examples or self.config.few_shot]]
        response = self._call('generator', self.prompts['generator'], examples,
                              {'state': to_rows(state), 'goal': task(state), 'u': scale,
                               'num_subgoals': self.config.num_subgoals},
                              lambda: {'subgoals': ['Move the player to cell [1,1].']})
        raw = response.get('subgoals')
        if not isinstance(raw, list) or len(raw) > self.config.num_subgoals:
            raise RequestFailed('Verbal generator must return a bounded subgoals list')
        goals: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip() or len(item) > 240:
                raise RequestFailed('Each verbal subgoal must be a nonempty string of at most 240 characters')
            if item.strip() not in goals:
                goals.append(item.strip())
        return goals

    def action(self, state: Board, goal: str, source: Board | None = None) -> int:
        rows = to_rows(state)
        response = self._call('policy', self.prompts['policy'], self._policy_examples(),
                              {'source': to_rows(source if source is not None else state), 'state': rows,
                               'state_coordinates': board_coordinates(rows), 'subgoal': goal},
                              lambda: {'action': 'DOWN'})
        action = response.get('action')
        if not isinstance(action, str) or action not in ACTION_NAMES:
            raise RequestFailed('Verbal policy must return one action from UP/DOWN/LEFT/RIGHT')
        return ACTION_NAMES.index(action)

    def judge(self, state: Board, goal: str, source: Board | None = None) -> Judgment:
        examples = []
        for question, _answer in ([] if self.demonstrations else self.reference.policy_examples()):
            description = describe_transition(question['state'], question['subgoal'])
            for current, reached in ((question['state'], False), (question['subgoal'], True)):
                examples.append(({'source': question['state'], 'state': current, 'subgoal': description},
                                 asdict(Judgment(reached, float(reached),
                                        'The specified position is occupied.' if reached else
                                        'The specified destination has not been reached.'))))
        if self.demonstrations:
            examples = []
            for row in self.demonstrations:
                examples.append(({'source': row['source'], 'state': row['after'], 'subgoal': row['goal']},
                                 asdict(Judgment(row['reached'], float(row['reached']),
                                                'A box occupies the specified goal square.' if row['reached'] else
                                                'The specified goal square is not yet occupied by a box.'))))
        rows = to_rows(state)
        response = self._call('judge', self.prompts['judge'], examples,
                              {'source': to_rows(source if source is not None else state), 'state': rows,
                               'state_coordinates': board_coordinates(rows), 'subgoal': goal},
                              lambda: {'reached': False, 'progress': 0.0, 'reason': 'Requested condition is not met.'})
        reached, progress = response.get('reached'), response.get('progress', 0.0)
        reason = response.get('reason')
        if (not isinstance(reason, str) or not reason.strip() or not isinstance(reached, bool) or isinstance(progress, bool) or not isinstance(progress, (int, float))
                or not np.isfinite(progress) or not 0 <= progress <= 1):
            raise RequestFailed('Judge must return boolean reached, nonempty reason, and optional progress in [0,1]')
        return Judgment(reached, float(progress), reason.strip())

    def value(self, state: Board) -> float:
        self.calls['value'] += 1
        return self.reference.value(state)

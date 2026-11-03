"""The post-hoc retry changes generator inputs without changing its calibrated controller."""
import json
from typing import Any, cast

from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents, board_coordinates
from scripts.verbal_icl.components import Example, VerbalComponents
from scripts.verbal_icl.demonstrations import VerbalDemo
from scripts.verbal_icl.grounded_probe import GroundedComponents


class RecordingAPI:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def call(self, _endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        del mock
        self.payloads.append(payload)
        role = payload['metadata']['component']
        answer = {'generator': {'subgoals': ['Place a box on goal [2,3].']},
                  'policy': {'action': 'RIGHT'}, 'judge': {'reached': False, 'reason': 'Goal unoccupied.'}}[role]
        return {'output_text': json.dumps(answer)}


def test_generator_grounding_preserves_policy_and_judge_payloads() -> None:
    state = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    rows = to_rows(state)
    config = ExperimentConfig(few_shot=1, num_subgoals=1, json_max_retries=1)
    examples: list[Example] = [{'context': 'train', 'target_index': 0, 'source': rows,
                                'goal': 'Place a box on goal [2,3].', 'u': 0.5}]
    demos: list[VerbalDemo] = [{'source': rows, 'state': rows, 'after': rows, 'goal': examples[0]['goal'],
                               'action': 3, 'reached': False}]
    prompts = {'generator': 'Ground all coordinates', 'policy': 'Frozen policy', 'judge': 'Frozen judge'}
    baseline_api, grounded_api = RecordingAPI(), RecordingAPI()
    baseline = VerbalComponents(config, cast(ExperimentAPI, baseline_api), cast(LunaComponents, None),
                                examples, prompts, demonstrations=demos)
    grounded = GroundedComponents(config, cast(ExperimentAPI, grounded_api), cast(LunaComponents, None),
                                  examples, prompts, demonstrations=demos)
    for component in (baseline, grounded):
        component.propose(state, 0.5)
        component.action(state, examples[0]['goal'], state)
        component.judge(state, examples[0]['goal'], state)
    query = json.loads(grounded_api.payloads[0]['input'][-1]['content'][0]['text'])
    assert query['state_coordinates'] == board_coordinates(rows)
    assert baseline_api.payloads[1:] == grounded_api.payloads[1:]
    assert grounded.config is config and grounded.config.json_max_retries == 1
    assert len(grounded_api.payloads) == 3

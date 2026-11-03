"""Primitive controls preserve campaign parity and never query the generator."""
import json
from pathlib import Path
from typing import cast

import pytest

from carl.icl.config import ExperimentConfig, Limits
from carl.icl.types import Instance, TrainingData
from scripts.campaign_primitive_baseline import DESIGN, prepare_or_check, primitive_config, run_one
from scripts.tune_campaign_scales import configured, read_design


def test_primitive_config_matches_frozen_validation_resources() -> None:
    design = read_design(DESIGN)
    for family in design['families']:
        reference = configured(design, 'relative_u', 0, family, budget=128, instances=1)
        primitive = primitive_config(design, family)
        assert primitive.method == 'primitive' and primitive.split == 'scale_validation'
        assert primitive.action_budget == 128 and primitive.repeat_seeds == [11] and primitive.instances == 1
        assert primitive.limits == reference.limits and primitive.max_expansions == reference.max_expansions
        assert primitive.num_subgoals == reference.num_subgoals
        assert primitive.prompts == reference.prompts and primitive.value_target == reference.value_target
        assert primitive.value_examples == reference.value_examples and primitive.locked_policy_fingerprint == reference.locked_policy_fingerprint


def test_manifest_is_precommitted_and_refuses_mismatch(tmp_path: Path) -> None:
    first = {'provider': 'openai', 'cases': [{'state_hash': 'abc'}]}
    with pytest.raises(ValueError, match='--prepare'):
        prepare_or_check(tmp_path, first, prepare=False)
    prepare_or_check(tmp_path, first, prepare=True)
    assert json.loads((tmp_path / 'manifest.json').read_text()) == first
    prepare_or_check(tmp_path, first, prepare=False)
    with pytest.raises(ValueError, match='manifest changed'):
        prepare_or_check(tmp_path, {'provider': 'mock'}, prepare=False)


def test_mock_primitive_executes_and_replays_without_generator(tmp_path: Path) -> None:
    design = read_design(DESIGN)
    config = ExperimentConfig(method='primitive', few_shot=0, value_examples='initial', action_budget=4,
                              cache_dir=str(tmp_path / 'cache'), limits=Limits(max_requests=20))
    training = cast(TrainingData, {'contexts': []})
    item = cast(Instance, {'id': 'tiny', 'state': ['#####', '#---#', '#@$.#', '#---#', '#####'], 'level_key': 'x'})
    case = run_one(config, design, 'mock', training, item, 'primitive-test')
    assert case['verification']['verified'] and case['verification']['solution_replayed']
    assert case['outcome']['solved'] and case['outcome']['attempted_actions'] == 4
    assert case['api']['generator_calls'] == 0 and case['api']['value_calls'] > 0
    assert case['budget_tag'] == 'primitive-test'
    assert sum(len(node['actions']) for node in case['tree']['nodes']) == case['outcome']['attempted_actions']

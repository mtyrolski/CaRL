"""Portable measured verbal components for the shared 100-board launch executor."""
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import Board, TrainingData
from scripts.verbal_icl.components import Example
from scripts.verbal_icl.demonstrations import VerbalDemo
from scripts.verbal_icl.grounded_probe import GroundedComponents
from scripts.verbal_icl.reference import build_reference
from scripts.verbal_icl.search import search
from scripts.verbal_icl.types import Result


@dataclass(frozen=True)
class FrozenVerbal:
    """Train-only controller bundle with measured, model-judged scale labels."""

    config: ExperimentConfig
    prompts: dict[str, str]
    examples: list[Example]
    demonstrations: list[VerbalDemo]
    generator_examples: int
    reference_config: ExperimentConfig
    reference_training: TrainingData
    contract: dict[str, Any]

    def build(self, api: ExperimentAPI, audit_path: Path | None = None) -> GroundedComponents:
        """Keep component prompts/config fixed; caller owns API budgets and closure."""
        # Only reference.value is used; loading an unrelated explicit generator is unnecessary.
        value_config = self.reference_config.model_copy(update={'generator_checkpoint': None})
        reference = build_reference(value_config, self.reference_training, api)
        return GroundedComponents(self.config, api, reference, self.examples, self.prompts,
                                  audit_path, self.demonstrations, self.generator_examples)


def load_bundle(path: Path) -> FrozenVerbal:
    """Load exported training assets without accessing held-out solution witnesses."""
    payload = json.loads(path.read_text())
    if payload.get('schema') != 'carl-verbal-launch-v1':
        raise ValueError('Expected carl-verbal-launch-v1 bundle')
    if payload['calibration_status'] != 'completed':
        raise ValueError('Measured verbal calibration must be complete')
    config = ExperimentConfig.model_validate(payload['config'])
    reference_config = ExperimentConfig.model_validate(payload['reference']['config'])
    if config.model != reference_config.model or config.model != 'gpt-6-luna':
        raise ValueError('Verbal launch must preserve the measured Luna model')
    examples = cast(list[Example], payload['examples'])
    if len({row['context'] for row in examples}) < config.few_shot:
        raise ValueError('Measured calibration lacks the required distinct contexts')
    if any(not 0 <= row['u'] <= 1 for row in examples):
        raise ValueError('Observed percentiles must be in [0,1]')
    prompts = payload['prompts']
    if set(prompts) != {'generator', 'policy', 'judge'}:
        raise ValueError('Expected exactly generator/policy/judge prompts')
    modules = ['launch.py', 'components.py', 'grounded_probe.py', 'reference.py', 'search.py', 'types.py']
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in modules}
    contract = {'bundle_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'provenance': payload['provenance'], 'code_sha256': code_hashes,
                'scales': config.scales, 'action_cap': config.calibration.action_cap, 'seed': config.seed,
                'randomness': 'No local RNG; single-action model outputs and deterministic queue tie ordering',
                'api_and_usd_limits': 'Owned by the shared launch executor and its injected ExperimentAPI/ledger',
                'reference_usage': 'Original V and fingerprint; unused explicit generator checkpoint is not loaded',
                'task_success': 'environment.is_solved', 'subgoal_stop': 'independent Luna judge',
                'node_budget_semantics': 'popped frontier entries expanded, including scale/fallback entries',
                'limitations': 'Tiny observed calibration; u uses judge claims, not semantic ground truth. '
                               'Grounded generator wording was selected after observing validation failures. '
                               'This is a new 100-board launch, not a measured success claim.'}
    return FrozenVerbal(config, prompts, examples, cast(list[VerbalDemo], payload['demonstrations']),
                        payload['generator_examples'], reference_config,
                        cast(TrainingData, payload['reference']['training']), contract)


def run_case(bundle: FrozenVerbal, api: ExperimentAPI, state: Board, *, action_budget: int = 4000,
             node_budget: int = 500, trace: Callable[[dict[str, object]], None] | None = None,
             audit_path: Path | None = None) -> Result:
    """Execute one board with independent finite node/action/API/USD limits."""
    if min(action_budget, node_budget) < 1:
        raise ValueError('Node and action budgets must be positive')
    component = bundle.build(api, audit_path)
    return search(environment(state), component, state,
                  (bundle.config.scales[0], bundle.config.scales[1]), action_budget,
                  bundle.config.calibration.action_cap, node_budget, trace)

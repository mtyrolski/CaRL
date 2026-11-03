"""Post-hoc generator wording retry on exactly two previously observed failures."""
import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import board_coordinates
from scripts.verbal_icl.artifacts import append_jsonl, write_json
from scripts.verbal_icl.components import Example, VerbalComponents
from scripts.verbal_icl.demonstrations import VerbalDemo
from scripts.verbal_icl.reference import build_reference
from scripts.verbal_icl.runner import client, mark_interrupted_unknown
from scripts.verbal_icl.search import RolloutFailure, rollout


class GroundedComponents(VerbalComponents):
    """Change generator input only; controller and judge are inherited unchanged."""

    def _call(self, role: str, prompt: str, examples: list[tuple[dict[str, Any], dict[str, Any]]],
              question: dict[str, Any], mock: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if role != 'generator':
            return super()._call(role, prompt, examples, question, mock)
        enriched = {**question, 'state_coordinates': board_coordinates(question['state'])}
        original = self.config
        # Exactly one generator request, including malformed-JSON outcomes.
        self.config = original.model_copy(update={'json_max_retries': 0})
        try:
            return super()._call(role, prompt, examples, enriched, mock)
        finally:
            self.config = original


class PreviewAPI:
    """Capture exact generator requests without obtaining or claiming predictions."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def call(self, _endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        del mock
        self.payloads.append(payload)
        return {'output_text': '{"subgoals":[]}'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'), nargs='?', default='plan')
    parser.add_argument('--provider', choices=('openai', 'replay', 'mock'), default='openai')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/icl/verbal/grounded_generator.json')
    args = parser.parse_args()
    if args.command == 'run' and args.provider == 'openai' and not args.execute:
        parser.error('Paid calls require --execute')
    spec = json.loads(args.config.read_text())
    source = ROOT / spec['source_run']
    frozen = json.loads((source / 'checkpoint.json').read_text())
    calibration_path = source / 'calibration' / frozen['signature'] / 'calibration.json'
    calibration = json.loads(calibration_path.read_text())
    if calibration['signature'] != frozen['signature'] or calibration['status'] != 'completed':
        raise ValueError('A completed source verbal calibration is required')
    if spec['query_indices'] != [1, 3] or spec['action_cap'] != 8:
        raise ValueError('This retry is restricted to observed failures 01/03 and L=8')
    config = ExperimentConfig.model_validate(frozen['config'])
    if config.num_subgoals != 1:
        raise ValueError('Exactly one generated subgoal per case is required')
    prompts = {**frozen['prompts'], 'generator': spec['generator_prompt']}
    reference_contract = frozen['training']['provenance']['reference']
    reference_config = ExperimentConfig.model_validate(reference_contract['config'])
    examples = cast(list[Example], calibration['examples'])
    demos = cast(list[VerbalDemo], frozen['demonstrations'])
    preview_api = PreviewAPI()
    reference_api = ExperimentAPI(config.path('cache_dir'), 'mock', config.limits)
    try:
        reference = build_reference(reference_config, reference_contract['training'], reference_api)
        component = GroundedComponents(config, cast(ExperimentAPI, preview_api), reference,
                                       examples, prompts, demonstrations=demos,
                                       generator_examples=frozen['generator_examples'])
        queries = []
        for index in spec['query_indices']:
            previous = json.loads((source / f'query_{index:02d}.json').read_text())
            if any(row['judged_reached'] or row['solved'] for row in previous['attempts']):
                raise ValueError('Retry source must be an observed failure')
            queries.append({'index': index, 'query': previous['query'], 'scale': previous['scale']})
            component.propose(from_rows(previous['query']['state']), previous['scale'])
    finally:
        if reference_api.client is not None:
            reference_api.client.close()
    inputs = [Path(__file__), args.config, source / 'checkpoint.json', calibration_path]
    inputs.extend(Path(__file__).with_name(name) for name in
                  ('components.py', 'search.py', 'types.py', 'artifacts.py', 'reference.py'))
    contract = {'provider': args.provider, 'spec': spec, 'source_calibration': frozen['signature'],
                'queries': queries, 'generator_payloads': preview_api.payloads,
                'controller_prompts': {key: prompts[key] for key in ('policy', 'judge')},
                'source_files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in inputs},
                'interpretation': 'Two post-hoc retries on already observed failures; not an independent retest. '
                                  'Only generator wording and current-board coordinates changed. '
                                  'Policy/judge calibration reused unchanged; no semantic oracle or output repair.'}
    label = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output = ROOT / spec['output_dir'] / label
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        checkpoint = output / 'checkpoint.json'
        if checkpoint.exists() and json.loads(checkpoint.read_text()) != contract:
            raise ValueError('Frozen retry contract changed')
        write_json(checkpoint, contract)
        if args.command == 'plan':
            print(json.dumps({'checkpoint': str(checkpoint), 'queries': [1, 3], 'action_cap': 8}, indent=2))
            return
        credentials(args.provider)
        api = client(config, args.provider, calibration=False, stage='reserve')
        try:
            api.budget_tag = f'verbal_grounded_retry_{label}'
            reference = build_reference(reference_config, reference_contract['training'], api)
            component = GroundedComponents(config, api, reference, examples, prompts,
                                           output / 'raw_calls.jsonl', demos, frozen['generator_examples'])
            for case in queries:
                index, query, scale = case['index'], case['query'], case['scale']
                path = output / f'query_{index:02d}.json'
                started = path.with_suffix('.started.json')
                if path.exists() or started.exists():
                    raise ValueError(f'Existing retry requires inspection; never repeat automatically: {path}')
                write_json(started, case)
                state = from_rows(query['state'])
                trace = output / f'query_{index:02d}.trace.jsonl'
                def record(before: Any, action: int, after: Any, solved: bool) -> None:
                    append_jsonl(trace, {'before': to_rows(before), 'action': action,
                                         'after': to_rows(after), 'solved': solved})
                goals = component.propose(state, scale)
                attempts = []
                for goal in goals:
                    error_text = None
                    try:
                        attempt = rollout(environment(state), component, state, goal, 8, record)
                    except RolloutFailure as error:
                        attempt, error_text = error.attempt, str(error.cause)
                    result = asdict(attempt)
                    result.pop('endpoint')
                    attempts.append({'goal': goal, **result, 'error': error_text})
                write_json(path, {**case, 'attempts': attempts, 'api': api.metrics,
                                  'interpretation': contract['interpretation']})
                started.unlink()
                if any(row['error'] for row in attempts):
                    raise ValueError(f'API error preserved: {path}')
        except KeyboardInterrupt:
            mark_interrupted_unknown(api)
            raise
        finally:
            if api.client is not None:
                api.client.close()


if __name__ == '__main__':
    main()

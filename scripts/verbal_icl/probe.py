"""Bounded placement-goal calibration and 2–6 public query rollouts."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from scripts.verbal_icl.artifacts import write_json, append_jsonl
from carl.icl.boards import from_rows
from carl.icl.config import ROOT
from carl.icl.datasets import environment
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from scripts.verbal_icl.calibration import calibrate, signature
from scripts.verbal_icl.components import Example, VerbalComponents
from scripts.verbal_icl.demonstrations import prepare
from scripts.verbal_icl.runner import client, configured, mark_interrupted_unknown
from scripts.verbal_icl.search import RolloutFailure, rollout
from scripts.verbal_icl.preview import controller_payloads
from scripts.verbal_icl.reference import reference_data, build_reference

DATA = ROOT / 'analysis/luna_icl/campaign_100/goal_progress_probe'


def main() -> None:
    """Freeze all public inputs before any paid call; preserve interrupted cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'calibrate', 'run'))
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='mock')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cases', type=int, choices=range(2, 7), default=4)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/icl/verbal/tiny.json')
    args = parser.parse_args()
    if args.provider == 'openai' and args.command != 'plan' and not args.execute:
        parser.error('Paid calls require --execute')
    config, prompts, spec = configured(args.config)
    rows = json.loads((DATA / 'data_training_demonstrations.json').read_text())['demonstrations']
    training, demos = prepare(rows)
    reference_config, reference_training = reference_data(spec, config.model)
    training.setdefault('provenance', {})['reference'] = {
        'config': reference_config.model_dump(mode='json'), 'training': reference_training}
    preview_api = ExperimentAPI(config.path('cache_dir'), 'mock', config.limits)
    try:
        reference_preview = build_reference(reference_config, reference_training, preview_api)
        reference_fingerprint = reference_preview.fingerprint()
        reference_value_payload = reference_preview.value_payload(from_rows(training['contexts'][0]['state']))
    finally:
        if preview_api.client is not None:
            preview_api.client.close()
    queries = json.loads((DATA / 'data_validation_queries.json').read_text())['instances'][:args.cases]
    generator_examples = spec.get('generator_examples')
    label = signature(config, training, prompts, args.provider, generator_examples)
    output = ROOT / spec['output_dir'] / label
    frozen = {'signature': label, 'provider': args.provider, 'config': config.model_dump(mode='json'),
              'prompts': prompts, 'generator_examples': generator_examples,
              'reference_policy_fingerprint': reference_fingerprint,
              'reference_value_payload_preview': reference_value_payload, 'controller_prompt_preview': controller_payloads(config, prompts, demos),
              'training': training, 'demonstrations': demos, 'queries': queries,
              'warning': 'New model-judged calibration only. Historical exact-state labels are invalid. '
                         'Judge mistakes remain visible, without a hidden semantic oracle.'}
    frozen['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text()) != frozen:
        raise ValueError('Frozen inputs differ; choose a fresh output directory')
    write_json(checkpoint, frozen)
    if args.command == 'plan':
        print(json.dumps({'checkpoint': str(checkpoint), 'calibration_contexts': len(training['contexts']),
                          'candidates': sum(len(row['candidates']) for row in training['contexts']),
                          'queries': len(queries), 'action_cap': config.calibration.action_cap}, indent=2))
        return
    credentials(args.provider)
    api = client(config, args.provider, calibration=args.command == 'calibrate', stage=spec['ledger_stage'])
    try:
        api.budget_tag = f'verbal_placement_{args.command}_{label}'
        reference = build_reference(reference_config, reference_training, api)
        component = VerbalComponents(config, api, reference, [], prompts, output / 'raw_calls.jsonl', demos, generator_examples)
        with FileLock(str(output / '.lock'), timeout=0):
            try:
                if args.command == 'calibrate':
                    result = calibrate(config, training, component, prompts, output / 'calibration')
                    print(json.dumps({'status': result['status'], 'path': str(output)}, indent=2))
                    return
                calibration = json.loads((output / 'calibration' / label / 'calibration.json').read_text())
                if calibration['status'] != 'completed' or calibration['signature'] != label:
                    raise ValueError('Matching completed verbal calibration is required')
                component.examples = cast(list[Example], calibration['examples'])
                for index, query in enumerate(queries):
                    path = output / f'query_{index:02d}.json'
                    started = path.with_suffix('.started.json')
                    if path.exists():
                        saved = json.loads(path.read_text())
                        if any(row['error'] for row in saved['attempts']):
                            raise ValueError(f'Inspect failed case before continuing: {path}')
                        continue
                    if started.exists():
                        raise ValueError(f'Inspect interrupted case before retrying: {started}')
                    state, scale = from_rows(query['state']), config.scales[index % 2]
                    write_json(started, {'query': query, 'scale': scale})
                    trace = output / f'query_{index:02d}.trace.jsonl'
                    def record(before: Any, action: int, after: Any, solved: bool) -> None:
                        from carl.icl.boards import to_rows
                        append_jsonl(trace, {'before': to_rows(before), 'action': action,
                                             'after': to_rows(after), 'solved': solved})
                    goals = component.propose(state, scale)
                    attempts = []
                    for goal in goals:
                        error_text = None
                        try:
                            attempt = rollout(environment(state), component, state, goal,
                                              min(config.action_budget, config.calibration.action_cap), record)
                        except RolloutFailure as error:
                            attempt, error_text = error.attempt, str(error.cause)
                        result = asdict(attempt)
                        result.pop('endpoint')
                        attempts.append({'goal': goal, **result, 'error': error_text})
                    write_json(path, {'query': query, 'scale': scale, 'attempts': attempts, 'api': api.metrics})
                    started.unlink()
                    if any(row['error'] for row in attempts):
                        raise ValueError(f'API failure preserved at {path}')
            except KeyboardInterrupt:
                mark_interrupted_unknown(api)
                raise
    finally:
        if api.client is not None:
            api.client.close()


if __name__ == '__main__':
    main()

"""Explicit entrypoint for the experimental verbal-goal calibration and search."""
import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from scripts.verbal_icl.calibration import calibrate, signature
from scripts.verbal_icl.components import Example, VerbalComponents
from scripts.verbal_icl.search import search, verify_trace

DEFAULT_CONFIG = ROOT / 'configs/icl/verbal/smoke.json'
CALIBRATION_ROOT = ROOT / 'analysis/luna_icl/verbal_relative/calibration'


def merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Merge nested configuration mappings without mutating either input."""
    result = dict(base)
    for key, value in other.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def read_spec(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Resolve relative JSON inheritance and reject cycles."""
    path = path.resolve()
    if path in seen:
        raise ValueError('Cyclic verbal experiment config')
    data = json.loads(path.read_text())
    parent = data.pop('extends', None)
    return merge(read_spec(path.parent / parent, seen | {path}), data) if parent else data


def configured(path: Path) -> tuple[ExperimentConfig, dict[str, str], dict[str, Any]]:
    """Use the existing model/limit settings with a separate explicit experiment spec."""
    spec = read_spec(path)
    base = load_config(spec['base_config'])
    config = ExperimentConfig.model_validate(merge(base.model_dump(), spec['overrides']))
    prompts = spec['prompts']
    if set(prompts) != {'generator', 'policy', 'judge'} or not all(isinstance(p, str) and p for p in prompts.values()):
        raise ValueError('Provide nonempty generator, policy and judge instructions')
    if spec['ledger_stage'] != 'reserve':
        raise ValueError('This exploratory experiment uses the shared reserve stage')
    return config, prompts, spec


def client(config: ExperimentConfig, provider: str, *, calibration: bool, stage: str) -> ExperimentAPI:
    """Separate clients still share the durable campaign ledger."""
    api = ExperimentAPI(config.path('cache_dir'), provider,
                        config.calibration.limits if calibration else config.limits)
    api.stage = stage
    api.budget_tag = 'verbal_relative_calibration' if calibration else 'verbal_relative_search'
    return api


def mark_interrupted_unknown(api: ExperimentAPI) -> None:
    """Keep unknown charges reserved after interruption of this client's request."""
    if api.ledger is None:
        return
    with sqlite3.connect(api.ledger.path) as db:
        rows = db.execute("SELECT id FROM attempts WHERE stage=? AND tag=? AND status='reserved'",
                          (api.stage, api.budget_tag)).fetchall()
    for (identifier,) in rows:
        api.ledger.mark_unknown(identifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'calibrate', 'run'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='mock')
    parser.add_argument('--execute', action='store_true', help='Explicitly allow paid OpenAI calls')
    parser.add_argument('--resume', action='store_true', help='Reuse matching completed search checkpoints')
    args = parser.parse_args()
    if args.provider == 'openai' and args.command != 'plan' and not args.execute:
        parser.error('OpenAI calls require --execute')
    if args.execute and args.provider != 'openai':
        parser.error('--execute requires --provider openai')
    config, prompts, spec = configured(args.config)
    training, instances = load_data(config)
    label = signature(config, training, prompts, args.provider)
    calibration_file = CALIBRATION_ROOT / label / 'calibration.json'
    if args.command == 'plan':
        saved_calibration = json.loads(calibration_file.read_text()) if calibration_file.exists() else {}
        print(json.dumps({'provider': args.provider, 'model': config.model, 'family': config.family,
                          'split': config.split, 'boards': len(instances), 'action_budget': config.action_budget,
                          'action_cap': config.calibration.action_cap, 'semantic_calibration': str(calibration_file),
                          'calibration_ready': saved_calibration.get('status') == 'completed',
                          'calibration_status': saved_calibration.get('status', 'missing'), 'ledger_stage': spec['ledger_stage'],
                          'warning': 'Model-judged verbal hits are not verified semantic truth; 50 boards is a separate future experiment.'}, indent=2))
        return
    credentials(args.provider)
    os.environ['CARL_CAMPAIGN_STAGE'] = spec['ledger_stage']
    initial = from_rows(training['contexts'][0]['state'])
    if args.command == 'calibrate':
        api = client(config, args.provider, calibration=True, stage=spec['ledger_stage'])
        api.budget_tag = f'verbal_relative_calibration_{label}'
        reference = LunaComponents(config, api, environment(initial), training)
        CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(CALIBRATION_ROOT / f'{label}.lock'), timeout=0):
                result = calibrate(config, training, VerbalComponents(config, api, reference, [], prompts),
                                   prompts, CALIBRATION_ROOT)
        except KeyboardInterrupt:
            mark_interrupted_unknown(api)
            raise
        print(json.dumps({'status': result['status'], 'contexts_with_hits': result['contexts_with_hits'],
                          'trials': result['trials'], 'path': str(calibration_file), 'metrics': api.metrics}, indent=2))
        return
    if not calibration_file.is_file():
        raise ValueError(f'Run semantic calibration first: {calibration_file}')
    calibration = json.loads(calibration_file.read_text())
    if calibration.get('status') != 'completed' or calibration.get('signature') != label:
        raise ValueError('Semantic calibration is incomplete or belongs to different components')
    output = ROOT / spec['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        for instance in instances:
            state = from_rows(instance['state'])
            contract = {'board_id': instance['id'], 'state_hash': state_hash(state), 'seed': config.seed,
                        'calibration': label, 'config': config.model_dump(mode='json'), 'provider': args.provider}
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            path = output / f'{digest}.json'
            started, trace = output / f'{digest}.started.json', output / f'{digest}.trace.jsonl'
            if path.exists():
                saved = json.loads(path.read_text())
                if not args.resume or saved['contract'] != contract:
                    raise ValueError(f'Existing search checkpoint requires a matching --resume: {path}')
                if saved['result']['finish_reason'] in {'api_error', 'api_budget'}:
                    raise ValueError(f'Inspect failed checkpoint before retrying: {path}')
                saved_events = [json.loads(line) for line in trace.read_text().splitlines()]
                verify_trace(environment(state), state, saved_events, saved['result']['attempted_actions'])
                continue
            if started.exists() or trace.exists():
                raise ValueError(f'Unfinished run requires inspection before retry: {started}')
            api = client(config, args.provider, calibration=False, stage=spec['ledger_stage'])
            api.budget_tag = f'verbal_relative_search_{digest}'
            env = environment(state)
            reference = LunaComponents(config, api, env, training)
            component = VerbalComponents(config, api, reference, cast(list[Example], calibration['examples']), prompts)
            write_json(started, {'status': 'started', 'contract': contract})
            trace.touch(exist_ok=False)
            def save_event(event: dict[str, object]) -> None:
                with trace.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event) + '\n')
            try:
                result = search(env, component, state, (config.scales[0], config.scales[1]), config.action_budget,
                                config.calibration.action_cap, config.max_expansions, save_event)
            except KeyboardInterrupt:
                mark_interrupted_unknown(api)
                raise
            trace_events = [json.loads(line) for line in trace.read_text().splitlines()]
            verify_trace(env, state, trace_events, result.attempted_actions)
            if result.solved:
                current = state.copy()
                for action in result.action_path:
                    if env.is_solved(current):
                        raise ValueError('Reported solution includes post-terminal actions')
                    current = env.next_state(current, action)
                if not env.is_solved(current):
                    raise ValueError('Reported solution does not replay')
            write_json(path, {'contract': contract, 'result': asdict(result), 'api': api.metrics,
                              'component_calls': component.calls})
            started.unlink()
            print(json.dumps({'board_id': instance['id'], 'solved': result.solved, 'finish': result.finish_reason,
                              'actions': result.attempted_actions, 'path': str(path)}), flush=True)
            if result.finish_reason in {'api_error', 'api_budget'}:
                raise ValueError('Search stopped; inspect preserved checkpoint before retrying')


if __name__ == '__main__':
    main()

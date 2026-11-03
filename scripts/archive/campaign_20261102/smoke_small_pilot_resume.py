"""Exercise the actual solver and pilot checkpoints on two validation boards, offline."""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import components_for_run
from carl.icl.luna import LunaComponents
from carl.icl.search import search
from scripts.archive.campaign_20261102.campaign_pilot_runner import TrackingAPI, completion_class, run_jobs, schedule, verify_checkpoint
from scripts.archive.campaign_20261102.campaign_scale_stress import DurableEnvironment, DurableTree, verify_actions
from scripts.archive.campaign_20261102.tune_campaign_scales import failure_category, verify_tree

DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/small_boards_v1/resume_smoke'


class OfflineEmptyGenerator(LunaComponents):
    """Make the documented primitive fallback explicit without extra simulator probes."""

    def propose(self, state: Any, scale: float) -> list[Any]:
        del state, scale
        return []


def execute(job: dict[str, Any], trace_path: Path, config: ExperimentConfig, training: Any, item: Any) -> dict[str, Any]:
    """Run the real search and CaRL environment with an explicitly offline mock provider."""
    resolved = ExperimentConfig.model_validate({**config.model_dump(), 'action_budget': job['budget']})
    reference = components_for_run(resolved, 'replay', training, from_rows(item['state']))
    calibrated = copy.deepcopy(reference.calibrated)
    if calibrated is None:
        raise ValueError('The validation contract is missing its training calibration')
    calibrated['provider'] = 'mock'
    api = TrackingAPI(resolved.path('cache_dir'), 'mock', resolved.limits, max_generator_calls=64,
                      max_total_tokens=500000)
    tree = DurableTree(trace_path)
    env = DurableEnvironment(environment(from_rows(item['state'])), tree)
    component = OfflineEmptyGenerator(resolved, api, env, training, calibrated=calibrated)
    result = search(env, component, from_rows(item['state']), resolved, job['seed'], observer=tree)
    verification = verify_tree(tree, from_rows(item['state']), job['budget'])
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    verify_actions(tree, from_rows(item['state']), events)
    if result.finish_reason == 'api_error' or not verification['verified']:
        raise ValueError('Offline validation solver failed or did not replay')
    return {'job': job, 'status': 'complete', 'completion_class': completion_class(result, job['budget'], failed=False),
            'failure_category': failure_category(result), 'budget_scope': None, 'outcome': asdict(result),
            'unexpected_exception_type': None, 'verification': verification, 'api': dict(api.metrics),
            'diagnostics': dict(component.diagnostics), 'config': resolved.model_dump(mode='json')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty directory for the offline validation resume test')
    output.mkdir(parents=True, exist_ok=True)
    configs = {}
    for method, path in (('relative_u', 'relative_u_retry.json'), ('absolute_d', 'absolute_d_retry.json')):
        base = load_config(ROOT / 'configs/icl/campaign_data' / path)
        configs[method] = ExperimentConfig.model_validate({**base.model_dump(), 'data_dir': 'data/icl/campaign_dataset_v2',
            'family': 'real_12x12x4', 'split': 'scale_validation', 'instances': 2, 'repeat_seeds': [11],
            'action_budget': 8, 'max_expansions': 16, 'cache_dir': str(output / 'mock_cache')})
    training, items = load_data(configs['relative_u'])
    training_2, items_2 = load_data(configs['absolute_d'])
    if training != training_2 or items != items_2 or len(items) != 2:
        raise ValueError('The two methods did not load the same two full validation boards')
    by_id = {item['id']: item for item in items}
    simulated = {'seed': 11, 'main_budget': 8, 'budgets': [8], 'boards': [{'id': row['id']} for row in items]}
    jobs = schedule(simulated, 'offline-validation-shard-not-a-pilot-freeze')
    calls: list[str] = []

    def runner(job: dict[str, Any], trace: Path) -> dict[str, Any]:
        calls.append(job['key'])
        return execute(job, trace, configs[job['method']], training, by_id[job['board_id']])

    def validator(job: dict[str, Any], record: dict[str, Any], trace: Path) -> None:
        verify_checkpoint(job, record, trace, configs[job['method']], by_id[job['board_id']])

    first = run_jobs(jobs[:1], output, runner, resume=False)
    first_path = output / 'trials' / f"{jobs[0]['key']}.json"
    before = first_path.read_bytes()
    if first['completed'] != 1 or calls != [jobs[0]['key']]:
        raise ValueError('First offline checkpoint was not completed exactly once')
    result = run_jobs(jobs, output, runner, resume=True, validator=validator, workers=args.workers)
    if (result['expected'] != 4 or result['completed'] != 4 or result['failed'] or result['missing']
            or calls.count(jobs[0]['key']) != 1 or first_path.read_bytes() != before):
        raise ValueError('Offline resume repeated a completed case or failed to finish the shard')
    summary = {'provider': 'mock', 'split': 'scale_validation', 'family': 'real_12x12x4',
               'boards': 2, 'budgets': [8], 'methods': list(configs), 'first_checkpoint_preserved': True,
               'resume_workers': args.workers,
               'first_checkpoint_reexecuted': False, 'physical_provider_requests': 0,
               'note': 'Real search and CaRL transitions; mocked value and explicit empty generator. Not a Luna result.',
               'progress': result}
    (output / 'resume_smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Create a pilot freeze only after complete small-board validation and stress audits."""
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.budget_sweep import CODE_FILES
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import load_data
from carl.icl.experiment import calibration_path, components_for_run
from carl.icl.search import SearchResult
from scripts.archive.campaign_20261102.campaign_pilot_runner import EXTRA_CODE, HTTP_RETRY_POLICY, METHODS, digest, relative_path, validate_freeze
from scripts.archive.campaign_20261102.campaign_real12_stress import checked_sources, load_trace, prepare_plan
from scripts.archive.campaign_20261102.campaign_scale_stress import verify_actions
from scripts.archive.campaign_20261102.estimate_campaign_pilot import ledger_snapshot
from scripts.archive.campaign_20261102.estimate_real12_pilot import project
from scripts.archive.campaign_20261102.tune_campaign_scales import verify_tree

DEFAULT_TUNING = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/solver'
DEFAULT_STRESS = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/stress_256'
DEFAULT_DATA = ROOT / 'data/icl/campaign_dataset_12x12_v1'
DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/freeze.json'


def audited_stress(tuning: Path, stress: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently check the saved stress contract, outcome and every recorded action."""
    plan, config, item, _training = prepare_plan(tuning, stress)
    manifest = json.loads((stress / 'manifest.json').read_text())
    case = json.loads((stress / 'case.json').read_text())
    if manifest != plan:
        raise ValueError('Stress manifest differs from the recomputed validated plan')
    if (case['provider'] != 'openai' or case['seed'] != plan['seed'] or case['instance'] != item
            or case['config'] != config.model_dump(mode='json') or case['budget_tag'] != plan['ledger_tag']
            or case['policy_fingerprint'] != plan['policy_fingerprint']):
        raise ValueError('Stress output does not match its immutable contract')
    outcome = SearchResult(**case['outcome'])
    tree, events = load_trace(stress / 'trace.jsonl', outcome)
    finished = [event for event in events if event['event'] == 'finish']
    if len(finished) != 1 or finished[0]['outcome'] != case['outcome']:
        raise ValueError('Stress trace is incomplete or differs from the result')
    verified = verify_tree(tree, from_rows(item['state']), plan['action_budget'])
    verify_actions(tree, from_rows(item['state']), events)
    solved_early = outcome.finish_reason == 'solved' and outcome.solved
    full_budget = (case.get('evidence_full_budget') and outcome.attempted_actions == plan['action_budget']
                   and outcome.finish_reason in {'action_budget', 'solved'})
    if verified != case['verification'] or not verified['verified'] or not (solved_early or full_budget):
        raise ValueError('Stress must replay successfully and either solve or exercise the main budget')
    return plan, case


def build_freeze(tuning: Path, stress: Path, data: Path) -> dict[str, Any]:
    """Freeze only measured settings and explicitly preserve cost uncertainty."""
    manifest, summary, cases = checked_sources(tuning)
    plan, stress_case = audited_stress(tuning, stress)
    design = manifest['design']
    budgets, main = design['pilot_budget_candidates'], design['candidate_main_budget']
    if (len(budgets) != 3 or budgets != sorted(set(budgets)) or main not in budgets
            or main != plan['action_budget'] or max(budgets) % main):
        raise ValueError('Pilot budget candidates do not match the audited stress design')
    factor = max(budgets) // main
    stress_config = plan['resolved_config']
    limits = dict(stress_config['limits'])
    for field in ('max_requests', 'max_total_input_tokens', 'max_output_tokens', 'wall_seconds'):
        limits[field] *= factor
    configs: dict[str, ExperimentConfig] = {}
    conditions: dict[str, Any] = {}
    for method in METHODS:
        scales = summary['methods'][method]['selected_pair']
        matches = [row for row in manifest['resolved_conditions'] if row['method'] == method
                   and row['family'] == 'real_12x12x4' and row['scales'] == scales]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify the selected validated condition for {method}')
        config = ExperimentConfig.model_validate({**matches[0], 'name': f'pilot_small_{method}',
            'data_dir': relative_path(data), 'split': 'pilot100', 'instances': 100,
            'action_budget': main, 'max_expansions': stress_config['max_expansions'] * factor,
            'limits': limits})
        configs[method] = config
    training, items = load_data(configs[METHODS[0]])
    if len(items) != 100:
        raise ValueError('Small pilot cohort is not complete')
    for method in METHODS:
        other_training, other = load_data(configs[method])
        if other != items or other_training != training:
            raise ValueError('Frozen methods do not use the same cohort and training data')
        component = components_for_run(configs[method], 'replay', training, from_rows(items[0]['state']))
        if (component.fingerprint() != manifest['policy_fingerprint'] or component.calibrated is None
                or component.calibrated['signature'] != manifest['calibration_signature']):
            raise ValueError('Pilot policy or calibration differs from validation')
        conditions[method] = {'config': configs[method].model_dump(mode='json'),
                              'policy_fingerprint': component.fingerprint(),
                              'calibration_signature': component.calibrated['signature']}
    integrity_path = ROOT / 'analysis/luna_icl/campaign_100/small_boards_v1/integrity.json'
    integrity = json.loads(integrity_path.read_text())
    pilot_path = data / 'real_12x12x4/pilot100.json'
    if (integrity['boards'] != 100 or integrity['unique_layouts'] != 100 or integrity['witnesses_replayed'] != 100
            or not integrity['every_witness_solved'] or any(integrity['overlap_by_existing_split'].values())
            or integrity['overlap_observed_requests_or_history'] or digest(pilot_path) != integrity['public_manifest_sha256']):
        raise ValueError('Small-board cohort integrity audit is missing or invalid')
    prices_path = ROOT / 'configs/icl/campaign_budget.json'
    pricing = json.loads(prices_path.read_text())
    ledger = ledger_snapshot(ROOT / pricing['ledger_path'], pricing)
    projection = project(cases, summary, pricing['models']['gpt-6-luna'], ledger)
    if projection['independent_budgets'] != budgets:
        raise ValueError('The cost scenario budgets differ from validation')
    sources = set(CODE_FILES) | set(EXTRA_CODE) | {'scripts/archive/campaign_20261102/freeze_small_pilot.py',
               'scripts/archive/campaign_20261102/campaign_real12_stress.py', 'scripts/archive/campaign_20261102/tune_real12_campaign_scales.py',
               'scripts/archive/campaign_20261102/estimate_real12_pilot.py', 'configs/icl/campaign_budget.json',
               relative_path(integrity_path), relative_path(tuning / 'manifest.json'),
               relative_path(tuning / 'summary.json'), relative_path(stress / 'manifest.json'),
               relative_path(stress / 'case.json'), relative_path(stress / 'trace.jsonl'),
               relative_path(data / 'campaign_dataset_manifest.json'), relative_path(data / 'training.json'),
               relative_path(data / 'value_examples.json'), relative_path(pilot_path)}
    sources.update(relative_path(path) for path in tuning.glob('tune_*.json'))
    for config in configs.values():
        component = components_for_run(config, 'replay', training, from_rows(items[0]['state']))
        sources.add(relative_path(calibration_path(config, component)))
    boards = [{'id': row['id'], 'state_hash': state_hash(from_rows(row['state'])),
               'layout_id': canonical_layout_id(from_rows(row['state']))} for row in items]
    return {'version': 1, 'status': 'frozen', 'purpose': 'small_board_pilot', 'family': 'real_12x12x4',
            'split': 'pilot100', 'seed': manifest['seed'], 'budgets': budgets, 'main_budget': main,
            'boards': boards, 'conditions': conditions,
            'audit': {'tuning_dir': relative_path(tuning), 'stress_dir': relative_path(stress),
                      'data_dir': relative_path(data), 'integrity_report': relative_path(integrity_path)},
            'max_generator_calls': plan['max_generator_calls'] * factor,
            'max_total_tokens': plan['max_total_tokens'] * factor,
            'http_retry_policy': HTTP_RETRY_POLICY,
            'limit_scaling_note': 'Scale audited stress limits by max(budgets)/main_budget equally for every run.',
            'validation_solved_counts': {method: summary['methods'][method]['pairs'] for method in METHODS},
            'stress_result': {'outcome': stress_case['outcome']['finish_reason'],
                              'actions': stress_case['outcome']['attempted_actions'], 'replay_verified': True,
                              'evidence_full_budget': bool(stress_case.get('evidence_full_budget'))},
            'cost_projection': projection, 'ledger_at_freeze': ledger,
            'private_witness_sha256': integrity['private_witness_sha256'],
            'source_sha256': {name: digest(ROOT / name) for name in sorted(sources)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-dir', type=Path, default=DEFAULT_TUNING)
    parser.add_argument('--stress-dir', type=Path, default=DEFAULT_STRESS)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.name != 'freeze.json':
        raise ValueError('The finalized artifact must be named freeze.json')
    if args.output.exists():
        raise ValueError('Freeze already exists and must not be overwritten')
    freeze = build_freeze(args.tuning_dir.resolve(), args.stress_dir.resolve(), args.data_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.freeze-audit-') as directory:
        candidate = Path(directory) / 'freeze.json'
        write_json(candidate, freeze)
        validate_freeze(candidate)
        # A hard link publishes only a completely validated file and fails if
        # another process created the final immutable freeze concurrently.
        os.link(candidate, args.output)
    print(f'Created audited freeze: {args.output}; pilot remains unlaunched.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'Freeze audit not satisfied: {error}') from None

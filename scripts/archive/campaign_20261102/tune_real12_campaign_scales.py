"""Run matched scale tuning only on full 12x12x4 campaign validation boards.

The run is deliberately separate from the earlier two-family experiment.
Preparation and replay of existing responses do not make provider calls.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT
from carl.icl.datasets import load_data
from carl.icl.experiment import credentials
from carl.icl.search import SearchResult
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from scripts.archive.campaign_20261102.tune_campaign_scales import (
    _sha,
    build_manifest,
    configured,
    family_limit_summary,
    run_case,
    summarize,
    verify_tree,
)

DESIGN = ROOT / 'configs/icl/campaign_data/scale_tuning/design_real12_v1.json'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1'
PROTOCOL = OUTPUT / 'protocol.md'
FAMILY = 'real_12x12x4'
METHODS = {'relative_u', 'absolute_d', 'fixed'}


def read_design(path: Path) -> dict[str, Any]:
    """Require the exact board family, balanced grid and positive resource caps."""
    design = json.loads(path.read_text(encoding='utf-8'))
    if design['version'] != 1 or set(design['config_by_method']) != METHODS:
        raise ValueError('Expected all three matched methods')
    if design['families'] != [FAMILY]:
        raise ValueError('Only full real 12x12x4 validation boards are permitted')
    if set(design['scale_pairs']) != METHODS or any(len(pairs) != 2 for pairs in design['scale_pairs'].values()):
        raise ValueError('Each method must have two scale pairs')
    for high, low in design['scale_pairs']['relative_u']:
        if not 0 < low < high < 1:
            raise ValueError('Relative levels must satisfy 0 < low < high < 1')
    for method in ('absolute_d', 'fixed'):
        if any(not 0 < low < high for high, low in design['scale_pairs'][method]):
            raise ValueError('Absolute and fixed levels must be positive and descending')
    if not 1 <= design['instances_per_family'] <= 4 or design['budget'] < 1:
        raise ValueError('Invalid number of validation boards or action budget')
    if not 0 < design['early_limit_block_fraction'] <= 1:
        raise ValueError('Invalid early resource-stop threshold')
    if any(design[key] < 1 for key in ('max_generator_calls', 'max_expansions', 'max_total_tokens')):
        raise ValueError('Resource limits must be positive')
    return design


def manifest_for(design_path: Path, design: dict[str, Any], provider: str, workers: int) -> dict[str, Any]:
    """Freeze the original harness dependencies and this 12x12 protocol."""
    manifest = build_manifest(design_path, design, budget=design['budget'],
                              instances=design['instances_per_family'], provider=provider, workers=workers)
    manifest['hashes'].update({str(Path(__file__).resolve()): _sha(Path(__file__)), str(PROTOCOL): _sha(PROTOCOL)})
    manifest['purpose'] = 'exploratory real_12x12x4-only matched scale selection; no held-out data'
    return manifest


def read_case(path: Path, config: Any, item: Any, index: int, provider: str, seed: int, tag: str) -> dict[str, Any]:
    """Validate a completed case and replay all recorded actions on resume."""
    case = json.loads(path.read_text(encoding='utf-8'))
    if (case['config'] != config.model_dump(mode='json') or case['instance'] != item
            or case['provider'] != provider or case['seed'] != seed or case['pair_index'] != index
            or case.get('budget_tag') != tag):
        raise ValueError(f'Saved case differs from the frozen contract: {path}')
    tree = SearchTreeRecorder()
    tree.nodes = [TreeNode(**node) for node in case['tree']['nodes']]
    tree.result = SearchResult(**case['outcome'])
    if verify_tree(tree, from_rows(item['state']), config.action_budget) != case['verification']:
        raise ValueError(f'Saved trace does not replay: {path}')
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=DESIGN)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'solver')
    parser.add_argument('--provider', choices=('openai', 'replay'), default='replay')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Inspect the manifest without model calls')
    modes.add_argument('--prepare', action='store_true', help='Write the immutable manifest; no model calls')
    args = parser.parse_args()
    design_path = args.design.resolve()
    design = read_design(design_path)
    credentials('replay' if args.plan or args.prepare else args.provider)
    # One worker limits simultaneous ledger reservations; all conditions use the same limits.
    manifest = manifest_for(design_path, design, args.provider, workers=1)
    if args.plan:
        print(json.dumps(manifest, indent=2))
        return
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Manifest changed; use a new output directory')
        if not manifest_path.exists():
            if any(path.name != '.lock' for path in output.iterdir()):
                raise ValueError('Nonempty output lacks a manifest')
            write_json(manifest_path, manifest)
        if args.prepare:
            print(f'Prepared: {manifest_path}', flush=True)
            return
        cases: list[dict[str, Any]] = []
        methods = summarize(design, cases, design['instances_per_family'])
        for board_index in range(design['instances_per_family']):
            for index in range(2):
                for method in design['config_by_method']:
                    config = configured(design, method, index, FAMILY, budget=design['budget'],
                                        instances=design['instances_per_family'])
                    training, items = load_data(config)
                    item = items[board_index]
                    name = f"{config.name}_{item['id']}_seed{design['seed']}"
                    tag = f"{design['tag_prefix']}_{method}_{index}_{item['id']}_seed{design['seed']}"
                    case_path = output / f'{name}.json'
                    if case_path.exists():
                        case = read_case(case_path, config, item, index, args.provider, design['seed'], tag)
                    else:
                        case = run_case(config, args.provider, item, training, design['max_generator_calls'],
                                        design['max_total_tokens'], design['seed'], tag, 1)
                        case['pair_index'] = index
                        write_json(case_path, case)
                    cases.append(case)
                    methods = summarize(design, cases, design['instances_per_family'])
                    write_json(output / 'summary.json', {'status': 'running', 'tie_rule': design['tie_rule'],
                                                          'methods': methods})
                    outcome = case['outcome']
                    print(f"{method}[{index}] {item['id']}: {outcome['finish_reason']} "
                          f"actions={outcome['attempted_actions']}", flush=True)
                    if (outcome['finish_reason'] == 'api_error' or not case['verification']['verified']
                            or case.get('failure_category') == 'campaign_stage_guard'):
                        write_json(output / 'summary.json', {'status': 'blocked_error', 'tie_rule': design['tie_rule'],
                                                              'methods': methods})
                        raise SystemExit(2)
        families = family_limit_summary(cases, [FAMILY], design['early_limit_block_fraction'])
        blocked = families[FAMILY]['blocked'] or any(value['selected_pair'] is None for value in methods.values())
        write_json(output / 'summary.json', {'status': 'blocked_limits_or_incomplete' if blocked else 'complete',
                                              'tie_rule': design['tie_rule'], 'family_limits': families,
                                              'methods': methods})
        if blocked:
            raise SystemExit(2)


if __name__ == '__main__':
    main()

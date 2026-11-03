"""Inspect and archive a failed validation case without retrying or mutating it."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from filelock import FileLock

from carl.icl.boards import from_rows
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import load_data
from carl.icl.search import SearchResult
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from scripts.archive.campaign_20261102.resume_calibration_v2 import invalid_policy_output
from scripts.archive.campaign_20261102.tune_campaign_scales import build_manifest, read_design, verify_tree

DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/solver'
DEFAULT_ARCHIVE = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/recovery'
DEFAULT_DESIGN = ROOT / 'configs/icl/campaign_data/scale_tuning/design.json'


def sha(raw: bytes) -> str:
    """Hash the exact bytes to detect a changed case or model response."""
    return hashlib.sha256(raw).hexdigest()


def ledger_snapshot(tag: str) -> dict[str, Any]:
    """Read all case attempts and events, including unsettled reservations."""
    budget = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    ledger_path = ROOT / budget['ledger_path']
    with sqlite3.connect(ledger_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        attempts = [dict(row) for row in db.execute(
            'SELECT * FROM attempts WHERE stage=? AND tag=? ORDER BY rowid', ('scale_validation', tag))]
        events = [dict(row) for row in db.execute(
            'SELECT * FROM events WHERE stage=? AND tag=? ORDER BY id', ('scale_validation', tag))]
    return {'attempts': attempts, 'events': events}


def inspect_case(case_path: Path, output: Path, design_path: Path) -> dict[str, Any]:
    """Validate the frozen contract, failed trace and most recent invalid cache entry."""
    if case_path.parent.resolve() != output.resolve():
        raise ValueError('The case must be in the specified solver directory')
    manifest = json.loads((output / 'manifest.json').read_text())
    design = read_design(design_path)
    rebuilt = build_manifest(design_path, design, budget=manifest['budget'],
                             instances=manifest['instances_per_family'], provider=manifest['provider'], workers=1)
    if manifest != rebuilt or manifest['provider'] != 'openai':
        raise ValueError('Original manifest, code or data have changed')
    raw_case = case_path.read_bytes()
    case = json.loads(raw_case)
    if (case['outcome']['finish_reason'] != 'api_error'
            or case['outcome']['error'] != 'Invalid JSON from model component'
            or case['failure_category'] != 'model_parse_or_validation_error'):
        raise ValueError('This workflow only handles a malformed policy JSON response')
    config = ExperimentConfig.model_validate(case['config'])
    training, items = load_data(config)
    matching = [item for item in items if item == case['instance']]
    if len(matching) != 1 or case['seed'] != manifest['seed'] or case['provider'] != 'openai':
        raise ValueError('Case identity does not match the validation configuration')
    if case['config'] not in manifest['resolved_conditions'] or case['frozen_policy'] != manifest['policy_fingerprint']:
        raise ValueError('Case policy or resolved condition does not match the manifest')
    if case['calibration_signature'] != manifest['calibration_signature']:
        raise ValueError('Calibration signature changed')
    tree = SearchTreeRecorder()
    tree.nodes = [TreeNode(**node) for node in case['tree']['nodes']]
    tree.result = SearchResult(**case['outcome'])
    if verify_tree(tree, from_rows(case['instance']['state']), config.action_budget) != case['verification']:
        raise ValueError('Failed attempt does not replay')
    snapshot = ledger_snapshot(case['budget_tag'])
    logical = [event for event in snapshot['events'] if event['kind'] == 'logical_request']
    if not logical:
        raise ValueError('No ledger request for the failed case')
    digest = logical[-1]['request_hash']
    if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
        raise ValueError('Malformed cache hash')
    cache_path = config.path('cache_dir') / 'openai' / f'{digest}.json'
    raw_cache = cache_path.read_bytes()
    cache = json.loads(raw_cache)
    request_digest = sha(json.dumps(cache['request'], sort_keys=True).encode())
    if cache['request_hash'] != digest or request_digest != digest or not invalid_policy_output(cache):
        raise ValueError('Last case request is not the invalid completed policy response')
    return {'case_path': str(case_path), 'case_hash': sha(raw_case), 'cache_path': str(cache_path),
            'cache_hash': sha(raw_cache), 'request_hash': digest, 'tag': case['budget_tag'],
            'failed_actions': case['outcome']['attempted_actions'], 'ledger': snapshot,
            'manifest_hash': sha((output / 'manifest.json').read_bytes()), 'case': case,
            'training': training, 'instance': matching[0], 'config': config}


def save_immutable(path: Path, raw: bytes) -> None:
    """Never replace archived evidence, including if preparation is repeated."""
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Archive differs: {path}')
        return
    path.write_bytes(raw)
    if path.read_bytes() != raw:
        raise OSError(f'Archive did not persist: {path}')


def prepare(info: dict[str, Any], directory: Path) -> None:
    """Archive failure and ledger before touching the active cache."""
    directory.mkdir(parents=True, exist_ok=True)
    save_immutable(directory / 'failed_case.json', Path(info['case_path']).read_bytes())
    save_immutable(directory / 'invalid_response.json', Path(info['cache_path']).read_bytes())
    metadata = {k: v for k, v in info.items() if k not in {'case', 'training', 'instance', 'config'}}
    save_immutable(directory / 'archive_metadata.json', json.dumps(metadata, indent=2, sort_keys=True).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--design', type=Path, default=DEFAULT_DESIGN)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    case_path, output, design_path = args.case.resolve(), args.output.resolve(), args.design.resolve()
    if args.check:
        info = inspect_case(case_path, output, design_path)
        print(json.dumps({k: info[k] for k in ('request_hash', 'tag', 'failed_actions', 'case_hash', 'cache_hash')}, indent=2))
        return
    # The original tuning process owns this lock. Never overlap with it.
    with FileLock(str(output / '.lock'), timeout=0):
        info = inspect_case(case_path, output, design_path)
        directory = args.archive.resolve() / info['tag']
        prepare(info, directory)
        print(f'Archived failure without touching active cache: {directory}')


if __name__ == '__main__':
    main()

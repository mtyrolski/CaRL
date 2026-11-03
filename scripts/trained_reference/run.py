"""Plan or explicitly execute 100 local trained-model shared-search runs."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version
from typing import Any, cast, Protocol
from collections import Counter

from carl.icl.config import ExperimentConfig
from carl.icl.types import Board, BoardEnvironment, SearchComponents

from filelock import FileLock

from scripts.trained_reference.common import ROOT, digest
from scripts.verbal_icl.artifacts import append_jsonl, write_json

DEFAULT = ROOT / 'configs/trained_reference/launch100.json'


class TrainedRuntime(SearchComponents, Protocol):
    """The local trained adapter or a scripted technical-test substitute."""

    @property
    def env(self) -> BoardEnvironment: ...

    diagnostics: Counter[str]


def prepare(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read public boards and hash local checkpoint bytes, without loading model tensors."""
    config = json.loads(path.read_text())
    if (config['instances'] != 100 or config['generator_scales'] != [8, 4]
            or config['max_expansions'] != 500 or config['action_budget'] != 4000
            or config['num_subgoals'] != 1 or config['num_beams'] != 4):
        raise ValueError('Launch contract requires 100 boards, scales8/4, nodes500/actions4000, one proposal/beam4')
    cohort_path = ROOT / config['cohort']
    instances = json.loads(cohort_path.read_text())['instances']
    if len(instances) != 100 or len({row['id'] for row in instances}) != 100:
        raise ValueError('Expected 100 distinct public board IDs')
    public_fields = {'id', 'board_id', 'state', 'family', 'split', 'trajectory_id', 'source_path',
                     'source_sha256', 'source_index', 'generator_version', 'seed', 'state_hash', 'layout_id', 'level_key'}
    if any(set(row) - public_fields for row in instances):
        raise ValueError('Cohort contains fields beyond public board inputs')
    instances = [{key: row[key] for key in ('id', 'state', 'level_key')} for row in instances]
    for row in instances:
        if len(row['state']) != 12 or any(len(line) != 12 for line in row['state']):
            raise ValueError('Expected 12x12 boards')
    root = Path(config['checkpoint_root']).expanduser().resolve()
    roles = {f'generator/{scale}': config['generator_paths'][str(scale)] for scale in (8, 4)}
    roles.update(cllp=config['controller_path'], value=config['value_path'])
    checkpoint_hashes = {}
    for role, relative in roles.items():
        directory = root / relative
        checkpoint_hashes[role] = {}
        for filename in ('config.json', config['weights_file']):
            file = directory / filename
            if not file.is_file():
                raise FileNotFoundError(f'Local checkpoint file required, no network fallback: {file}')
            checkpoint_hashes[role][filename] = digest(file)
        generation_config = directory / 'generation_config.json'
        if generation_config.is_file():
            checkpoint_hashes[role]['generation_config.json'] = digest(generation_config)
    sources = [Path(__file__), Path(__file__).with_name('launch_verification.py'),
               ROOT / 'scripts/probe_trained_sokoban.py', ROOT / 'carl/icl/search.py',
               ROOT / 'carl/icl/planning/search_types.py', ROOT / 'carl/icl/search_tree.py',
               ROOT / 'carl/environment/sokoban/tokenizer.py', ROOT / 'carl/environment/sokoban/core.py',
               ROOT / 'carl/environment/sokoban/env.py']
    contract = {'name': 'trained_fixed_k_shared_search', 'config': config,
                'cohort_sha256': digest(cohort_path), 'checkpoint_sha256': checkpoint_hashes,
                'code_sha256': {str(file.relative_to(ROOT)): digest(file) for file in sources},
                'node_budget_semantics': 'Non-stale frontier entries expanded; distinct from attempted primitive actions',
                'interpretation': 'New execution of local trained checkpoints through shared search with primitive fallback. '
                                  'Not the original native k=8,4,1 AdaSubS loop and not the historical 91/100 result.',
                'library_versions': {name: version(name) for name in ('torch', 'transformers', 'numpy')},
                'network': 'No model downloads or API calls; local_files_only=True'}
    return config, instances, contract


def execute_case(components: TrainedRuntime, initial: Board, search_config: ExperimentConfig, seed: int,
                 directory: Path, contract: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    """Durably complete a case; only verified completed cases may be resumed."""
    from carl.icl.search import search, SearchTransition
    from carl.icl.search_tree import SearchTreeRecorder
    from scripts.trained_reference.launch_verification import verify_saved

    directory.mkdir(parents=True, exist_ok=True)
    completed, started = directory / 'case.json', directory / 'started.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        if not resume or saved['contract'] != contract or saved['status'] != 'completed':
            raise ValueError(f'Existing case requires matching completed --resume: {completed}')
        verify_saved(components.env, initial, saved, search_config.action_budget, search_config.max_expansions)
        return saved
    if started.exists():
        raise ValueError(f'Interrupted case requires inspection; no automatic rerun: {started}')
    write_json(started, {'status': 'started', 'contract': contract})
    class DurableRecorder(SearchTreeRecorder):
        """Record observed endpoints without additional model calls."""

        def on_transition(self, transition: SearchTransition) -> None:
            super().on_transition(transition)
            append_jsonl(directory / 'trace.jsonl', asdict(self.nodes[-1]))

    recorder = DurableRecorder()
    result = search(components.env, components, initial, search_config, seed, observer=recorder)
    saved = {'status': 'completed', 'contract': contract, 'outcome': asdict(result),
             'nodes': [asdict(node) for node in recorder.nodes],
             'expansions': [asdict(row) for row in recorder.expansions],
             'diagnostics_cumulative': dict(components.diagnostics)}
    saved['verification'] = verify_saved(components.env, initial, saved,
                                         search_config.action_budget, search_config.max_expansions)
    write_json(completed, saved)
    started.unlink()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('plan', 'status', 'run'), default='plan')
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--execute', action='store_true', help='Explicitly load local torch checkpoints and run inference')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and not args.execute:
        parser.error('Local model inference requires --execute')
    config, instances, contract = prepare(args.config)
    label = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output = ROOT / config['output_dir'] / label
    if args.command != 'run':
        print(json.dumps({'name': contract['name'], 'boards': len(instances), 'max_expansions': 500,
                          'action_budget': 4000, 'signature': label, 'output': str(output),
                          'completed': len(list(output.glob('*/case.json'))),
                          'checkpoint_sha256': contract['checkpoint_sha256'],
                          'interpretation': contract['interpretation'], 'inference_executed': False}, indent=2))
        return
    # Heavy model imports and construction happen only after explicit execution.
    import torch
    from carl.icl.boards import from_rows, state_hash
    from scripts.probe_trained_sokoban import ProbeConfig, TrainedComponents

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        manifest = output / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError('Frozen trained launch contract changed')
        write_json(manifest, contract)
        torch.set_num_threads(config['torch_threads'])
        components = TrainedComponents(cast(ProbeConfig, config))
        search_config = ExperimentConfig.model_validate({
            'method': 'fixed', 'scales': [8, 4], 'fixed_cap': 'horizon', 'num_subgoals': 1,
            'action_budget': 4000, 'max_expansions': 500, 'calibration': {'action_cap': 8}})
        for index, instance in enumerate(instances):
            state = from_rows(instance['state'])
            case_contract = {'launch_signature': label, 'board_id': instance['id'], 'state_hash': state_hash(state)}
            result = execute_case(components, state, search_config, config['search_seed'],
                                  output / f'{index:03d}', case_contract, resume=args.resume)
            print(json.dumps({'board_id': instance['id'], 'solved': result['outcome']['solved'],
                              'finish': result['outcome']['finish_reason']}), flush=True)


if __name__ == '__main__':
    main()

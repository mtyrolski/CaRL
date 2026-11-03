"""Calibrate verbal goals on training trajectories with the same controller and judge."""
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.verbal_icl.artifacts import write_json, append_jsonl
from carl.icl.boards import from_rows, to_rows
from carl.icl.calibration import target_indices, weighted_percentiles
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.types import TrainingData
from scripts.verbal_icl.components import VerbalComponents
from scripts.verbal_icl.demonstrations import training_goal
from scripts.verbal_icl.search import RolloutFailure, rollout


def signature(config: ExperimentConfig, training: TrainingData, prompts: dict[str, str], provider: str,
              generator_examples: int | None = None) -> str:
    """Bind labels to the semantic policy/judge, training data and calibration contract."""
    sources = [Path(__file__), Path(__file__).with_name('components.py'), Path(__file__).with_name('search.py'),
               Path(__file__).with_name('types.py'), Path(__file__).with_name('demonstrations.py'),
               Path(__file__).with_name('artifacts.py'), Path(__file__).with_name('reference.py')]
    sources.extend(ROOT / path for path in ('carl/icl/luna.py', 'carl/icl/boards.py', 'carl/icl/api.py',
        'carl/icl/experiment_api.py', 'carl/icl/responses.py', 'carl/environment/sokoban/core.py',
        'carl/environment/sokoban/env.py', 'carl/environment/sokoban/tokenizer.py'))
    settings = config.model_dump(mode='json', exclude={'name', 'family', 'split', 'instances', 'action_budget',
        'max_expansions', 'scales', 'num_subgoals', 'tuning_status', 'output_dir', 'cache_dir', 'data_dir', 'limits'})
    settings['calibration'].pop('limits')
    contract = {'version': 2, 'generator_examples': generator_examples,
                'example_selection': 'all successful targets, round-robin across contexts, increasing target index', 'config': settings, 'training': training, 'prompts': prompts,
                'provider': 'mock' if provider == 'mock' else 'openai',
                'sampling': 'deterministic single-action policy, local response cache; repeats are not independent LLM samples',
                'code': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def calibrate(config: ExperimentConfig, training: TrainingData, components: VerbalComponents,
              prompts: dict[str, str], directory: Path) -> dict[str, Any]:
    """Save each train rollout before the next. Errors remain visible and are never auto-retried."""
    fingerprint = signature(config, training, prompts, components.api.provider, getattr(components, 'generator_examples', None))
    directory = directory / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'signature': fingerprint, 'provider': components.api.provider,
                'contexts': config.train_boards, 'trials': config.calibration.trials,
                'cap': config.calibration.action_cap, 'seed': config.seed,
                'sampling': 'deterministic cached actions; repetitions are not independent model samples'}
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Semantic calibration manifest changed')
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    trials: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for context_index, context in enumerate(training['contexts'][:config.train_boards]):
        indices, _eligible = target_indices(config, context, context_index)
        seen: set[str] = set()
        for index in indices:
            candidate = context['candidates'][index]
            try:
                goal = training_goal(context['state'], candidate['subgoal'])
            except ValueError:
                continue
            if goal in seen:
                continue
            seen.add(goal)
            hits: list[int] = []
            for trial in range(config.calibration.trials):
                path = directory / f'context_{context_index:03d}_target_{index:03d}_trial_{trial:03d}.json'
                started = path.with_suffix('.started.json')
                trace = path.with_suffix('.trace.jsonl')
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['status'] != 'completed':
                        raise ValueError(f'Interrupted calibration trial requires inspection: {path}')
                else:
                    if started.exists():
                        raise ValueError(f'Unfinished calibration trial requires inspection: {started}')
                    if trace.exists():
                        raise ValueError(f'Unexpected existing calibration trace: {trace}')
                    write_json(started, {'status': 'started', 'context': context['id'],
                                         'target_index': index, 'trial': trial})
                    def save_action(before: Any, action: int, after: Any, solved: bool) -> None:
                        event = {'before': to_rows(before), 'action': action, 'after': to_rows(after), 'solved': solved}
                        append_jsonl(trace, event)
                    try:
                        attempt = rollout(environment(from_rows(context['state'])), components,
                                          from_rows(context['state']), goal, config.calibration.action_cap, save_action)
                        row = {'status': 'completed', 'context': context['id'], 'target_index': index,
                               'trial': trial, 'goal': goal, 'actions': attempt.actions,
                               'steps': [asdict(step) for step in attempt.steps],
                               'judged_hit': attempt.judged_reached, 'task_solved': attempt.solved,
                               'first_hit': len(attempt.actions) if attempt.judged_reached else None}
                    except RolloutFailure as error:
                        row = {'status': 'interrupted', 'context': context['id'], 'target_index': index,
                               'trial': trial, 'goal': goal, 'actions': error.attempt.actions,
                               'steps': [asdict(step) for step in error.attempt.steps], 'error': str(error.cause)}
                    write_json(path, row)
                    started.unlink()
                    if row['status'] != 'completed':
                        raise ValueError(f'Calibration trial interrupted; preserved at {path}')
                trials.append(row)
                if row['judged_hit']:
                    hits.append(row['first_hit'])
            targets.append({'context': context['id'], 'target_index': index, 'source': context['state'],
                            'goal': goal, 'k': candidate['k'], 'attempts': config.calibration.trials,
                            'hits': len(hits), 'p_hat': len(hits) / config.calibration.trials,
                            'd': sum(hits) / len(hits) if hits else None})
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        if row['hits']:
            groups[row['context']].append(row)
        else:
            row['excluded_reason'] = 'no_judged_hits'
    examples: list[dict[str, Any]] = []
    for rows in groups.values():
        weights, percentiles = weighted_percentiles([row['d'] for row in rows], [row['hits'] for row in rows])
        for row, weight, percentile in zip(rows, weights, percentiles, strict=True):
            row.update(weight=float(weight), u=float(percentile), excluded_reason=None)
    ordered_groups = [sorted(rows, key=lambda row: row['target_index']) for rows in groups.values()]
    for index in range(max((len(rows) for rows in ordered_groups), default=0)):
        examples.extend(rows[index] for rows in ordered_groups if index < len(rows))
    status = 'completed' if len(groups) >= config.few_shot else 'insufficient_coverage'
    result = {'status': status, 'signature': fingerprint, 'examples': examples, 'targets': targets,
              'trials': len(trials), 'contexts_with_hits': len(groups),
              'component_calls_this_invocation': dict(components.calls),
              'api_metrics_this_invocation': dict(components.api.metrics),
              'note': 'Hits are self-reported by the model judge, not verified semantic ground truth. '
                      'Cached deterministic actions make repeated trials dependent.'}
    write_json(directory / 'calibration.json', result)
    return result

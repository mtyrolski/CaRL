"""Atomically report stage 1 from saved campaign and assessment artifacts.

This renderer never loads a controller, executes a rollout, or infers missing
results. An unavailable offset is a data-availability event; a missing shard is
an infrastructure omission. Results are based only on the frozen assessment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

FAMILIES = ('12x12x4', '12x12x10')
OFFSETS = (1, 2, 4, 8, 16, 32)
MODELS = ('cllp8', 'cllp32', 'cllp4', 'cllp1', 'cllp16', 'policy94820')


def optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a saved JSON object without turning absence into a zero."""
    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected an object')
        return value, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, f'{path}: {type(error).__name__}: {error}'


def optional_csv(path: Path) -> tuple[list[dict[str, str]] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        with path.open(newline='', encoding='utf-8') as source:
            return list(csv.DictReader(source)), None
    except (OSError, UnicodeError, csv.Error) as error:
        return None, f'{path}: {type(error).__name__}: {error}'


def number(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def cpu_usage(runner: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return worker CPU seconds and their average logical-core use."""
    cpu = number(runner.get('elapsed_cpu_seconds'))
    resources = runner.get('resources') if isinstance(runner.get('resources'), dict) else {}
    workers = resources.get('worker_cpu_seconds')
    if cpu is None and isinstance(workers, dict) and workers:
        values = [number(value) for value in workers.values()]
        if all(value is not None for value in values):
            cpu = sum(value for value in values if value is not None)
    elapsed = number(runner.get('elapsed_seconds'))
    average = cpu / elapsed if cpu is not None and elapsed is not None and elapsed > 0 else None
    return cpu, average


def integer(value: Any) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def shown(value: Any, decimals: int = 2) -> str:
    value = number(value)
    return 'not available' if value is None else f'{value:.{decimals}f}'


def percent(value: Any) -> str:
    value = number(value)
    return 'not available' if value is None else f'{100 * value:.1f}%'


def count(value: Any) -> str:
    value = integer(value)
    return 'not available' if value is None else str(value)


def cell(row: dict[str, str]) -> str:
    """Format a pointwise rate, never calling missing observations failures."""
    point = number(row.get('p_hat'))
    low, high = number(row.get('bootstrap_low')), number(row.get('bootstrap_high'))
    if point is not None and low is not None and high is not None:
        return f'{100 * point:.1f}% [{100 * low:.1f}%, {100 * high:.1f}%]'
    if point is not None:
        return f'{100 * point:.1f}% [interval unavailable]'
    lower, upper = number(row.get('bound_low')), number(row.get('bound_high'))
    if lower is not None and upper is not None:
        return f'not estimable; missing-data bounds {100 * lower:.1f}–{100 * upper:.1f}%'
    return 'not available'


def table(header: list[str], rows: list[list[Any]]) -> str:
    def escape(item: Any) -> str:
        return str(item).replace('|', r'\|').replace('\n', ' ')
    result = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    result.extend('| ' + ' | '.join(escape(item) for item in row) + ' |' for row in rows)
    return '\n'.join(result)


def planned_rows(summary: list[dict[str, str]], family: str, models: list[str]) -> list[list[str]]:
    """One L=32 result per model and offset; absent rows stay unavailable."""
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary:
        if row.get('family') != family or row.get('source_family') != 'all' or integer(row.get('budget')) != 32:
            continue
        k = integer(row.get('k'))
        if k is None:
            continue
        key = (row.get('model_id', ''), k)
        if key in lookup:
            raise ValueError(f'duplicate summary row: {family} {key}')
        lookup[key] = row
    result = []
    for model in models:
        for k in OFFSETS:
            row = lookup.get((model, k))
            if row is None:
                result.append([model, str(k), 'not available', 'not available', 'not available', 'not available'])
                continue
            result.append([model, str(k), f"{count(row.get('hits'))} / {count(row.get('planned'))} / "
                                          f"{count(row.get('complete'))}", cell(row),
                           count(row.get('unavailable_target_slots')), count(row.get('missing'))])
    return result


def diagnostics(summary: list[dict[str, str]], family: str, models: list[str]) -> list[list[str]]:
    """Pool disjoint k cells at L=32; successful durations remain conditional."""
    result = []
    for model in models:
        rows = [row for row in summary if row.get('family') == family and row.get('source_family') == 'all'
                and row.get('model_id') == model and integer(row.get('budget')) == 32]
        if not rows:
            result.append([model] + ['not available'] * 7)
            continue
        def total(key: str) -> int | None:
            values = [integer(row.get(key)) for row in rows]
            return None if any(value is None for value in values) else sum(value for value in values if value is not None)
        successes = total('hits')
        observed = total('complete')
        duration_parts = [(integer(row.get('hits')), number(row.get('mean_success_actions'))) for row in rows]
        weighted = [(n, mean) for n, mean in duration_parts if n and mean is not None]
        duration = (sum(n * mean for n, mean in weighted) / sum(n for n, _ in weighted)
                    if weighted and all(n is not None and (n == 0 or mean is not None)
                                        for n, mean in duration_parts) else None)
        result.append([model, count(observed), count(successes), shown(duration),
                       count(total('task_solved_by_budget')), count(total('total_no_ops_full_rollout')),
                       count(total('trials_with_nontrivial_cycle_full_rollout')),
                       f"{count(total('failures_at_action_limit_full_rollout'))} / "
                       f"{count(total('failures_at_task_terminal_full_rollout'))}"])
    return result


def source_breakdown(summary: list[dict[str, str]], models: list[str]) -> list[list[str]]:
    rows = [row for row in summary if row.get('source_family') != 'all' and row.get('model_id') in models
            and integer(row.get('k')) == 16 and integer(row.get('budget')) == 32]
    return [[row.get('family', '?'), row.get('source_family', '?'), row.get('model_id', '?'),
             f"{count(row.get('hits'))} / {count(row.get('planned'))} / {count(row.get('complete'))}", cell(row),
             count(row.get('unavailable_target_slots')), count(row.get('missing'))]
            for row in sorted(rows, key=lambda x: (x.get('family', ''), x.get('source_family', ''),
                                                    models.index(x.get('model_id', ''))))]


def findings(summary: list[dict[str, str]], paired: list[dict[str, str]]) -> str | None:
    """Describe only complete CLLP-8/32 cells and their paired board intervals."""
    lookup: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in summary:
        k = integer(row.get('k'))
        if integer(row.get('budget')) != 32 or k not in (16, 32):
            continue
        key = (row.get('family', ''), row.get('source_family', ''), row.get('model_id', ''), k)
        if key in lookup:
            return None
        lookup[key] = row

    def complete_row(family: str, source: str, model: str, k: int) -> dict[str, str] | None:
        row = lookup.get((family, source, model, k))
        if row is None:
            return None
        hits, planned, completed, missing = (integer(row.get(key)) for key in ('hits', 'planned', 'complete', 'missing'))
        value = number(row.get('p_hat'))
        if (hits is None or planned is None or completed is None or missing is None or value is None
                or planned < 1 or hits < 0 or hits > planned or completed != planned or missing != 0
                or not 0 <= value <= 1):
            return None
        return row

    def paired_difference(family: str, k: int, planned: int) -> tuple[float, float, float, int] | None:
        matches = [row for row in paired if row.get('family') == family and row.get('source_family') == 'all'
                   and integer(row.get('k')) == k and integer(row.get('budget')) == 32
                   and (row.get('left'), row.get('right')) in {('cllp8', 'cllp32'), ('cllp32', 'cllp8')}]
        if len(matches) != 1:
            return None
        row = matches[0]
        if (integer(row.get('planned_pairs')) != planned or integer(row.get('paired_pairs')) != planned
                or integer(row.get('missing_pairs')) != 0):
            return None
        delta, low, high = (number(row.get(key)) for key in ('difference', 'bootstrap_low', 'bootstrap_high'))
        layouts = integer(row.get('paired_layouts'))
        if delta is None or low is None or high is None or layouts is None or layouts < 1 or low > high:
            return None
        if row['left'] == 'cllp32':
            delta, low, high = -delta, -high, -low
        return delta, low, high, layouts

    primary: list[str] = []
    for family in FAMILIES:
        for k in (16, 32):
            one = complete_row(family, 'all', 'cllp8', k)
            two = complete_row(family, 'all', 'cllp32', k)
            if one is None or two is None or one['planned'] != two['planned']:
                return None
            planned = integer(one['planned'])
            if planned is None:
                return None
            comparison = paired_difference(family, k, planned)
            if comparison is None:
                return None
            delta, low, high, layouts = comparison
            primary.append(f"- **{family}, k={k}:** CLLP-8 {one['hits']}/{one['planned']} "
                           f"({percent(one['p_hat'])} board-weighted), CLLP-32 {two['hits']}/{two['planned']} "
                           f"({percent(two['p_hat'])}); paired CLLP-8 minus CLLP-32 "
                           f"{100 * delta:+.1f} percentage points, 95% paired board bootstrap "
                           f"[{100 * low:+.1f}, {100 * high:+.1f}] pp; {layouts} layouts.")

    sources = []
    for source, label in (('proposal_gamma2_12x12x10', 'gamma2'),
                          ('synthetic_reverse_12x12x10', 'synthetic reverse-board')):
        one = complete_row('12x12x10', source, 'cllp8', 32)
        two = complete_row('12x12x10', source, 'cllp32', 32)
        if one is None or two is None or one['planned'] != two['planned']:
            return None
        sources.append(f"{label}: CLLP-8 {one['hits']}/{one['planned']} ({percent(one['p_hat'])}), "
                       f"CLLP-32 {two['hits']}/{two['planned']} ({percent(two['p_hat'])})")
    return ('\n'.join(primary) + '\n\nAt k=32 on ×10, the separate source cohorts give ' + '; '.join(sources)
            + '. These comparisons are exploratory: exact checkpoint training exposure is unknown, and k is '
              'a recorded primitive-action offset, not a minimum distance. Differences between source families '
              'do not isolate the causal effect of box count or checkpoint horizon.')


def failure_locations(rows: list[dict[str, str]]) -> list[list[str]]:
    """Aggregate final-player coordinates descriptively, never combining layouts as a heatmap."""
    reasons: Counter[tuple[str, str, str]] = Counter()
    positions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get('status') != 'complete':
            continue
        family, model, reason = row.get('family', '?'), row.get('model_id', '?'), row.get('stop_reason', '?')
        reasons[(family, model, reason)] += 1
        raw = row.get('final_player', '')
        try:
            point = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(point, list) and len(point) == 2:
            positions[(family, model)][f'({point[0]}, {point[1]})'] += 1
    keys = sorted({(family, model) for family, model, _ in reasons})
    result = []
    for family, model in keys:
        by_reason = ', '.join(f'{reason}: {n}' for (f, m, reason), n in sorted(reasons.items())
                              if (f, m) == (family, model))
        top = ', '.join(f'{point}: {n}' for point, n in positions[(family, model)].most_common(3)) or 'not available'
        result.append([family, model, by_reason, top])
    return result


def make_report(run: Path) -> str:
    """Render even when evaluation or the assessment has not finished."""
    run = run.expanduser().resolve()
    stage = run / 'stage_01'
    assessment_dir = stage / 'assessment'
    inputs = {
        'campaign': run / 'campaign_manifest.json',
        'data': stage / 'data_v2/data_manifest.json',
        'exposure': stage / 'data_v2/exposure_register.json',
        'plan': stage / 'evaluation/plan.json',
        'contract': stage / 'evaluation/contract.json',
        'runner': stage / 'evaluation/status.json',
        'assessment': assessment_dir / 'assessment.json',
        'checkpoint_provenance': stage / 'checkpoint_provenance.json',
        'benchmark': stage / 'benchmark/benchmark_summary.json',
        'memo_timing': stage / 'memo_clean_timing/status.json',
        'memo_parity': stage / 'memo_clean_timing_parity.json',
        'component_parity': stage / 'public_component_batch_parity.json',
        'supervisor': run / 'stage_01_supervision/run_manifest.json',
    }
    loaded: dict[str, dict[str, Any]] = {}
    issues = []
    for name, path in inputs.items():
        value, error = optional_json(path)
        if value is not None:
            loaded[name] = value
        if error:
            issues.append(error)
    summary, summary_error = optional_csv(assessment_dir / 'summary.csv')
    paired, paired_error = optional_csv(assessment_dir / 'paired_differences.csv')
    failures, failures_error = optional_csv(assessment_dir / 'failure_locations.csv')
    for error in (summary_error, paired_error, failures_error):
        if error:
            issues.append(error)
    campaign, data = loaded.get('campaign', {}), loaded.get('data', {})
    plan, contract, runner = loaded.get('plan', {}), loaded.get('contract', {}), loaded.get('runner', {})
    assessment = loaded.get('assessment', {})
    models = list(plan.get('model_ids') or assessment.get('models') or MODELS)
    declared = assessment.get('state')
    if summary is not None:
        keys = [(row.get('family'), row.get('source_family'), row.get('model_id'),
                 row.get('k'), row.get('budget')) for row in summary]
        if len(keys) != len(set(keys)) or any(None in key for key in keys):
            issues.append(f'{assessment_dir / "summary.csv"}: duplicated or incomplete cell identity')
            summary = None
    expected_cells = {(family, 'all', model, str(k), '32') for family in FAMILIES
                      for model in models for k in OFFSETS}
    found_cells = {(row.get('family'), row.get('source_family'), row.get('model_id'), row.get('k'), row.get('budget'))
                   for row in (summary or [])}
    narrative = findings(summary, paired) if summary is not None and paired is not None else None
    is_complete = (declared == 'complete' and runner.get('state') == 'complete'
                   and integer(assessment.get('missing_model_trials')) == 0
                   and integer(runner.get('missing_shards')) == 0
                   and expected_cells <= found_cells and narrative is not None and not issues)
    state = 'complete' if is_complete else 'partial'
    blocks = ['# Stage 1: frozen Sokoban policy transfer',
              f'**Report status: {state}.** Results are from saved artifacts; absent artifacts are not zero results.',
              f"Campaign: `{campaign.get('campaign_id', 'not available')}`. Global deadline: "
              f"`{campaign.get('clock', {}).get('deadline_utc', 'not available')}`. "
              f"Frozen evaluation split: `{plan.get('split', 'not available')}`."]
    if issues:
        blocks.append('Read/validation issues: ' + '; '.join(issues))
    if not assessment:
        blocks.append('**Assessment is not yet available.** No per-model rates, failure counts or figures are '
                      'claimed here. Run the saved assessment after evaluation stops; missing shards remain missing.')
    else:
        blocks.append(f"Assessment snapshot: `{declared}`; {count(assessment.get('completed_model_trials'))} / "
                      f"{count(assessment.get('planned_model_trials'))} model trials replayed, "
                      f"{count(assessment.get('missing_model_trials'))} missing; "
                      f"{count(len(assessment.get('invalid_shards', [])) if isinstance(assessment.get('invalid_shards'), list) else None)} "
                      'invalid shards. An earlier partial snapshot may differ from the later runner status.')
    if runner:
        blocks.append(f"Runner snapshot: `{runner.get('state', 'not available')}`; "
                      f"{count(runner.get('completed_shards'))} / {count(runner.get('planned_shards'))} shards, "
                      f"{count(runner.get('missing_shards'))} missing; stop reason: "
                      f"`{runner.get('stop_reason') or 'none recorded'}`.")

    if is_complete:
        blocks.append('## Findings\n\n'
                      'CLLP-8 versus CLLP-32 at the fixed 32-action budget:\n\n' + str(narrative))
    else:
        blocks.append('## Findings\n\nNo comparison is inferred while the final assessment, paired '
                      'differences, required source cells or completed runner are missing or incomplete.')

    blocks.append('## Data and measurement contract\n\n'
                  'Five CLLP checkpoints receive the current board and the **complete future target board**; '
                  '`policy94820` receives the current board only. This is full-state reaching, not an '
                  'intent-conditioned controller or a complete-task planning comparison. Greedy argmax uses '
                  'one deterministic attempt per pair; no-ops count towards the 32 primitive-action limit. '
                  'The first hit determines success at all smaller budgets. Full-task completion is reported separately.')
    blocks.append('Input encoding uses `carl/environment/sokoban/tokenizer.py`: 291 tokens for a 12×12 '
                  'CLLP state–target pair, and 146 tokens for the current-state-only policy. All checkpoints '
                  'run in evaluation mode with greedy argmax action selection.')
    family_layouts = plan.get('family_layouts') if isinstance(plan.get('family_layouts'), dict) else {}
    blocks.append('Frozen layout counts: ' + '; '.join(f"{family} = {len(family_layouts[family]) if isinstance(family_layouts.get(family), list) else 'not available'}"
                                                        for family in FAMILIES) + '. '
                  f"Contexts per layout: {count(plan.get('contexts_per_layout'))}; "
                  f"planned model trials: {count(plan.get('planned_trials'))}.")
    family_counts = data.get('family_counts') if isinstance(data.get('family_counts'), list) else []
    if family_counts:
        blocks.append('Prepared source inventory (the selected evaluation prefix can be smaller):\n\n' + table(
            ['Family', 'Source family', 'Split', 'Layouts'],
            [[r.get('family', '?'), r.get('source_family', '?'), r.get('split', '?'), count(r.get('boards'))]
             for r in family_counts]))
    blocks.append('The ×10 cohort contains two historically inspected source distributions '
                  '(gamma2 and synthetic reverse-board), reported separately. The ×4 source is under `rl-data/validation`; '
                  'historic exposure and old checkpoint training overlap are not proven absent. The checkpoint '
                  'training splits are unknown: CLLP-1/4/8 trainer metadata names a `cllp` run, while CLLP-16/32 '
                  'names `cllp_mcts`; the differences cannot be attributed to horizon alone. The evaluated '
                  'policy-94820 is not established as the checkpoint named best (90079). Different source '
                  'distributions and box counts are confounded; this is exploratory, not a causal box-count '
                  'or verified out-of-distribution test. Historical validation/test and every frozen evaluation '
                  'layout remain protected from training and tuning in subsequent stages.')
    blocks.append('Starts were selected before evaluation, preferring a non-trivial k=32 target where one exists '
                  'and using other starts for any remaining slots. Short trajectories were not dropped for '
                  'lacking k=32. `k` is an offset in recorded primitive actions, not a push count or shortest '
                  'distance. Existing but repeated contexts are not independent boards.\n\n'
                  f"Data manifest: [{stage.name}/data_v2/data_manifest.json](stage_01/data_v2/data_manifest.json); "
                  '[exposure register](stage_01/data_v2/exposure_register.json); '
                  '[unavailable offsets](stage_01/assessment/unavailable_offsets.csv).')

    blocks.append('## Checkpoints and input contracts')
    model_contracts = contract.get('models') if isinstance(contract.get('models'), dict) else {}
    hashes = contract.get('source_sha256') if isinstance(contract.get('source_sha256'), dict) else {}
    checkpoint_rows = []
    for model in models:
        row = model_contracts.get(model, {})
        checkpoint = row.get('checkpoint')
        config = str(Path(checkpoint) / 'config.json') if checkpoint else None
        weights = str(Path(checkpoint) / 'pytorch_model.bin') if checkpoint else None
        checkpoint_rows.append([model, checkpoint or 'not available', row.get('input', 'not available'),
                                hashes.get(config, 'not available'), hashes.get(weights, 'not available')])
    blocks.append(table(['Model', 'Checkpoint', 'Input', 'Config SHA-256', 'Weights SHA-256'], checkpoint_rows))
    blocks.append('The detailed [checkpoint provenance audit](stage_01/checkpoint_provenance.json) retains '
                  'trainer metadata and the limits of inference about the original training data.')

    if summary is not None and assessment:
        blocks.append('## Exact-state success at L=32\n\n'
                      'Each point weights original layouts equally; intervals are 95% board bootstrap. '
                      '`hits / planned / completed` counts eligible targets only. `Unavailable` counts '
                      'missing targets or contexts from the data plan; `infra missing` counts planned model '
                      'attempts without a valid completed trace. These are not failures. A partial cell has '
                      'no `p_hat`; its displayed interval is a missing-data bound, not a confidence interval. '
                      'When every sampled board is a hit or every sampled board is a miss, the naive percentile '
                      'bootstrap can collapse to [100%, 100%] or [0%, 0%] because the sample has no variation; '
                      'such an interval does not establish population certainty.')
        for family in FAMILIES:
            blocks.append(f'### {family}\n\n' + table(
                ['Model', 'k', 'Hits / planned / completed', 'Board mean [95% CI]', 'Unavailable', 'Infra missing'],
                planned_rows(summary, family, models)))
        blocks.append('## Duration, cycles and task completion\n\n'
                      'The following pools disjoint k cells at L=32. Successful duration is the mean number '
                      'of attempted primitive actions **conditioned on success**. `Solved` counts observed '
                      'full-task completions during a target-reaching attempt, not independent complete-task '
                      'planning runs. No-ops and non-trivial cycles are separate; the latter revisits a '
                      'previous full state after a changed action. Counts use completed attempts only.')
        for family in FAMILIES:
            blocks.append(f'### {family}\n\n' + table(
                ['Model', 'Completed', 'Hits', 'Mean success actions', 'Task solved', 'No-ops',
                 'Trials with cycle', 'Action-limit / terminal misses'], diagnostics(summary, family, models)))
        breakdown = source_breakdown(summary, models)
        if breakdown:
            blocks.append('## Source-family breakdown at k=16, L=32\n\n' + table(
                ['Family', 'Source', 'Model', 'Hits / planned / completed', 'Board mean [95% CI]',
                 'Unavailable', 'Infra missing'], breakdown))
        if failures is not None:
            locations = failure_locations(failures)
            blocks.append('## Failure locations\n\n'
                          'Counts are over completed non-hits. The top final-player coordinates are descriptive '
                          'row/column positions pooled over differently shaped layouts, not an aligned spatial '
                          'density map. Full start, target and reached boards and coordinates remain in the CSV.\n\n'
                          + (table(['Family', 'Model', 'Stop reasons', 'Top final coordinates (row, col)'], locations)
                             if locations else 'No completed failed attempts were present in this snapshot.'))
    elif assessment and summary is None:
        blocks.append('**Assessment summary is missing or unreadable.** No outcome table can be computed safely.')

    blocks.append('## CPU benchmark, resource use and parity')
    resources = campaign.get('resources') if isinstance(campaign.get('resources'), dict) else {}
    blocks.append(f"CPU topology reported by WSL: {count(resources.get('physical_cores_reported_by_psutil'))} physical "
                  f"and {count(resources.get('logical_affinity_count'))} allowed logical CPUs; "
                  f"RAM {shown(number(resources.get('memory_total_bytes')) / 2**30 if number(resources.get('memory_total_bytes')) is not None else None)} GiB; "
                  f"GPU observed: {resources.get('gpu_observed', 'not available')}. Stage 1 runs on CPU.")
    chosen = resources.get('chosen') if isinstance(resources.get('chosen'), dict) else {}
    blocks.append(f"Chosen frozen configuration: {count(chosen.get('workers'))} persistent workers × "
                  f"{count(chosen.get('threads_per_worker'))} intra-op threads, batch {count(chosen.get('batch_size'))}; "
                  f"memoisation: `{chosen.get('memo', 'not available')}`. The peak planned worker thread count "
                  'is bounded by the recorded CPU affinity.')
    benchmark = loaded.get('benchmark', {})
    bench_rows = benchmark.get('runs') if isinstance(benchmark.get('runs'), list) else []
    if bench_rows:
        blocks.append('Development benchmark, same CLLP-8 cohort (not final evaluation):\n\n' + table(
            ['Workers', 'Threads/worker', 'Trials/s', 'Actions/s', 'Wall s', 'CPU s', 'Peak host RAM',
             'Peak swap GiB'],
            [[count(row.get('workers')), count(row.get('threads')), shown(row.get('trials_per_second')),
              shown(row.get('actions_per_second')), shown(row.get('elapsed_seconds')), shown(row.get('cpu_seconds')),
              percent(row.get('resources', {}).get('peak_host_used_fraction')),
              shown(number(row.get('resources', {}).get('peak_swap_used_bytes')) / 2**30
                    if number(row.get('resources', {}).get('peak_swap_used_bytes')) is not None else None)]
             for row in bench_rows]))
    timing = loaded.get('memo_timing', {})
    if timing:
        blocks.append(f"Clean six-model development timing: {count(timing.get('planned_trials'))} planned trials "
                      f"in {shown(timing.get('elapsed_seconds'))} s; {shown(timing.get('new_trials_per_second'))} "
                      f"trials/s and {shown(timing.get('new_actions_per_second'))} primitive actions/s.")
    parity = loaded.get('memo_parity', {})
    if parity:
        blocks.append(f"Memoisation parity: {count(parity.get('pairs_across_models'))} trial traces compared; "
                      f"{count(parity.get('different_traces'))} different. No-op simulator steps are still counted.")
    component = loaded.get('component_parity', {})
    if component:
        blocks.append(f"Public component / batched action parity: actions `{component.get('all_actions_match')}`, "
                      f"logits close `{component.get('all_logits_close')}` on "
                      f"{count(component.get('cases'))} development cases; this is a sample check.")
    run_resource = runner.get('resources') if isinstance(runner.get('resources'), dict) else {}
    if runner:
        cpu_seconds, average_cores = cpu_usage(runner)
        blocks.append(f"Evaluation peak host RAM: {percent(run_resource.get('peak_host_used_fraction'))}; "
                      f"peak child RSS: {shown(number(run_resource.get('peak_children_rss_bytes')) / 2**30 if number(run_resource.get('peak_children_rss_bytes')) is not None else None)} GiB; "
                      f"peak swap: {shown(number(run_resource.get('peak_swap_used_bytes')) / 2**30 if number(run_resource.get('peak_swap_used_bytes')) is not None else None)} GiB; "
                      f"new actions: {count(run_resource.get('new_actions'))}, "
                      f"new trials: {count(run_resource.get('new_trials'))}, elapsed {shown(runner.get('elapsed_seconds'))} s; "
                      f"worker CPU seconds: {shown(cpu_seconds)}; average logical cores used "
                      f"(CPU seconds / elapsed wall seconds): {shown(average_cores)}. "
                      f"Final throughput, if reported by the runner: {shown(runner.get('new_trials_per_second'))} trials/s "
                      f"and {shown(runner.get('new_actions_per_second'))} actions/s.")

    blocks.append('## Files, commands and continuation\n\n'
                  '[Figure PDF](stage_01/assessment/frozen_policy_reachability.pdf), '
                  '[SVG](stage_01/assessment/frozen_policy_reachability.svg), '
                  '[summary CSV](stage_01/assessment/summary.csv), '
                  '[paired differences](stage_01/assessment/paired_differences.csv), '
                  '[failure locations](stage_01/assessment/failure_locations.csv), '
                  '[per-board records](stage_01/assessment/per_board.csv), '
                  '[individual cases](stage_01/assessment/cases.csv), '
                  '[raw evaluation shards](stage_01/evaluation/shards/), '
                  '[plan](stage_01/evaluation/plan.json), '
                  '[contract](stage_01/evaluation/contract.json), '
                  '[status](stage_01/evaluation/status.json). Files are listed for discoverability; '
                  'some may not exist yet in a partial report.')
    supervisor = loaded.get('supervisor', {})
    stages = supervisor.get('stages') if isinstance(supervisor.get('stages'), list) else []
    campaign_stages = campaign.get('stages') if isinstance(campaign.get('stages'), dict) else {}
    stage_info = campaign_stages.get('01') if isinstance(campaign_stages.get('01'), dict) else {}
    replacement = stage_info.get('finalizer_replacement')
    if isinstance(replacement, dict):
        old = next((item for item in stages if item.get('name') == replacement.get('previous_stage')), None)
        new = next((item for item in stages if item.get('name') == replacement.get('new_stage')), None)
        if old and integer(old.get('exit_code')) == 143:
            blocks.append(f"Finalizer supervision: `{old.get('name')}` exited with code 143 during its "
                          f"intentional replacement by `{replacement.get('new_stage')}` "
                          f"(replacement status: `{new.get('status', 'not available') if new else 'not available'}`). "
                          'This was not a controller or rollout failure.')
    evaluation_stage = next((item for item in stages if item.get('name') == 'frozen_policy_evaluation'), None)
    if evaluation_stage and isinstance(evaluation_stage.get('command'), list):
        command = [str(value) for value in evaluation_stage['command']]
        timeout = number(evaluation_stage.get('timeout_seconds'))
        blocks.append(f"Evaluation process supervisor status: `{evaluation_stage.get('status', 'not available')}`; "
                      f"exit code: `{evaluation_stage.get('exit_code', 'not available')}`.\n\n"
                      'Recorded evaluation command, from the supervisor manifest:\n\n```bash\n'
                      + shlex.join(command) + '\n```')
        if timeout is not None and timeout > 0:
            supervised = ['uv', 'run', '--offline', '--no-sync', 'python', '-m',
                          'scripts.intent_experiment_supervisor', 'run', '--name',
                          'frozen_policy_evaluation_resume', '--kind', 'normal', '--timeout', str(timeout),
                          str(run / 'stage_01_supervision'), '--', *command, '--resume']
            blocks.append('Resume only if the original evaluation owner is stopped, the saved supervisor is '
                          'still active, and both the stage and global deadlines have not expired. The '
                          'original stop timestamps remain unchanged:\n\n```bash\n' + shlex.join(supervised)
                          + '\n```\n\nIf that supervisor is already terminal, follow the continuation rule '
                          'in the campaign manifest; do not reset the eight-hour clock.')
    else:
        blocks.append('The exact evaluation command has not yet been recorded; inspect '
                      '`stage_01_supervision/run_manifest.json` before resuming.')
    assessment_stage = next((item for item in reversed(stages) if item.get('name') in
                             {'final_assessment', 'assessment', 'assessment_evaluation'}), None)
    commands = campaign.get('commands') if isinstance(campaign.get('commands'), dict) else {}
    assessment_command = commands.get('stage01_automatic_assessment')
    automatic_commands = stage_info.get('automatic_commands')
    if not assessment_command and isinstance(automatic_commands, list) and automatic_commands:
        assessment_command = automatic_commands[0]
    if isinstance(assessment_command, list):
        assessment_command = shlex.join([str(value) for value in assessment_command])
    if isinstance(assessment_command, str) and assessment_command:
        blocks.append('Recorded assessment command:\n\n```bash\n' + assessment_command + '\n```')
    elif assessment_stage and isinstance(assessment_stage.get('command'), list):
        blocks.append('Recorded assessment command:\n\n```bash\n'
                      + shlex.join([str(value) for value in assessment_stage['command']]) + '\n```')
    elif (contract.get('input_pairs_path') and plan.get('split') and plan.get('layout_limit_per_family')
          and plan.get('contexts_per_layout') and models):
        generated = ['uv', 'run', '--offline', '--no-sync', 'python', '-m', 'scripts.assess_overnight_cpu_policy',
                     '--pairs', str(contract['input_pairs_path']), '--shards', str(stage / 'evaluation/shards'),
                     '--models', *models, '--split', str(plan['split']), '--max-boards',
                     str(plan['layout_limit_per_family']), '--max-contexts', str(plan['contexts_per_layout']),
                     '--output', str(assessment_dir)]
        blocks.append('Reproducible assessment command constructed from the frozen plan and paths '
                      '(the finalizer records its exact launch command in the campaign manifest):\n\n```bash\n'
                      + shlex.join(generated) + '\n```')
    blocks.append('Next-stage entry: read the [active campaign pointer](../overnight_cpu_active.json) and '
                  '[campaign manifest](campaign_manifest.json), preserving the original eight-hour deadline. '
                  'The next stage is not launched by the following read-only check:\n\n```bash\n'
                  + str(campaign.get('status_command') or 'uv run --offline --no-sync python -m scripts.overnight_cpu_campaign --require-time')
                  + '\n```')
    return '\n\n'.join(blocks) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else run / 'stage_01_report.md'
    report = make_report(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(report, encoding='utf-8')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()

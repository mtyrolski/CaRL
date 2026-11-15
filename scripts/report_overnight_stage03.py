"""Create the stage-three report from complete, hash-verified assessments.

Historical observations and newly generated responses remain separate. A
"fresh" response does not imply a previously unseen board or base model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shlex
import sys
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import bootstrap

Record = dict[str, Any]
METHOD_LABELS = {'base': 'Base', 'icl_legacy': 'ICL', 'in_context': 'ICL',
                 'lora_validation_220': 'LoRA 220', 'checkpoint220': 'LoRA 220'}
SOURCE_LABELS = {'proposal_gamma2_12x12x10': 'gamma2', 'synthetic_reverse_12x12x10': 'reverse',
                 'rl_validation_12x12x4': 'RL validation', 'real_12x12x4': 'real',
                 'generated_v1_12x12x10': 'generated v1'}
CONDITION_LABELS = {'original': 'Original', 'permuted': 'Permuted', 'rot90_ccw': 'Rotated',
                    'horizontal_reflection': 'Reflected'}
COLOURS = {'original': '#21618c', 'permuted': '#bd5626', 'rot90_ccw': '#47834f',
           'horizontal_reflection': '#865a9b'}


def file_sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path: Path) -> Record:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[Record]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[Record]) -> None:
    from io import StringIO

    stream = StringIO(newline='')
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    atomic_text(path, stream.getvalue())


def verified_assessment(path: Path) -> tuple[Record, list[Record], list[Record], list[Record], list[Record]]:
    """Fail closed on partial exports or modified assessment files."""
    manifest, status = read_json(path / 'assessment_manifest.json'), read_json(path / 'status.json')
    if status.get('state') != 'complete' or manifest['finished_jobs'] != manifest['planned_jobs']:
        raise ValueError(f'Assessment is not complete: {path}')
    for name in ('assessments.jsonl', 'summary.csv', 'paired_comparisons.csv', 'equivariance_summary.csv'):
        if manifest['outputs_sha256'].get(name) != file_sha(path / name):
            raise ValueError(f'Assessment output differs from manifest: {path / name}')
    for name, expected in manifest['inputs_sha256'].items():
        source = Path(name)
        actual = file_sha(source) if source.is_file() else None
        if actual != expected:
            raise ValueError(f'Assessment input differs from frozen snapshot: {source}')
    rows = read_jsonl(path / 'assessments.jsonl')
    if len(rows) != manifest['planned_jobs']:
        raise ValueError(f'Assessment raw record count differs from manifest: {path}')
    return (manifest, rows, read_csv(path / 'summary.csv'), read_csv(path / 'paired_comparisons.csv'),
            read_csv(path / 'equivariance_summary.csv'))


def board_stats(rows: list[Record], metric: str, seed: int = 20261115) -> dict[str, Any]:
    """One vote per original layout; multiple permutation seeds are averaged first."""
    planned = {row['layout_id'] for row in rows}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row['output_status'] != 'complete':
            continue
        if metric == 'valid':
            value = float(row['valid_nontrivial'])
        elif metric == 'bfs':
            if not row['valid_nontrivial']:
                value = 0.0
            elif isinstance(row.get('bounded_path_found'), bool):
                value = float(row['bounded_path_found'])
            else:
                continue
        else:
            raise ValueError('Unsupported board metric')
        grouped[row['layout_id']].append(value)
    means = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    ci = bootstrap(means, random.Random(seed), count=2000)
    return {'planned_boards': len(planned), 'known_boards': len(means),
            'mean': sum(means) / len(means) if means else None,
            'ci_low': ci[0] if ci else None, 'ci_high': ci[1] if ci else None,
            'planned_outputs': len(rows), 'completed_outputs': sum(row['output_status'] == 'complete' for row in rows),
            'missing_outputs': sum(row['output_status'] == 'missing' for row in rows),
            'unavailable_outputs': sum(row['output_status'] == 'permutation_unavailable' for row in rows)}


def source_counts(rows: list[Record]) -> dict[str, int]:
    by_layout: dict[str, str] = {}
    for row in rows:
        layout, source = row['layout_id'], row['source_family']
        if layout in by_layout and by_layout[layout] != source:
            raise ValueError('One layout has multiple source families in the same panel')
        by_layout[layout] = source
    return dict(sorted(Counter(by_layout.values()).items()))


def figure_data(history: list[Record], fresh: list[Record]) -> list[Record]:
    """Build three labelled panels; never pool historical and fresh cohorts."""
    old = [row for row in history if row['cohort_id'] == 'legacy_transfer_40' and row['family'] == '12x12x10'
           and row['source_family'] == 'proposal_gamma2_12x12x10' and row['transform'] == 'identity'
           and row['condition'] in ('original', 'permuted')]
    panels = [('historical_gamma2', old, ('base', 'icl_legacy', 'lora_validation_220'), ('original', 'permuted'))]
    for family, name in (('12x12x4', 'fresh_x4'), ('12x12x10', 'fresh_x10')):
        selected = [row for row in fresh if row['family'] == family and row['condition'] in ('original', 'd4')]
        panels.append((name, selected, ('base', 'in_context', 'checkpoint220'),
                       ('original', 'rot90_ccw', 'horizontal_reflection')))
    result = []
    for panel, rows, methods, conditions in panels:
        if not rows:
            raise ValueError(f'Missing planned figure panel: {panel}')
        mix = source_counts(rows)
        for method in methods:
            for condition in conditions:
                selected = [row for row in rows if row['method'] == method and
                            (row['condition'] == condition if condition in ('original', 'permuted') else
                             row['condition'] == 'd4' and row['transform'] == condition)]
                if not selected:
                    raise ValueError(f'Missing planned plot cases: {panel}/{method}/{condition}')
                identity = json.dumps([panel, method, condition], separators=(',', ':'))
                seed = int.from_bytes(hashlib.sha256(f'20261115\0{identity}'.encode()).digest()[:8], 'big')
                result.append({'panel': panel, 'method': method, 'condition': condition,
                               'source_layout_counts': mix, **board_stats(selected, 'valid', seed),
                               'valid_outputs': sum(row.get('valid_nontrivial') is True for row in selected),
                               'whole_goal_outputs': sum(row.get('requests_whole_goal') is True for row in selected)})
    return result


def mode_diagnostics(rows: list[Record]) -> list[Record]:
    """Count a structural mode among every parseable response, including invalid intents."""
    groups: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    keys = ('cohort_id', 'family', 'source_family', 'method', 'condition', 'transform')
    for row in rows:
        if row['condition'] != 'original' or row['transform'] != 'identity':
            continue
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        completed = [row for row in group if row['output_status'] == 'complete']
        parseable = [row for row in completed if row['json_valid']]
        signatures = Counter(row['intent_signature'] for row in parseable if row.get('intent_signature') is not None)
        mode_count = max(signatures.values(), default=0)
        winners = sorted(signature for signature, count in signatures.items() if count == mode_count)
        support = [row for row in completed if row.get('support_status') == 'two_support_examples']
        support_coord = [row for row in support if row.get('coordinate_effect_count', 0) > 0]
        result.append({**dict(zip(keys, key)), 'planned_outputs': len(group), 'completed_outputs': len(completed),
            'parseable_outputs': len(parseable), 'parseable_with_structural_signature': sum(signatures.values()),
            'parseable_without_structural_signature': len(parseable) - sum(signatures.values()),
            'most_common_structural_intent': winners[0] if winners else None,
            'tied_mode_count': len(winners), 'most_common_count': mode_count,
            'most_common_share_of_completed': mode_count / len(completed) if completed else None,
            'most_common_share_of_parseable': mode_count / len(parseable) if parseable else None,
            'two_support_outputs': len(support), 'two_support_outputs_with_coordinates': len(support_coord),
            'matches_support_coordinate': sum(row.get('matches_support_coordinate') is True for row in support),
            'matches_support_predicate_coordinate': sum(row.get('matches_support_predicate_coordinate') is True for row in support)})
    return result


def overlap_diagnostics(history_contexts: list[Record], frozen_contexts: list[Record]) -> Record:
    historical = {row['layout_id'] for row in history_contexts}
    fresh = {row['layout_id'] for row in frozen_contexts}
    if len(fresh) != len(frozen_contexts):
        raise ValueError('Frozen fresh contexts repeat a canonical layout')
    counts = Counter(f"{row['family']} / {row['source_family']}" for row in frozen_contexts
                     if row['layout_id'] in historical)
    cohorts = defaultdict(set)
    for row in history_contexts:
        cohorts[row['cohort_id']].add(row['layout_id'])
    return {'fresh_boards': len(fresh), 'historical_union_boards': len(historical),
            'fresh_overlap_union': len(fresh & historical), 'fresh_not_in_historical_union': len(fresh - historical),
            'overlap_by_family_source': dict(sorted(counts.items())),
            'overlap_by_historical_cohort': {name: len(fresh & layouts) for name, layouts in sorted(cohorts.items())}}


def token_cap_diagnostics(predictions: list[Record], token_cap: int) -> list[Record]:
    grouped: dict[tuple[str, str, str, str], list[Record]] = defaultdict(list)
    for row in predictions:
        if row.get('prediction_status') == 'complete':
            grouped[tuple(row[key] for key in ('family', 'source_family', 'method', 'transform'))].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        known = [row['generated_tokens'] for row in rows if isinstance(row.get('generated_tokens'), int)
                 and not isinstance(row['generated_tokens'], bool)]
        if any(count < 0 or count > token_cap for count in known):
            raise ValueError('Generated token count is outside the frozen cap')
        result.append({**dict(zip(('family', 'source_family', 'method', 'transform'), key)),
                       'completed_outputs': len(rows), 'token_counts_known': len(known),
                       'token_counts_unknown': len(rows) - len(known), 'token_cap': token_cap,
                       'at_token_cap': sum(count == token_cap for count in known)})
    return result


def symmetry_degeneracy(plan: Record) -> Record:
    cases: dict[tuple[str, str], bool] = {}
    for case in plan['cases']:
        if case['transform'] == 'identity':
            continue
        key = (case['context_id'], case['transform'])
        identical = case['original_rows'] == case['source_rows']
        if key in cases and cases[key] != identical:
            raise ValueError('Methods disagree on whether a D4 input changed')
        cases[key] = identical
    return {'planned_transformed_boards': len(cases), 'identical_to_original': sum(cases.values()),
            'by_transform': {name: {'planned': sum(transform == name for _, transform in cases),
                                    'identical': sum(value for (_, transform), value in cases.items() if transform == name)}
                             for name in sorted({transform for _, transform in cases})}}


def plot(output: Path, rows: list[Record]) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure = plt.figure(figsize=(8.0, 6.3))
    grid = figure.add_gridspec(2, 2, left=.085, right=.985, bottom=.19, top=.87, hspace=.61, wspace=.2)
    axes = {'historical_gamma2': figure.add_subplot(grid[0, :]), 'fresh_x4': figure.add_subplot(grid[1, 0]),
            'fresh_x10': figure.add_subplot(grid[1, 1])}
    for panel, axis in axes.items():
        data = [row for row in rows if row['panel'] == panel]
        methods = (('base', 'icl_legacy', 'lora_validation_220') if panel == 'historical_gamma2'
                   else ('base', 'in_context', 'checkpoint220'))
        conditions = (('original', 'permuted') if panel == 'historical_gamma2' else
                      ('original', 'rot90_ccw', 'horizontal_reflection'))
        offsets = (-.12, .12) if panel == 'historical_gamma2' else (-.16, 0.0, .16)
        for index, method in enumerate(methods):
            for offset, condition in zip(offsets, conditions, strict=True):
                item = next(row for row in data if row['method'] == method and row['condition'] == condition)
                x, value = index + offset, item['mean']
                if value is None:
                    axis.text(x, .03, 'n/a', ha='center', color=COLOURS[condition], fontsize=7)
                    continue
                low, high = item['ci_low'], item['ci_high']
                if low is not None and high is not None:
                    axis.vlines(x, low, high, color=COLOURS[condition], linewidth=0.9)
                    axis.hlines([low, high], x-.025, x+.025, color=COLOURS[condition], linewidth=0.9)
                axis.plot(x, value, 'o', color=COLOURS[condition], markersize=4, clip_on=False)
        mix = data[0]['source_layout_counts']
        description = ', '.join(f'{SOURCE_LABELS.get(key, key)}: {value}' for key, value in mix.items())
        n = sum(mix.values())
        title = ('Historical gamma2' if panel == 'historical_gamma2' else
                 'Fresh responses · 12×12×4' if panel == 'fresh_x4' else 'Fresh responses · 12×12×10')
        axis.set_title(f'{title}  ·  {n} planned original boards\n{description}', fontsize=9)
        axis.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
        axis.set_xlim(-.5, 2.5)
        axis.set_ylim(-.03, 1.03)
        axis.grid(axis='y', alpha=.2, linewidth=.6)
        axis.set_axisbelow(True)
        if panel != 'fresh_x10':
            axis.set_ylabel('Valid unsatisfied intent / known output')
    handles = [Line2D([], [], marker='o', linestyle='none', color=COLOURS[name], label=CONDITION_LABELS[name])
               for name in ('original', 'permuted', 'rot90_ccw', 'horizontal_reflection')]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=4, frameon=False, fontsize=8)
    figure.text(.5, .015, 'Bars: 95% bootstrap of original boards; missing output is not scored.\n'
                'Permutations reuse saved responses across three seeds; D4 orientations are not new boards.\n'
                'A valid intent may request the entire puzzle goal; validity does not imply usefulness.\n'
                'Endpoint intervals can be degenerate in small samples.',
                ha='center', va='bottom', fontsize=7.5)
    for extension in ('pdf', 'svg', 'png'):
        path = output / f'stage_03_figure.{extension}'
        temporary = output / f'.stage_03_figure.{extension}.{os.getpid()}.tmp'
        try:
            figure.savefig(temporary, format=extension, dpi=190, bbox_inches='tight')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(figure)


def display_ratio(numerator: Any, denominator: Any) -> str:
    n, d = int(numerator), int(denominator)
    return f'{n}/{d} ({100*n/d:.1f}%)' if d else f'{n}/{d} (n/a)'


def unresolved_count(row: Record) -> int:
    count = (int(row['valid_nontrivial']) - int(row['bounded_paths_found'])
             - int(row['bfs_infrastructure_errors']))
    if count < 0:
        raise ValueError('BFS outcomes exceed the valid-intent denominator')
    return count


def mode_text(signature: str | None) -> str:
    if signature is None:
        return 'n/a'
    try:
        item = json.loads(signature)
        parsed = {key: [json.loads(value) for value in item[key]] for key in ('effects', 'preserve')}
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':')).replace('|', '\\|')
    except (ValueError, KeyError, TypeError):
        return signature.replace('|', '\\|')


def representative_examples(history: list[Record], fresh: list[Record]) -> list[Record]:
    """Select by declared source/ID ordering, never by solver success."""
    eligible = [row for row in history + fresh if row['condition'] == 'original'
                and row['transform'] == 'identity' and row['output_status'] == 'complete']
    eligible.sort(key=lambda row: tuple(row[key] for key in ('cohort_id', 'family', 'source_family', 'method',
                                                               'layout_id', 'context_id', 'job_id')))
    chosen: set[str] = set()
    result = []
    criteria = (
        ('First wall error', lambda row: row.get('wall_error') is True),
        ('First range error on a different output', lambda row: row.get('range_error') is True),
        ('First fresh valid intent with a recorded BFS result',
         lambda row: row['cohort_id'] == 'stage03_fresh_evaluation' and row.get('valid_nontrivial') is True
                     and isinstance(row.get('bounded_path_found'), bool)),
    )
    for label, matches in criteria:
        row = next((item for item in eligible if item['job_id'] not in chosen and matches(item)), None)
        if row is not None:
            chosen.add(row['job_id'])
        result.append({'criterion': label, 'record': row})
    return result


def fenced(text: str, language: str = '') -> str:
    lengths = [len(match.group()) for match in re.finditer(r'`+', text)]
    marker = '`' * max([4, *(length + 1 for length in lengths)])
    return f'{marker}{language}\n{text}\n{marker}'


def report_text(history_summary: list[Record], fresh_summary: list[Record], mode_rows: list[Record],
                decision: Record, recommendation: Record, fresh_status: Record, fresh_plan: Record,
                historical_manifest: Record, fresh_manifest: Record, chronology: Record, overlap: Record,
                cap_rows: list[Record], equivariance: list[Record], paired: list[Record], examples: list[Record],
                symmetry: Record, benchmark: Record, runtime: Record, fresh_contract: Record, links: Record) -> str:
    lines = ['# Stage 3: Sokoban intent grounding', '',
        'This is an exploratory generator audit, not a measurement of an intent-conditioned controller '
        'or complete-task planning. “Fresh” means newly generated Qwen responses. The boards were already '
        'used in stages 1/2, and 10-box sources had earlier exposure; base-model and CLLP training exposure '
        'is unknown. Known adapter training layouts were checked separately, as recorded below.', '',
        'Every method uses the same strict unique-key JSON parser. The primary metric removes at most one '
        'complete enclosing Markdown fence symmetrically; strict results remain available in each raw assessment. '
        'The requested predicates must be unsatisfied individually at the start. Wall, range, already-satisfied '
        'and preservation errors may overlap. A valid intent may request the whole puzzle goal; '
        'validity alone does not establish usefulness.', '',
        'For D4, the full current board and reference actions are transformed, and generated predicates '
        'are mapped back to the original orientation. Reference actions verify the simulator transform and '
        'are not model inputs. The same two independent ICL support examples remain fixed in all orientations; '
        'their own boards and annotations remain mutually consistent.', '',
        'Historical legacy ICL prompts were saved at inference time. Legacy base and LoRA prompts '
        'were reconstructed from the corresponding historical source and cached tokenizer; their '
        'original rendered prompts were not saved. Representation-cohort prompts were captured.', '',
        'The deterministic node/depth budgets are 16 primitive actions and 2500 expanded BFS nodes. '
        'A 5 s safety timeout is recorded separately; a cutoff means unresolved. Found action paths are '
        'replayed in the real environment. BFS is a solver and does not establish learned policy achievement.', '',
        '![Board-level valid intent rates for original, permuted and D4-transformed inputs](stage_03_figure.png)', '',
        'Three seed-fixed derangements preserve the raw output multiset, including invalid and missing outputs, '
        'within each cohort, board family and source. They reuse the same responses: the three permuted rows '
        'per board are not independent generations. Board-bootstrap intervals condition on these mappings. '
        'Intervals such as [0,0] or [1,1] may be a degenerate empirical bootstrap of uniform outcomes '
        'in a small sample, not proof of zero population uncertainty. '
        'Historical cohorts share some layouts but are never pooled. '
        f"For the fresh D4 inputs, {symmetry['identical_to_original']}/{symmetry['planned_transformed_boards']} "
        'transformed boards are identical to their originals.', '',
        f"Development decision: `{decision.get('recommendation')}`; provisional={decision.get('provisional')}. "
        f"Subsequent deterministic fallback: `{recommendation.get('recommendation')}`; "
        f"provisional={recommendation.get('provisional')}. The original decision was saved before fresh evaluation; "
        'the separate fallback file was written after evaluation started as a deterministic interpretation of the earlier rule.', '',
        '## Decision chronology', '', chronology['description'], '', '## Results and costs', '',
        f"Fresh generation: {fresh_status['completed']}/{fresh_status['planned']} outputs completed, "
        f"{fresh_status.get('missing', 0)} missing and {fresh_status.get('infra_errors', 0)} infrastructure errors; "
        f"state `{fresh_status['state']}`. Generation time {fresh_status.get('generation_seconds', 'unknown')} s, "
        f"wall time {fresh_status.get('elapsed_seconds', 'unknown')} s, "
        f"{fresh_status.get('generated_tokens', 'unknown')} generated tokens. No paid inference or new training was used.",
        f"Across fresh responses, {sum(row['at_token_cap'] for row in cap_rows)}/"
        f"{sum(row['token_counts_known'] for row in cap_rows)} with known token counts reached the configured output "
        'token cap. Reaching the cap alone does not prove that a response was truncated.',
        '', '| Assessment | Planned jobs | Completed jobs | BFS timeouts | BFS/replay errors |',
        '|---|---:|---:|---:|---:|',
        f"| Historical v2 | {historical_manifest['planned_jobs']} | {historical_manifest['finished_jobs']} | "
        f"{historical_manifest['timeouts']} | {historical_manifest['solver_infrastructure_errors']} |",
        f"| Fresh responses | {fresh_manifest['planned_jobs']} | {fresh_manifest['finished_jobs']} | "
        f"{fresh_manifest['timeouts']} | {fresh_manifest['solver_infrastructure_errors']} |", '']
    lines.append('Development throughput benchmark: CPU FP32, four threads, batch size 3; '
                 'the wall time of each batch is allocated equally over its outputs. '
                 + ', '.join(f"{METHOD_LABELS[method]} {benchmark['method_statistics'][method]['seconds_per_output']:.2f} s/output "
                             f"({benchmark['method_statistics'][method]['outputs']} outputs)"
                             for method in ('base', 'in_context', 'checkpoint220')) + '. '
                 'This measurement informed the frozen 24+24 board allocation.')
    lines.append('')
    exposure = Counter(str(row.get('adapter_training_exposure')) for row in fresh_plan['context_manifest'])
    for name, source_rows in (('Historical', history_summary), ('Fresh', fresh_summary)):
        calls = sum(int(row['bfs_actual_calls']) for row in source_rows)
        nodes = sum(int(row['bfs_expanded_nodes_actual']) for row in source_rows)
        edges = sum(int(row['bfs_edges_actual']) for row in source_rows)
        seconds = sum(float(row['bfs_seconds_actual']) for row in source_rows)
        lines.append(f'{name} bounded search: {calls} actual solver calls, {nodes} expanded nodes, '
                     f'{edges} tested primitive edges, {seconds:.2f} s search time; exact cached checks '
                     'are counted once and replay time is recorded separately in raw outputs.')
    lines.append('')
    lines += [f'Fresh evaluation cohort: {len(fresh_plan["context_manifest"])} original boards; '
              f'{overlap["fresh_overlap_union"]} also occur in the two historical generator cohorts '
              f'and {overlap["fresh_not_in_historical_union"]} do not. '
              f'Overlap by family and source: {overlap["overlap_by_family_source"]}. '
              'All fresh-response boards were already in the campaign stage 1/2 evaluation.',
              'The four-box boards come from RL validation, while the ten-box boards come from '
              'gamma2 and reverse-generated sources. Differences between these families cannot be '
              'attributed to box count alone.',
              f'Known adapter-training overlap flags: {dict(exposure)}. '
              'These checks do not establish '
              'that pretraining or CLLP checkpoints never saw the boards.', '',
        '| Cohort | Family | Source | Method | Pairing / transform | Complete / planned | Invalid JSON | '
        'Invalid intent | Valid / complete | BFS paths / complete | No path / valid | '
        'Wall / range / already true | Goal requests | Solver errors |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in history_summary + fresh_summary:
        pairing = row['condition'] if row['transform'] == 'identity' else row['transform']
        lines.append(f"| {row['cohort_id']} | {row['family']} | {row['source_family']} | {row['method']} | {pairing} | "
                     f"{row['completed_outputs']}/{row['requested_outputs']} | "
                     f"{row['invalid_json']} | {row['invalid_intent']} | "
                     f"{display_ratio(row['valid_nontrivial'], row['completed_outputs'])} | "
                     f"{display_ratio(row['bounded_paths_found'], row['completed_outputs'])} | "
                     f"{display_ratio(unresolved_count(row), row['valid_nontrivial'])} | "
                     f"{row['wall_errors']} / {row['range_errors']} / {row['any_effect_initially_true']} | "
                     f"{row['whole_goal_requests']} | "
                     f"{row['bfs_infrastructure_errors']} |")
    lines += ['', 'The permuted requested count includes three reassignments of each saved output. '
              'Denominators above are response slots, not independent boards. Missing and unresolvable results '
              'remain visible in the underlying CSV; solver errors make the path numerator a lower bound. '
              '“No path” counts valid intents without a found path or solver error, including search cutoffs; '
              'it does not establish impossibility.', '',
              '## Paired validity differences', '',
              'Comparison minus original; intervals resample original layouts. Permutation intervals are '
              'conditional on the three frozen assignments and saved response pool.', '',
              '| Cohort | Family | Source | Method | Comparison | Paired layouts | Difference | 95% interval |',
              '|---|---|---|---|---|---:|---:|---:|']
    for row in paired:
        mean = float(row['mean_comparison_minus_original']) if row['mean_comparison_minus_original'] else None
        low = float(row['ci_low']) if row['ci_low'] else None
        high = float(row['ci_high']) if row['ci_high'] else None
        estimate = f'{mean:+.3f}' if mean is not None else 'n/a'
        bounds = f'[{low:+.3f}, {high:+.3f}]' if low is not None and high is not None else 'n/a'
        name = row['condition'] if row['condition'] == 'permuted' else row['transform']
        lines.append(f"| {row['cohort_id']} | {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{name} | {row['paired_layouts']} | {estimate} | {bounds} |")
    lines += ['', '## Intent diversity on original outputs', '',
              'Predicate counts and conjunction sizes below are calculated among valid original outputs. '
              'Each array records counts of predicates or intent lengths, not independent board counts.', '',
              '| Cohort | Family | Source | Method | Valid outputs | Predicate counts | Conjunction sizes |',
              '|---|---|---|---|---:|---|---|']
    for row in history_summary + fresh_summary:
        if row['condition'] != 'original' or row['transform'] != 'identity':
            continue
        lines.append(f"| {row['cohort_id']} | {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{row['valid_nontrivial']} | `{row['effect_counts_valid']}` | `{row['conjunction_sizes_valid']}` |")
    lines += ['',
              'Most common structured intents below include all JSON-parseable original, identity responses, '
              'including semantically invalid ones; the primary share uses all completed outputs as denominator. '
              'Instruction text is excluded from the structural signature. ICL coordinate reuse uses only '
              'outputs whose exact prompts contain two parseable support examples.', '',
              '| Cohort | Family | Source | Method | Parseable / complete | Most common structured intent | '
              'Mode / complete | Mode / parseable | Support match / two-support outputs | Support match / coordinate outputs |',
              '|---|---|---|---|---:|---|---:|---:|---:|---:|']
    for row in mode_rows:
        lines.append(f"| {row['cohort_id']} | {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{display_ratio(row['parseable_outputs'], row['completed_outputs'])} | "
                     f"`{mode_text(row['most_common_structural_intent'])}` | "
                     f"{display_ratio(row['most_common_count'], row['completed_outputs'])} | "
                     f"{display_ratio(row['most_common_count'], row['parseable_outputs'])} | "
                     f"{display_ratio(row['matches_support_coordinate'], row['two_support_outputs'])} | "
                     f"{display_ratio(row['matches_support_coordinate'], row['two_support_outputs_with_coordinates'])} |")
    lines += ['', '## D4 structured-intent agreement', '',
              'Agreement compares the generated structured predicates after inverse coordinate mapping '
              'only when both the original and transformed outputs are valid. A mismatch does not prove '
              'a lack of understanding: multiple different intents can be valid for the same board. '
              'Natural-language instructions are not translated.', '',
              '| Family | Source | Method | Transform | Planned boards | Valid both | Same structured intent / valid both | Untranslated instructions |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for row in equivariance:
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | {row['transform']} | "
                     f"{row['planned_contexts']} | {row['valid_both']} | "
                     f"{display_ratio(row['same_structured_intent'], row['valid_both'])} | "
                     f"{row['instruction_untranslated']} |")
    lines += ['', '## Representative outputs', '',
              'Selection is deterministic by cohort, family, source, method, layout, context and job ID: '
              'the first wall error, the first different range error, and the first fresh valid intent with '
              'a recorded BFS result. The last criterion does not depend on whether the solver succeeds.']
    for item in examples:
        lines += ['', f"### {item['criterion']}", '']
        row = item['record']
        if row is None:
            lines.append('No eligible output was available.')
            continue
        lines.append(f"{row['cohort_id']} · {row['family']} · {row['source_family']} · {row['method']} · "
                     f"layout `{row['layout_id']}`. Valid: {row['valid_nontrivial']}; "
                     f"bounded BFS status: `{row.get('bfs_status')}`; path found: {row.get('bounded_path_found')}.")
        lines.append('Initial board:')
        lines.append(fenced('\n'.join(row['source_rows'])))
        lines.append('Raw model response:')
        lines.append(fenced(row['raw_output'], 'text'))
        lines.append('Checker errors: ' + json.dumps(row.get('validation_errors', []), ensure_ascii=False))
        search = row.get('bfs_search')
        if isinstance(search, dict):
            lines.append(f"BFS expanded {search['expanded_nodes']} nodes; action path: "
                         f"`{search.get('witness_actions')}`. This is a solver path, not a controller rollout.")
    lines += ['', 'The machine-readable `stage_03_mode_diagnostics.csv` additionally records ties, '
              'parseable responses without a structural signature, and the number with coordinate effects. '
              'The figure reports board-level valid-intent rates and source composition; paired bootstrap '
              'differences, D4 structured agreement, failed outputs and BFS costs are in the component assessments.', '',
              f"[Historical assessment](<{links['historical']}>) · "
              f"[Fresh assessment](<{links['fresh']}>) · "
              f"[Historical outputs and prompts](<{links['historical_prompts']}>) · "
              f"[Fresh full prompts](<{links['prompts']}>) · "
              '[Figure data](stage_03_figure_data.csv)', '',
              'The results do not establish transfer, improved complete-task planning, or an advantage from '
              'policy-relative scale conditioning.']
    lines += ['', '## Reproduction and checks', '',
              f"Hardware: {runtime['cpu']}; inference contract: device `{fresh_contract['device']}`, "
              f"dtype `{fresh_contract['dtype']}`, {fresh_contract['threads']} threads and "
              f"batch size {fresh_contract['batch_size']}. "
              f"Qwen3-0.6B revision `{runtime['model_revision']}`. "
              f"Elapsed stage time at report generation: {runtime['elapsed_seconds']:.1f} s "
              f"from {runtime['stage_start_utc']} (monotonic clock).", '',
              f"The repository `make test` completed with {runtime['repository_tests_passed']} tests passed; "
              'the focused importer, '
              'generator/input and assessment tests also passed. The individual recorded checks follow:', '']
    for check in runtime['checks']:
        lines.append(f"- `{check['name']}`: {check['result']} (exit {check['exit_code']}).")
    lines += ['', 'An initial independent prompt-input audit exited 1 because its checker compared '
              'the hash of a canonical prompt list with the hash of JSONL bytes. The corrected '
              'canonical-hash audit and subsequent verification both passed; this was a checker '
              'hash-definition error, not a demonstrated prompt error.', '',
              'A bootstrap seed reproducibility defect on resume was corrected in historical v2. '
              f"An independent v1/v2 audit found {runtime['v1_v2']['raw_rows'][0]} unchanged raw outcomes "
              f"and {runtime['v1_v2']['summary_rows'][0]} unchanged point-estimate rows; "
              'confidence intervals may differ because their random seeds were corrected.']
    lines += ['', 'Recorded supervised commands, from the repository root:', '']
    for label, command in runtime['commands'].items():
        lines += [f'**{label}**', '', fenced(command, 'sh'), '']
    lines += ['The recorded generator command contains an existing output directory and absolute '
              '`--stop-epoch` / `--stop-monotonic` values that are now in the past. A new inference '
              'run requires a separate directory and an appropriate supervised deadline; the current '
              'campaign clock must not be reset. The assessment and report can be reproduced from '
              'the retained raw results without repeating model inference.', '',
              'The following commands use separate output directories; select other unused paths if these '
              'already exist. They do not call the models:']
    for label, command in runtime['reproduction_commands'].items():
        lines += ['', f'**{label}**', '', fenced(command, 'sh')]
    return '\n'.join(lines) + '\n'


def chronology_excerpt(path: Path) -> dict[str, str]:
    text = path.read_text(encoding='utf-8')
    heading = '## Decision chronology'
    if heading not in text:
        raise ValueError('Preliminary report is missing the decision chronology')
    after = text.split(heading, 1)[1]
    excerpt = after.split('\n## ', 1)[0].strip()
    if not excerpt or '08:54:56' not in excerpt or '08:54:05' not in excerpt:
        raise ValueError('Decision chronology does not contain the recorded development amendment')
    return {'description': excerpt, 'sha256': hashlib.sha256(excerpt.encode()).hexdigest()}


def command_to_new_output(command: list[str], destination: Path) -> str:
    if command.count('--output') != 1:
        raise ValueError('Expected exactly one output argument in assessment command')
    updated = [argument for argument in command if argument != '--resume']
    index = updated.index('--output') + 1
    if index >= len(updated):
        raise ValueError('Missing assessment output path')
    updated[index] = str(destination)
    return shlex.join(updated)


def runtime_metadata(campaign: Path, revision: str, fresh_run: Path) -> Record:
    supervisor = read_json(campaign / 'stage_03_supervision/run_manifest.json')
    clock = read_json(campaign / 'stage_03_clock.json')
    stages = supervisor['stages']
    def named(name: str, *, require_success: bool = True) -> Record:
        matches = [stage for stage in stages if stage['name'] == name]
        if len(matches) != 1 or (require_success and matches[0].get('exit_code') != 0):
            raise ValueError(f'Required supervised stage is missing or unsuccessful: {name}')
        return matches[0]
    history = named('stage03_historical_permutation_v2')
    generator = named('stage03_fresh_evaluation', require_success=False)
    def matches_fresh_assessment(stage: Record) -> bool:
        command = stage.get('command', [])
        if ('scripts.overnight_stage03_assess' not in command or '--generator-run' not in command
                or stage.get('exit_code') != 0):
            return False
        index = command.index('--generator-run') + 1
        if index >= len(command):
            return False
        path = Path(command[index])
        return (path if path.is_absolute() else Path(supervisor['repository']) / path).resolve() == fresh_run.resolve()
    fresh_candidates = [stage for stage in stages if matches_fresh_assessment(stage)]
    if len(fresh_candidates) != 1:
        raise ValueError('Cannot identify the completed fresh-assessment command uniquely')
    checks = []
    required_tests = ('stage03_repository_tests', 'stage03_importer_tests', 'stage03_generation_tests_final',
                      'stage03_assessment_tests_v2', 'stage03_development_decision_tests')
    report_tests = sorted({stage['name'] for stage in stages if stage['name'].startswith('stage03_report_tests')})
    for name in (*required_tests, *report_tests):
        matches = [stage for stage in stages if stage['name'] == name]
        if not matches:
            raise ValueError(f'Required test stage missing: {name}')
        stage = matches[-1]
        text = Path(stage['output']).read_text(encoding='utf-8')
        summaries = re.findall(r'\d+ passed(?:[^\n]*)', text)
        if stage.get('exit_code') != 0 or not summaries:
            raise ValueError(f'Required test stage failed or has no test summary: {name}')
        checks.append({'name': name, 'result': summaries[-1].strip(' ='), 'exit_code': stage['exit_code']})
    repository = next(item for item in checks if item['name'] == 'stage03_repository_tests')
    repository_count = int(re.match(r'(\d+) passed', repository['result']).group(1))
    try:
        lines = Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines()
        cpu = next(line.split(':', 1)[1].strip() for line in lines if line.startswith('model name'))
    except (OSError, StopIteration):
        cpu = 'CPU model unavailable'
    first_audit = [stage for stage in stages if stage['name'] == 'fresh_input_audit_final']
    if len(first_audit) != 1 or first_audit[0].get('exit_code') != 1:
        raise ValueError('Recorded initial prompt-audit failure is missing or changed')
    named('fresh_input_audit_canonical_hash')
    named('fresh_input_audit_verify')
    v1_v2 = read_json(campaign / 'stage_03/historical_v1_v2_audit.json')
    if v1_v2.get('state') != 'passed':
        raise ValueError('The historical v1/v2 comparison has not passed')
    reproduction_root = campaign / 'stage_03/reproduction'
    history_reproduction = reproduction_root / 'historical'
    fresh_reproduction = reproduction_root / 'fresh'
    report_reproduction = reproduction_root / 'report'
    return {'cpu': cpu, 'model_revision': revision, 'stage_start_utc': clock['start_utc'],
            'elapsed_seconds': max(0.0, time.monotonic() - clock['start_monotonic']),
            'repository_tests_passed': repository_count, 'checks': checks, 'v1_v2': v1_v2,
            'commands': {'Historical v2 assessment': shlex.join(history['command']),
                         'Fresh generation (recorded, see deadline note)': shlex.join(generator['command']),
                         'Fresh assessment from saved output': shlex.join(fresh_candidates[0]['command']),
                         'Report from saved assessments': shlex.join([sys.executable, '-m',
                             'scripts.report_overnight_stage03', *sys.argv[1:]])},
            'reproduction_commands': {
                'Historical assessment to a new directory': command_to_new_output(history['command'], history_reproduction),
                'Fresh assessment to a new directory': command_to_new_output(fresh_candidates[0]['command'], fresh_reproduction),
                'Report from the reproduced assessments': shlex.join(['env', 'MPLBACKEND=Agg',
                    'MPLCONFIGDIR=/tmp/carl_overnight_matplotlib', 'uv', 'run', '--offline', '--no-sync',
                    'python', '-m',
                    'scripts.report_overnight_stage03', '--campaign', str(campaign), '--fresh-run', str(fresh_run),
                    '--historical', str(history_reproduction), '--fresh', str(fresh_reproduction),
                    '--output', str(report_reproduction)])}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--fresh-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    stage = campaign / 'stage_03'
    historical = (args.historical or stage / 'assessment_historical_v2').resolve()
    fresh = (args.fresh or stage / 'assessment_fresh').resolve()
    fresh_run = (args.fresh_run or stage / 'fresh_evaluation').resolve()
    output = (args.output or campaign).resolve()
    history_manifest, history_rows, history_summary, history_pairs, _history_equivariance = verified_assessment(historical)
    fresh_manifest, fresh_rows, fresh_summary, fresh_pairs, fresh_equivariance = verified_assessment(fresh)
    fresh_status = read_json(fresh_run / 'status.json')
    if fresh_status.get('state') not in ('complete', 'partial', 'failed', 'stopped'):
        raise ValueError('Fresh model generation has not stopped yet')
    fresh_plan = read_json(fresh_run / 'plan.json')
    fresh_contract = read_json(fresh_run / 'contract.json')
    fresh_predictions = read_jsonl(fresh_run / 'predictions.jsonl') if (fresh_run / 'predictions.jsonl').is_file() else []
    frozen = read_json(stage / 'fresh_evaluation_freeze.json')
    for key in ('device', 'dtype', 'threads', 'batch_size'):
        if frozen.get(key) != fresh_contract.get(key):
            raise ValueError(f'Frozen {key} differs from the generation contract')
    history_contexts_path = stage / 'data/contexts.jsonl'
    history_contexts = read_jsonl(history_contexts_path)
    expected_contexts = {(row['context_id'], row['layout_id'], row['family'], row['source_family'])
                         for row in frozen['contexts']}
    planned_contexts = {(row['context_id'], row['layout_id'], row['family'], row['source_family'])
                        for row in fresh_plan['context_manifest']}
    if expected_contexts != planned_contexts or len(planned_contexts) != len(frozen['contexts']):
        raise ValueError('Fresh generation contexts differ from the pre-evaluation freeze')
    if fresh_status.get('planned') != len(fresh_plan['cases']):
        raise ValueError('Fresh status and frozen case counts disagree')
    expected_input = fresh_manifest['inputs_sha256']
    for path in (fresh_run / 'plan.json', fresh_run / 'contract.json', fresh_run / 'prompts.jsonl',
                 fresh_run / 'predictions.jsonl'):
        key = str(path.resolve())
        actual = file_sha(path) if path.is_file() else None
        if key not in expected_input or expected_input[key] != actual:
            raise ValueError('Fresh assessment does not match the stopped generator snapshot')
    decision_path, recommendation_path, plan_path = (stage / name for name in
        ('development_decision.json', 'development_recommendation.json', 'evaluation_plan.json'))
    decision, recommendation = read_json(decision_path), read_json(recommendation_path)
    if read_json(plan_path)['model_revision'] != fresh_contract['model_revision']:
        raise ValueError('Frozen model revision differs from the generation contract')
    if (recommendation.get('source_decision_sha256') != file_sha(decision_path)
            or recommendation.get('source_plan_sha256') != file_sha(plan_path)
            or decision.get('source_sha256', {}).get('stage_plan') != file_sha(plan_path)
            or decision.get('test_results_accessed') is not False):
        raise ValueError('Development decision or fallback provenance differs')
    development_run = Path(decision['development_run'])
    for name, filename in (('plan', 'plan.json'), ('contract', 'contract.json'),
                           ('predictions', 'predictions.jsonl'), ('status', 'status.json')):
        if decision['source_sha256'][name] != file_sha(development_run / filename):
            raise ValueError(f'Development decision input changed: {name}')
    prior_audit = read_json(stage / 'historical_v1_v2_audit.json')
    if prior_audit.get('state') != 'passed':
        raise ValueError('Historical v1/v2 comparison has not passed')
    if not all(row['cohort_id'] == 'stage03_fresh_evaluation' for row in fresh_rows):
        raise ValueError('Fresh assessment contains an unexpected cohort')
    cap = int(fresh_contract['max_new_tokens'])
    cap_rows = token_cap_diagnostics(fresh_predictions, cap)
    overlap = overlap_diagnostics(history_contexts, frozen['contexts'])
    symmetry = symmetry_degeneracy(fresh_plan)
    modes = mode_diagnostics(history_rows + fresh_rows)
    paired = [row for row in history_pairs if row['metric'] == 'valid' and row['condition'] == 'permuted']
    paired += [row for row in fresh_pairs if row['metric'] == 'valid' and row['condition'] == 'd4']
    examples = representative_examples(history_rows, fresh_rows)
    plotted = figure_data(history_rows, fresh_rows)
    chronology = chronology_excerpt(campaign / 'stage_03_report.md')
    benchmark = read_json(development_run / 'status.json')
    runtime = runtime_metadata(campaign, read_json(plan_path)['model_revision'], fresh_run)
    output.mkdir(parents=True, exist_ok=True)
    links = {'historical': os.path.relpath(historical / 'report.md', output),
             'fresh': os.path.relpath(fresh / 'report.md', output),
             'historical_prompts': os.path.relpath(stage / 'data/outputs.jsonl', output),
             'prompts': os.path.relpath(fresh_run / 'prompts.jsonl', output)}
    write_csv(output / 'stage_03_figure_data.csv', plotted)
    write_csv(output / 'stage_03_mode_diagnostics.csv', modes)
    write_csv(output / 'stage_03_token_cap_counts.csv', cap_rows)
    write_csv(output / 'stage_03_paired_validity.csv', paired)
    atomic_text(output / 'stage_03_examples.json', json.dumps(examples, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    plot(output, plotted)
    report = report_text(history_summary, fresh_summary, modes, decision, recommendation, fresh_status, fresh_plan,
                         history_manifest, fresh_manifest, chronology, overlap, cap_rows, fresh_equivariance,
                         paired, examples, symmetry, benchmark, runtime, fresh_contract, links)
    atomic_text(output / 'stage_03_report.md', report)
    summary = {'schema_version': 1, 'overlap': overlap, 'symmetry_degeneracy': symmetry,
               'token_cap': cap, 'token_cap_groups': cap_rows,
               'mode_diagnostics': modes, 'history_summary': history_summary, 'fresh_summary': fresh_summary,
               'history_paired': history_pairs, 'fresh_paired': fresh_pairs, 'fresh_equivariance': fresh_equivariance,
               'fresh_generation_status': fresh_status,
               'development_decision': decision, 'development_recommendation': recommendation, 'runtime': runtime}
    atomic_text(output / 'stage_03_summary.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    source_paths = [stage / name for name in ('development_decision.json', 'development_recommendation.json',
        'evaluation_plan.json', 'fresh_evaluation_freeze.json', 'historical_v1_v2_audit.json',
        'data/contexts.jsonl', 'data/outputs.jsonl')]
    source_paths += [historical / 'assessment_manifest.json', fresh / 'assessment_manifest.json',
                     fresh_run / 'plan.json', fresh_run / 'contract.json', fresh_run / 'status.json',
                     development_run / 'status.json', campaign / 'stage_03_clock.json']
    if (fresh_run / 'predictions.jsonl').is_file():
        source_paths.append(fresh_run / 'predictions.jsonl')
    filenames = ('stage_03_report.md', 'stage_03_summary.json', 'stage_03_figure_data.csv',
                 'stage_03_mode_diagnostics.csv', 'stage_03_token_cap_counts.csv',
                 'stage_03_paired_validity.csv', 'stage_03_examples.json',
                 'stage_03_figure.pdf', 'stage_03_figure.svg', 'stage_03_figure.png')
    manifest = {'schema_version': 1, 'command': sys.argv, 'reporter_sha256': file_sha(Path(__file__)),
        'input_sha256': {str(path.resolve()): file_sha(path) for path in source_paths},
        'chronology_excerpt_sha256': chronology['sha256'], 'chronology_excerpt': chronology['description'],
        'mutable_supervisor_log': str((campaign / 'stage_03_supervision/decisions.log').resolve()),
        'recorded_commands': runtime['commands'], 'reproduction_commands': runtime['reproduction_commands'],
        'overlap': overlap, 'symmetry_degeneracy': symmetry, 'token_cap': cap,
        'outputs_sha256': {name: file_sha(output / name) for name in filenames}}
    atomic_text(output / 'stage_03_report_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(output), 'history_jobs': history_manifest['finished_jobs'],
        'fresh_jobs': fresh_manifest['finished_jobs'], 'fresh_known_outputs': fresh_status.get('completed'),
        'overlap': overlap, 'at_token_cap': sum(row['at_token_cap'] for row in cap_rows)}))


if __name__ == '__main__':
    main()

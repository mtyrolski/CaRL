"""Plot precomputed pool and action-budget sensitivity for fixed CLLP models.

Each point summarises absolute changes in u for the same successful targets;
contexts carry equal weight, and the supplied confidence intervals resample
canonical board layouts. This script does not select or modify rollouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

FAMILIES = ('12x12x4', '12x12x10')
CONTROLLERS = ('cllp8', 'cllp32')
VARIANTS = ('demo4_bfs6', 'demo4_bfs4')
POOL_COMPARISONS = (
    ('fixed', 'reduced', 'Fixed vs\nreduced'),
    ('fixed', 'demo_only', 'Fixed vs\ndemo only'),
    ('generator_base', 'generator_augmented', '4+4 vs\n4+4+GEN'),
)
BUDGET_COMPARISONS = ((8, 16, '8 → 16'), (8, 32, '8 → 32'), (16, 32, '16 → 32'))
COLOURS = {'cllp8': '#21618c', 'cllp32': '#bd5626'}
MARKERS = {'demo4_bfs6': 'o', 'demo4_bfs4': '^'}


def file_sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite estimate in sensitivity summary')
    return result


def count(value: str | None) -> int | None:
    if value is None or value == '':
        return None
    result = int(value)
    if result < 0:
        raise ValueError('Negative sensitivity denominator')
    return result


def build_figure_data(pool_rows: list[dict[str, str]], budget_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Select fixed planned comparisons, retaining explicit no-data entries."""
    index: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in pool_rows:
        if (row['scope'] != 'family' or row['source_family'] != 'all' or row['L'] != '32'
                or row['family'] not in FAMILIES or row['controller'] not in CONTROLLERS
                or row['pool_variant'] not in VARIANTS):
            continue
        comparison = (row['reference'], row['comparison'])
        if comparison not in {(ref, other) for ref, other, _ in POOL_COMPARISONS}:
            continue
        key = ('pool', row['family'], row['controller'], row['pool_variant'], *comparison)
        if key in index:
            raise ValueError(f'Duplicate pool summary: {key}')
        index[key] = row
    for row in budget_rows:
        if (row['source_family'] != 'all' or row['family'] not in FAMILIES
                or row['controller'] not in CONTROLLERS or row['pool_variant'] not in VARIANTS):
            continue
        comparison = (int(row['short_L']), int(row['long_L']))
        if comparison not in {(short, long) for short, long, _ in BUDGET_COMPARISONS}:
            continue
        key = ('budget', row['family'], row['controller'], row['pool_variant'], *comparison)
        if key in index:
            raise ValueError(f'Duplicate budget summary: {key}')
        index[key] = row
    result = []
    for family in FAMILIES:
        for panel, comparisons in (('pool', POOL_COMPARISONS), ('budget', BUDGET_COMPARISONS)):
            for first, second, label in comparisons:
                for controller in CONTROLLERS:
                    for variant in VARIANTS:
                        key = (panel, family, controller, variant, first, second)
                        source = index.get(key)
                        mean = number(source['mean_abs_delta_u']) if source else None
                        low = number(source['ci_low']) if source else None
                        high = number(source['ci_high']) if source else None
                        if ((mean is not None and not 0 <= mean <= 1) or (mean is None and low is not None)
                                or (low is None) != (high is None)
                                or (low is not None and not 0 <= low <= high <= 1)):
                            raise ValueError(f'Invalid confidence interval for {key}')
                        planned = count(source['planned_contexts']) if source else None
                        completed = count(source['complete_pool_contexts' if panel == 'pool' else 'complete_contexts']) if source else None
                        contexts = count(source['contexts_with_shared_success']) if source else None
                        targets = count(source['shared_success_targets']) if source else None
                        layouts = count(source['n_layouts' if panel == 'pool' else 'delta_layouts']) if source else None
                        if source and (contexts is None or completed is None or planned is None or targets is None
                                       or layouts is None or not 0 <= contexts <= completed <= planned
                                       or not 0 <= layouts <= contexts or (contexts > 0 and layouts == 0)
                                       or targets < contexts or (mean is not None) != (contexts > 0)):
                            raise ValueError(f'Inconsistent denominators for {key}')
                        result.append({'panel': panel, 'family': family, 'controller': controller,
                            'pool_variant': variant, 'reference': first, 'comparison': second,
                            'comparison_label': label.replace('\n', ' '), 'limit': 32 if panel == 'pool' else None,
                            'planned_contexts': planned, 'complete_contexts': completed,
                            'contexts_with_shared_success': contexts, 'shared_success_targets': targets,
                            'independent_layouts': layouts, 'mean_abs_delta_u': mean, 'ci_low': low, 'ci_high': high,
                            'status': 'available' if mean is not None else 'no_data'})
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_data(path: Path, rows: list[dict[str, Any]]) -> None:
    from io import StringIO

    stream = StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def draw(rows: list[dict[str, Any]], output: Path) -> float:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    values = [value for row in rows for value in (row['mean_abs_delta_u'], row['ci_high']) if value is not None]
    maximum = max(values, default=0.0)
    y_max = min(1.0, max(0.10, math.ceil((maximum + 0.025) * 20) / 20))
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.3), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.22, wspace=0.11, hspace=0.52)
    offsets = {('cllp8', 'demo4_bfs6'): -0.16, ('cllp8', 'demo4_bfs4'): -0.055,
               ('cllp32', 'demo4_bfs6'): 0.055, ('cllp32', 'demo4_bfs4'): 0.16}
    for row_index, family in enumerate(FAMILIES):
        for column, (panel, comparisons) in enumerate((('pool', POOL_COMPARISONS), ('budget', BUDGET_COMPARISONS))):
            ax = axes[row_index, column]
            plotted = 0
            for index, (first, second, _) in enumerate(comparisons):
                subset = [row for row in rows if row['panel'] == panel and row['family'] == family
                          and row['reference'] == first and row['comparison'] == second
                          and row['status'] == 'available']
                if not subset:
                    ax.text(index, y_max * 0.09, 'No data', ha='center', va='bottom', color='#666666', fontsize=7)
                for point in subset:
                    x = index + offsets[(point['controller'], point['pool_variant'])]
                    colour, marker = COLOURS[point['controller']], MARKERS[point['pool_variant']]
                    if point['ci_low'] is not None:
                        ax.vlines(x, point['ci_low'], point['ci_high'], color=colour, linewidth=0.9)
                        ax.hlines([point['ci_low'], point['ci_high']], x - 0.025, x + 0.025,
                                  color=colour, linewidth=0.9)
                    ax.plot(x, point['mean_abs_delta_u'], marker=marker, markersize=4.4,
                            color=colour, linestyle='none', markeredgecolor='white', markeredgewidth=0.35,
                            clip_on=False)
                    plotted += 1
            if not plotted:
                ax.text(0.5, 0.54, 'No data', transform=ax.transAxes, ha='center', va='center', color='#666666')
            else:
                missing = [variant for variant in VARIANTS if not any(
                    row['panel'] == panel and row['family'] == family and row['pool_variant'] == variant
                    and row['status'] == 'available' for row in rows)]
                if missing:
                    variant_labels = ', '.join('4+6' if value == 'demo4_bfs6' else '4+4' for value in missing)
                    ax.text(0.98, 0.96, f'{variant_labels}: No data', transform=ax.transAxes,
                            ha='right', va='top', color='#666666', fontsize=7)
            ax.set_xticks(range(3), [item[2] for item in comparisons])
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(0, y_max)
            ax.grid(axis='y', alpha=0.22, linewidth=0.6)
            ax.set_axisbelow(True)
            family_label = family.replace('x', ' × ')
            ax.set_title(f"{family_label}  ·  " + ('Pool comparison, L = 32' if panel == 'pool' else 'Action limit'))
            if column == 0:
                ax.set_ylabel('Mean |Δu|')
    handles = [Line2D([], [], marker='o', color=COLOURS[model], linestyle='none', label=model.upper())
               for model in CONTROLLERS]
    handles += [Line2D([], [], marker=MARKERS[variant], color='#444444', linestyle='none',
                       label=('Original pool: 4+6' if variant == 'demo4_bfs6' else 'Original pool: 4+4'))
                for variant in VARIANTS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=8.2)
    fig.text(0.5, 0.015,
             'Common successful targets only; equal context weights; bars: 95% layout bootstrap.\n'
             'GEN proposes full boards; CLLP receives full boards. GEN comparison uses the same 4+4 base.\n'
             'Reduced pools use 2+3 or 2+2. Both columns share a data-dependent y-axis limit.',
             ha='center', va='bottom', fontsize=7)
    for extension in ('pdf', 'svg', 'png'):
        target = output / f'pool_budget_sensitivity.{extension}'
        temporary = output / f'.pool_budget_sensitivity.{extension}.{os.getpid()}.tmp'
        try:
            fig.savefig(temporary, format=extension, dpi=200, bbox_inches='tight', pad_inches=0.08)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)
    return y_max


def verified_inputs(analysis: Path, budget: Path) -> tuple[Path, Path, dict[str, str]]:
    sensitivity_path = analysis / 'sensitivity_summary.csv'
    budget_path = budget / 'budget_sensitivity_summary.csv'
    analysis_manifest_path = analysis / 'analysis_manifest.json'
    budget_manifest_path = budget / 'budget_sensitivity_manifest.json'
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding='utf-8'))
    budget_manifest = json.loads(budget_manifest_path.read_text(encoding='utf-8'))
    output_hashes = analysis_manifest['output_sha256']
    if file_sha(sensitivity_path) != output_hashes['sensitivity_summary.csv']:
        raise ValueError('Pool sensitivity CSV differs from the completed analysis manifest')
    profiles_hash = file_sha(analysis / 'profiles.jsonl')
    if profiles_hash != output_hashes['profiles.jsonl'] or profiles_hash != budget_manifest['profiles_sha256']:
        raise ValueError('The budget and pool summaries do not share the same frozen profiles')
    paths = (sensitivity_path, budget_path, analysis_manifest_path, budget_manifest_path)
    return sensitivity_path, budget_path, {str(path.resolve()): file_sha(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Completed profile analysis directory')
    parser.add_argument('--budget', type=Path, required=True, help='Completed budget sensitivity directory')
    parser.add_argument('--output', type=Path, required=True, help='Figure output directory')
    args = parser.parse_args()
    sensitivity, budget, inputs = verified_inputs(args.analysis, args.budget)
    data = build_figure_data(read_csv(sensitivity), read_csv(budget))
    args.output.mkdir(parents=True, exist_ok=True)
    y_max = draw(data, args.output)
    for row in data:
        row['figure_ymax'] = y_max
    write_data(args.output / 'figure_data.csv', data)
    names = ('pool_budget_sensitivity.pdf', 'pool_budget_sensitivity.svg', 'pool_budget_sensitivity.png', 'figure_data.csv')
    manifest = {'schema_version': 1, 'command': list(os.sys.argv), 'input_sha256': inputs,
                'plot_source_sha256': file_sha(Path(__file__)),
                'outputs_sha256': {name: file_sha(args.output / name) for name in names},
                'population': 'shared successful targets within complete paired profiles',
                'aggregation': 'equal context weight; supplied 95% bootstrap intervals resample canonical layouts',
                'controller_input': 'current and complete target board', 'generator_output': 'complete board',
                'fixed_pool_variants': list(VARIANTS), 'generator_comparison': 'same four demo and four BFS targets, with or without one generated target',
                'budget_comparison': 'same fixed pool at each pair of action limits',
                'relative_scale_delta': 'absolute difference in the success-weighted midpoint ECDF u',
                'meanings': {'planned_contexts': 'all contexts assigned this original fixed-pool variant',
                             'complete_contexts': 'contexts where both compared profiles are complete',
                             'contexts_with_shared_success': 'complete contexts with at least one target reached under both settings',
                             'shared_success_targets': 'number of shared reached targets across complete contexts',
                             'independent_layouts': 'distinct D4 layouts contributing to the estimate'},
                'y_axis': {'minimum': 0, 'maximum': y_max, 'maximum_chosen_from_data': True}}
    atomic_text(args.output / 'pool_budget_sensitivity_manifest.json',
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output.resolve()), 'plot_rows': len(data),
                      'available': sum(row['status'] == 'available' for row in data), 'y_max': y_max}))


if __name__ == '__main__':
    main()

"""Render offline campaign CSVs as Plotly HTML and vector figures."""
import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.plot_data_export.common import OUTPUT

COLORS = {'relative_u': '#2857a4', 'absolute_d': '#d26a24'}


def load(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / f'{name}.csv')


def solve_rate(directory: Path) -> go.Figure:
    """Show only measured per-budget runs and board-paired differences."""
    rates = load(directory, 'solve_rate_by_budget')
    pairs = load(directory, 'paired_deltas')
    figure = make_subplots(rows=1, cols=3, subplot_titles=('Real 12×12×4', 'Generated 32×32×10', 'Paired Δ (percentage points)'))
    for column, family in enumerate(('real_12x12x4', 'generated_32x32x10'), start=1):
        family_rows = cast(pd.DataFrame, rates.loc[rates.family == family])
        if family_rows.empty:
            figure.add_trace(go.Scatter(x=[128, 512], y=[None, None], showlegend=False,
                                        hoverinfo='skip'), row=1, col=column)
            figure.add_annotation(text='Outside current pilot scope<br>No full-solver pilot observations', x=0.5, y=0.5,
                                  xref='paper', yref='paper', showarrow=False)
        for method in ('relative_u', 'absolute_d'):
            rows = cast(pd.DataFrame, family_rows.loc[(family_rows.method == method) & family_rows.solve_rate.notna()]).sort_values(by='budget_actions')
            if rows.empty:
                continue
            upper = (rows.wilson_high - rows.solve_rate).clip(lower=0) * 100
            lower = (rows.solve_rate - rows.wilson_low).clip(lower=0) * 100
            figure.add_trace(go.Scatter(x=rows.budget_actions, y=rows.solve_rate * 100, name=method,
                mode='markers', marker={'size': 15 if method == 'relative_u' else 9,
                    'symbol': 'circle-open' if method == 'relative_u' else 'diamond'},
                line={'color': COLORS[method]}, error_y={'type': 'data', 'symmetric': False, 'array': upper, 'arrayminus': lower,
                    'thickness': 5 if method == 'relative_u' else 2},
                customdata=np.stack([rows.denominator_boards, rows.expected_boards, rows.verified_successes], axis=-1),
                hovertemplate='B=%{x}<br>solve rate=%{y:.1f}%<br>successes=%{customdata[2]}<br>N=%{customdata[0]}/%{customdata[1]}<extra>%{fullData.name}</extra>'), row=1, col=column)
        if not family_rows.empty:
            observed = cast(pd.DataFrame, family_rows.loc[(family_rows.budget_actions == 256) & (family_rows.method == 'relative_u')])
            if not observed.empty:
                item = observed.iloc[0]
                figure.add_annotation(text=f"B*=256: {int(item.verified_successes)}/{int(item.denominator_boards)}; "
                                           f"Wilson upper={100 * item.wilson_high:.1f}%", x=0.02, y=0.93,
                                      xref='paper', yref='paper', showarrow=False, xanchor='left')
    rows = cast(pd.DataFrame, pairs.loc[pairs.delta_relative_minus_absolute.notna()]).sort_values(by='budget_actions')
    if not rows.empty:
        figure.add_trace(go.Scatter(x=rows.budget_actions, y=100 * rows.delta_relative_minus_absolute,
            mode='lines+markers', name='relative − absolute', line={'color': '#5c4184'},
            customdata=np.stack([rows.complete_pairs, rows.expected_pairs, rows.full_cohort_delta_lower,
                                 rows.full_cohort_delta_upper], axis=-1),
            hovertemplate='B=%{x}<br>Δ=%{y:.1f} pp<br>pairs=%{customdata[0]}/%{customdata[1]}'
                          '<br>full cohort bounds=[%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>'), row=1, col=3)
    if not rows.empty:
        point = rows.iloc[0]
        figure.add_annotation(text=f'B={int(point.budget_actions)}: {int(point.complete_pairs)}/{int(point.expected_pairs)} paired boards'
                              f'<br>Full-cohort Δ bounds: {100 * point.full_cohort_delta_lower:g} to '
                              f'{100 * point.full_cohort_delta_upper:g} pp',
                              x=0.84, y=0.93, xref='paper', yref='paper', showarrow=False)
    figure.update_yaxes(title_text='Solve rate (%)', range=[-2, 103], row=1, col=1)
    figure.update_yaxes(title_text='Solve rate (%)', range=[-2, 103], row=1, col=2)
    figure.update_yaxes(title_text='Δ relative − absolute (pp)', row=1, col=3)
    figure.update_xaxes(title_text='Action budget B', tickmode='array', tickvals=[128, 256, 512], range=[100, 540])
    figure.update_layout(title='Independent action-budget runs · Wilson 95% intervals · partial pilot', height=520)
    return figure


def board_heatmap(rows: list[str]) -> go.Heatmap:
    symbols = {'#': 0, '-': 1, '.': 2, '$': 3, '*': 4, '@': 5, '+': 6}
    palette = ['#293442', '#f4ead9', '#8bc2a2', '#bb8054', '#e2ab51', '#e96056', '#e96056']
    colorscale = [[i / 6, color] for i, color in enumerate(palette)]
    return go.Heatmap(z=[[symbols[ch] for ch in row] for row in rows], zmin=0, zmax=6, colorscale=colorscale,
                      text=[list(row) for row in rows], texttemplate='%{text}', showscale=False,
                      hovertemplate='row=%{y}, col=%{x}<br>%{text}<extra></extra>')


def calibration_figure(directory: Path) -> go.Figure:
    """Show training boards, target-wise p-hat and weighted time ECDFs."""
    targets = load(directory, 'figure1_targets')
    ecdf = load(directory, 'figure1_ecdf')
    quantiles = load(directory, 'figure1_quantiles')
    contexts = targets.context_id.drop_duplicates().tolist()
    titles = [title for context in contexts for title in (f'{context}: training board', 'Target p̂ by offset k', 'Weighted ECDF of d')]
    figure = make_subplots(rows=len(contexts), cols=3, subplot_titles=titles, horizontal_spacing=0.09)
    for index, context in enumerate(contexts, start=1):
        rows = cast(pd.DataFrame, targets.loc[targets.context_id == context]).sort_values(by='k')
        figure.add_trace(board_heatmap(json.loads(rows.iloc[0]['state'])), row=index, col=1)
        figure.update_yaxes(autorange='reversed', row=index, col=1)
        figure.add_trace(go.Scatter(x=rows.k, y=rows.p_hat, mode='markers+lines', name=context,
            customdata=rows[['target_id', 'R_attempted', 'hits']].to_numpy(), showlegend=False,
            hovertemplate='k=%{x}<br>p̂=%{y:.3f}<br>target=%{customdata[0]}<br>hits=%{customdata[2]}/%{customdata[1]}<extra></extra>'), row=index, col=2)
        distribution = cast(pd.DataFrame, ecdf.loc[ecdf.context_id == context]).sort_values(by='d')
        start = float(distribution.d.min()) - 0.2
        figure.add_trace(go.Scatter(x=[start, *distribution.d], y=[0, *distribution.F_right], mode='lines+markers',
                                    line={'shape': 'hv'}, showlegend=False, name='F(d)'), row=index, col=3)
        for _, quantile in quantiles[quantiles.context_id == context].iterrows():
            if pd.notna(quantile.quantile_d):
                figure.add_trace(go.Scatter(x=[quantile.quantile_d], y=[quantile.percentile],
                    mode='markers+text', text=[f"f={quantile.percentile:g}, d={quantile.quantile_d:g}"],
                    textposition='top left' if quantile.quantile_d == distribution.d.max() else 'top right',
                    showlegend=False), row=index, col=3)
        figure.update_xaxes(title_text='Trajectory offset k (actions)', row=index, col=2)
        figure.update_yaxes(title_text='P(exact hit within L)', range=[-0.05, 1.05], row=index, col=2)
        figure.update_xaxes(title_text='Mean successful first-hit time d (actions)', row=index, col=3)
        figure.update_yaxes(title_text='Weighted F(d)', range=[-0.05, 1.05], row=index, col=3)
    figure.update_layout(title='Empirical Figure 1 · training contexts · quantile = min d with F(d) ≥ f', height=340 * len(contexts))
    return figure


def component_figure(directory: Path) -> go.Figure:
    policy = load(directory, 'component_policy')
    generator = load(directory, 'component_generator')
    value = load(directory, 'component_value')
    figure = make_subplots(rows=1, cols=3, subplot_titles=('CLLP exact hit', 'Generator: reached distinct goal', 'V along trajectories (8 boards)'))
    for variant, rows in policy.groupby('variant', sort=True):
        stats = rows.groupby('k').exact_hit.agg(['sum', 'count']).reset_index()
        figure.add_trace(go.Scatter(x=stats.k, y=stats['sum'] / stats['count'], mode='lines+markers+text',
            text=[f'{int(a)}/{int(b)}' for a, b in zip(stats['sum'], stats['count'], strict=True)],
            textposition='top center', name=f'policy: {variant}'), row=1, col=1)
    for variant, rows in generator.groupby('variant', sort=True):
        stats = rows.groupby('k').reached_any.agg(['sum', 'count']).reset_index()
        figure.add_trace(go.Scatter(x=stats.k, y=stats['sum'] / stats['count'], mode='lines+markers+text',
            text=[f'{int(a)}/{int(b)}' for a, b in zip(stats['sum'], stats['count'], strict=True)],
            textposition='bottom center', name=f'generator: {variant}'), row=1, col=2)
    selected = cast(pd.DataFrame, value.loc[(value.variant == 'current_frozen') & value.predicted_value.notna()])
    for board, rows in selected.groupby('board_id', sort=True):
        rows = cast(pd.DataFrame, rows).sort_values(by='position')
        fraction = rows.position / (rows.position + rows.remaining).replace(0, np.nan)
        figure.add_trace(go.Scatter(x=fraction, y=rows.predicted_value, mode='lines+markers',
            name=board, legendgroup='value', showlegend=False,
            customdata=rows[['position', 'remaining', 'reference_value']].to_numpy(),
            hovertemplate='trajectory fraction=%{x:.2f}<br>V=%{y:.3f}<br>remaining=%{customdata[1]}'
                          '<br>reference=%{customdata[2]:.3f}<extra>%{fullData.name}</extra>'), row=1, col=3)
    figure.update_xaxes(title_text='Trajectory offset k', tickmode='array', tickvals=[1, 2, 4, 8], row=1, col=1)
    figure.update_xaxes(title_text='Requested k', row=1, col=2)
    figure.update_xaxes(title_text='Position / trajectory length', row=1, col=3)
    figure.update_yaxes(title_text='Exact-hit fraction', range=[-0.08, 1.15], row=1, col=1)
    figure.update_yaxes(title_text='Requests with reached distinct goal', range=[-0.08, 1.15], row=1, col=2)
    figure.update_yaxes(title_text='Predicted V', row=1, col=3)
    figure.update_layout(title='Component validation · labels show hits / requests · not full-board solve rate', height=520)
    return figure


def cost_figure(directory: Path) -> go.Figure:
    runs = load(directory, 'runs')
    rows = cast(pd.DataFrame, runs.loc[runs.execution_status == 'complete']).copy()
    figure = make_subplots(rows=1, cols=4, subplot_titles=('Actions', 'Provider tokens', 'Requests', 'Stop reasons'))
    for method, group in rows.groupby('method', sort=True):
        color = COLORS.get(str(method), '#555555')
        figure.add_trace(go.Box(y=group.attempted_actions, name=method, marker_color=color, legendgroup=method), row=1, col=1)
        figure.add_trace(go.Box(y=cast(pd.DataFrame, group[['input_tokens', 'output_tokens']]).sum(axis=1, min_count=2),
                               name=method, marker_color=color, legendgroup=method, showlegend=False), row=1, col=2)
        figure.add_trace(go.Box(y=group.provider_requests, name=method, marker_color=color,
                               legendgroup=method, showlegend=False), row=1, col=3)
        observed = runs[(runs.method == method) & runs.execution_status.isin(['complete', 'failed', 'interrupted'])]
        counts = observed.completion_class.fillna('interrupted_or_unknown').value_counts()
        figure.add_trace(go.Bar(x=counts.index, y=counts.values, name=method, marker_color=color,
                                text=[f'{100 * count / len(observed):.1f}%' for count in counts.values],
                                textposition='outside', legendgroup=method, showlegend=False), row=1, col=4)
    figure.update_yaxes(title_text='Executed actions / run', row=1, col=1)
    figure.update_yaxes(title_text='Provider input + output tokens / run', row=1, col=2)
    figure.update_yaxes(title_text='Physical requests / run', row=1, col=3)
    figure.update_yaxes(title_text='Observed runs (labels: share per method)', row=1, col=4)
    costs = runs.cost_settled_usd.sum()
    uncertain = runs.cost_unknown_reserved_usd.sum()
    figure.update_layout(title=f'Resource use · settled pilot USD {costs:.3f}, unknown reserved USD {uncertain:.3f} · mixed cache',
                         barmode='group', height=520)
    return figure


def make_figures(directory: Path) -> dict[str, go.Figure]:
    return {'A_solve_rate': solve_rate(directory), 'B_calibration': calibration_figure(directory),
            'C_components': component_figure(directory), 'D_cost_stops': cost_figure(directory)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'figures')
    parser.add_argument('--formats', nargs='+', choices=('html', 'svg', 'pdf'), default=['html', 'svg', 'pdf'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, figure in make_figures(args.data).items():
        for fmt in args.formats:
            target = args.output / f'{name}.{fmt}'
            if fmt == 'html':
                figure.write_html(target, include_plotlyjs=True, full_html=True)
            else:
                figure.write_image(target, width=1500, height=figure.to_dict()['layout'].get('height', 600))
            print(target)


if __name__ == '__main__':
    main()

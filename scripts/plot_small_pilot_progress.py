"""Plot paired solve rates from aggregate pilot checkpoints without opening boards."""
import argparse
import csv
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from carl.icl.config import ROOT

DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/pilot_small_v1'
METHODS = ('relative_u', 'absolute_d')


def figure(rows: list[dict[str, str]]) -> go.Figure:
    """Show only matched uncensored boards; open markers indicate incomplete coverage."""
    plot = go.Figure()
    for method in METHODS:
        selected = sorted((row for row in rows if row['method'] == method), key=lambda row: int(row['budget']))
        x, y, symbols, details = [], [], [], []
        for row in selected:
            expected = int(row['expected'])
            paired = int(row['paired_uncensored_boards'])
            solved = int(row['paired_solved'])
            if min(expected, paired, solved) < 0 or not solved <= paired <= expected:
                raise ValueError('Invalid paired solve-rate denominator or numerator')
            x.append(int(row['budget']))
            y.append(100 * solved / paired if paired else None)
            symbols.append('circle' if paired == expected else 'circle-open')
            details.append([solved, paired, expected, int(row['failed']), int(row['missing']), int(row['censored'])])
        plot.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=method, connectgaps=False,
                                 marker={'size': 11, 'symbol': symbols}, customdata=details,
                                 hovertemplate=('Budget %{x}<br>Paired solve rate %{y:.1f}%<br>'
                                                'Solved %{customdata[0]} / %{customdata[1]} paired boards'
                                                '<br>Expected %{customdata[2]}<br>Failed %{customdata[3]}'
                                                '<br>Missing %{customdata[4]}<br>Censored %{customdata[5]}<extra>%{fullData.name}</extra>')))
    plot.update_layout(title='Paired solve rate by independent action budget', xaxis_title='Action budget',
                       yaxis_title='Solved among paired, uncensored boards (%)',
                       yaxis={'range': [0, 100]}, template='plotly_white',
                       annotations=[{'text': 'Open markers: incomplete paired coverage; missing budgets are not zero.',
                                     'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': -0.2, 'showarrow': False}])
    return plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUTPUT / 'summary.csv')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'paired_solve_rate.html')
    args = parser.parse_args()
    with args.input.open(encoding='utf-8', newline='') as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure(rows).write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f'Saved paired solve-rate plot: {args.output}')


if __name__ == '__main__':
    main()

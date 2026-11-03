"""Visualize calibration reachability and weighted local percentiles offline."""
import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.config import ROOT, load_config
from carl.icl.experiment import calibration_components, calibration_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/icl/campaign_data/calibration_live_v2.json')
    parser.add_argument('--provider', choices=('mock', 'openai'), default='openai')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    components = calibration_components(config, 'mock' if args.provider == 'mock' else 'replay')
    directory = calibration_path(config, components).parent
    summary = json.loads((directory / 'calibration.json').read_text())
    rows = json.loads((directory / 'calibration_targets.json').read_text())
    if summary['status'] != 'completed' or summary['provider'] != args.provider:
        raise ValueError('Expected a completed calibration from the requested provider')
    contexts = [row['context'] for row in summary['contexts']]
    offsets = sorted({row['k'] for row in rows})
    by_key = {(row['context'], row['k']): row for row in rows}
    z = [[by_key[(context, k)]['p_hat'] if (context, k) in by_key else None for k in offsets] for context in contexts]
    text = [[f"hits={by_key[(context, k)]['hits']}/{by_key[(context, k)]['trials']}"
             if (context, k) in by_key else 'not sampled' for k in offsets] for context in contexts]
    figure = make_subplots(rows=1, cols=2, subplot_titles=('Exact-hit rate by sampled target', 'Weighted local percentile'))
    figure.add_trace(go.Heatmap(x=offsets, y=contexts, z=z, text=text, hovertemplate='k=%{x}<br>%{y}<br>%{text}<extra></extra>',
                                zmin=0, zmax=1, colorbar={'title': 'p̂'}, colorscale='Blues'), row=1, col=1)
    for context in contexts:
        selected = [row for row in rows if row['context'] == context and row['hits'] > 0]
        if selected:
            figure.add_trace(go.Scatter(x=[row['d'] for row in selected], y=[row['u'] for row in selected],
                                        text=[f"k={row['k']}, p̂={row['p_hat']:.3f}, weight={row['weight']:.3f}"
                                              for row in selected],
                                        mode='markers', marker={'size': [8 + 18 * row['weight'] for row in selected]},
                                        name=context, hovertemplate='%{text}<br>d=%{x}<br>u=%{y}<extra></extra>'), row=1, col=2)
    figure.update_xaxes(title_text='Trajectory offset k', row=1, col=1)
    figure.update_xaxes(title_text='Mean successful first-hit time d', row=1, col=2)
    figure.update_yaxes(title_text='Local percentile u', range=[0, 1], row=1, col=2)
    figure.update_layout(title='Training calibration: sampled targets, exact hits and local weighted percentiles',
                         template='plotly_white', height=650)
    output = args.output or ROOT / 'analysis/luna_icl/campaign_100/calibration_v2/coverage.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True)
    print(f'Plot: {output}')


if __name__ == '__main__':
    main()

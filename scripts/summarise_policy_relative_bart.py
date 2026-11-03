"""Export a readable report and Plotly learning curves for one completed run."""
import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    folder = parser.parse_args().run_directory
    report = json.loads((folder / 'summary.json').read_text())
    if report['config']['initialization'] != 'scratch':
        raise ValueError('This report is for scratch runs only')
    curves = pd.read_csv(folder / 'learning_curves.csv')
    figure = go.Figure()
    colors = {'relative': '#2452a6', 'absolute': '#a5363c', 'none': '#55616f'}
    rows = []
    for run in report['runs']:
        condition = run['condition']
        history = curves[curves.run == run['final']['run']]
        for split, dash in [('train', 'dot'), ('val', 'solid')]:
            figure.add_trace(go.Scatter(
                x=history.step, y=history[f'{split}_loss'], mode='lines+markers',
                name=f'{condition}: {split}',
                line=dict(color=colors[condition], dash=dash, width=2.5)))
        first, final, generated = run['initial'], run['final'], run['generation']
        rows.append(dict(
            condition=condition, train_start=first['train_loss'], train_end=final['train_loss'],
            train_change=final['train_loss']-first['train_loss'],
            validation_start=first['val_loss'], validation_end=final['val_loss'],
            validation_change=final['val_loss']-first['val_loss'],
            test_loss=final['test_loss'],
            valid_noncopy=generated['valid_noncopy'], requests=generated['requests'],
            controller_hits=generated['successful_noncopy_rollouts'],
            controller_trials=generated['rollout_trials']))
    figure.update_layout(
        template='plotly_white', width=1050, height=490,
        title='Small BART from scratch', xaxis_title='Optimiser updates',
        yaxis_title='Token cross-entropy', font=dict(size=17),
        legend=dict(orientation='h', y=1.12), margin=dict(l=80, r=30, t=115, b=70))
    figure.write_html(folder / 'learning_curves.html', include_plotlyjs=True)
    for extension in ['svg', 'pdf', 'png']:
        figure.write_image(folder / f'learning_curves.{extension}', scale=2 if extension=='png' else 1)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'loss_changes.csv', index=False)
    human = pd.DataFrame([dict(
        condition=row['condition'],
        train_CE=f"{row['train_start']:.4f} → {row['train_end']:.4f}",
        validation_CE=f"{row['validation_start']:.4f} → {row['validation_end']:.4f}",
        validation_change=f"{row['validation_change']:+.4f}",
        valid_noncopy=f"{row['valid_noncopy']}/{row['requests']}",
        controller_hits=f"{row['controller_hits']}/{row['controller_trials']}") for row in rows])
    text = [
        '# Small BART from scratch', '',
        f"Seed: {report['seed']}. Updates per condition: {report['config']['steps']}. "
        f"Batch: {report['config']['batch']}. Learning rate: {report['config']['lr']}.", '',
        'Generator weights are randomly initialised; the existing policy is frozen. '
        'All conditions reuse the same calibrated examples, initialisation and sampling seed.', '',
        human.to_markdown(index=False, disable_numparse=True), '',
        'Negative loss changes indicate a decrease. Teacher-forced loss does not establish '
        'valid or reachable generated boards. Zero controller trials means there was no '
        'valid non-copy target to execute.', '',
        '[Interactive curves](learning_curves.html) · [PDF](learning_curves.pdf) · '
        '[CSV changes](loss_changes.csv) · [Console log](console.log)', '',
        'This is the 750-example pilot, not full-data training or a completed latent-policy implementation.',
    ]
    (folder / 'summary.md').write_text('\n'.join(text)+'\n')
    print(frame.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

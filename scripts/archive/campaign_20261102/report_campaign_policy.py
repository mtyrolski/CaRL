"""Render a policy validation report and Plotly chart without further model calls."""
from __future__ import annotations

import json
import numpy as np
import plotly.graph_objects as go

from carl.icl.boards import from_rows, tiles
from carl.icl.datasets import verify_solution
from scripts.archive.campaign_20261102.campaign_policy_validation import ACTIONS, OFFSETS, OUTPUT, PRIVATE, ROOT, VARIANTS


def main() -> None:
    report = json.loads((OUTPUT / 'report.json').read_text())
    diagnostic_path = OUTPUT / 'simulator_successors.json'
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    lines = ['# Walidacja CLLP: dokładne trafienie stanu', '',
             'Cztery świeże plansze i te same 16 par dla wariantów. Każda para ma jedną próbę; '
             'L=8, rozkłady są próbkowane kategorycznie. To mały eksperyment walidacyjny, nie solve-rate.', '',
             '| Policy | k=1 | k=2 | k=4 | k=8 | Błędy |', '|---|---:|---:|---:|---:|---:|']
    figure = go.Figure()
    for variant in VARIANTS:
        by_k = report['summary'][variant]['by_offset']
        cols = [f"{by_k[str(k)]['hits']}/{by_k[str(k)]['cases']}" for k in OFFSETS]
        errors = sum(by_k[str(k)]['errors'] for k in OFFSETS)
        lines.append(f"| {variant} | {' | '.join(cols)} | {errors} |")
        figure.add_trace(go.Scatter(x=list(OFFSETS), y=[by_k[str(k)]['hits'] / by_k[str(k)]['cases']
                              if by_k[str(k)]['cases'] else None for k in OFFSETS],
                              text=cols, name=variant, mode='lines+markers+text', textposition='top center'))
    if diagnostic is not None:
        cols = [f"{diagnostic['summary'][str(k)]['hits']}/{diagnostic['summary'][str(k)]['cases']}" for k in OFFSETS]
        errors = sum(diagnostic['summary'][str(k)]['errors'] for k in OFFSETS)
        lines += ['', '## Osobna diagnostyka z symulatorem', '',
                  'Ten wariant dostaje cztery dodatkowe przejścia symulatora na każdą decyzję. '
                  'Nie jest częścią wyboru policy. W odróżnieniu od historycznego explore.py '
                  'wzbogacone są tylko bieżące zapytania, nie demonstracje.', '',
                  '| Policy | k=1 | k=2 | k=4 | k=8 | Błędy |', '|---|---:|---:|---:|---:|---:|',
                  f"| simulator_successors | {' | '.join(cols)} | {errors} |"]
    lines += ['', '## Przykład nietrafionego wielokrokowego rollouta', '']
    failed = next((row for row in report['cases'] if row['offset'] >= 4 and not row['exact_hit']
                   and row['error'] is None), None)
    if failed is not None:
        public = json.loads((ROOT / 'real_12x12x4/component_validation.json').read_text())['instances']
        witness = {item['board_id']: item['actions'] for item in json.loads(PRIVATE.read_text())['witnesses']}
        initial = next(item['state'] for item in public if item['id'] == failed['board_id'])
        actions = witness[failed['board_id']]
        states = verify_solution(from_rows(initial), actions)
        pos, offset = failed['position'], failed['offset']
        segment = actions[pos:pos + offset]
        pushes = [not np.array_equal(np.isin(tiles(states[pos + i]), [3, 4]),
                                     np.isin(tiles(states[pos + i + 1]), [3, 4]))
                  for i in range(offset)]
        lines += [f"Policy `{failed['variant']}`, plansza `{failed['board_id']}`, k={offset}, "
                  f"seed={failed['sampling_seed']}, L={failed['cap']}.", '',
                  'Odtworzony świadek (jedna poprawna droga, niekoniecznie najkrótsza): '
                  + ', '.join(f'{ACTIONS[a]}{" (push)" if push else ""}' for a, push in zip(segment, pushes, strict=True)) + '.', '',
                  'Próba policy: ' + ', '.join(step['action'] for step in failed['steps']) + '.', '',
                  f"No-op: {failed['no_ops']}; powtórzone stany: {failed['repeated_states']}; "
                  f"wcześniej rozwiązane zadanie: {failed['task_solved_early']}.", '',
                  'Inna poprawna droga do identycznego celu też jest liczona jako trafienie. '
                  'Pełne rozkłady i stany są w `report.json`.']
    else:
        lines.append('Brak ukończonego nietrafionego wielokrokowego rollouta.')
    figure.update_layout(title='CLLP: exact target hit / offset', xaxis_title='Witness offset k',
                         yaxis_title='Exact-hit rate', yaxis_range=[0, 1.1], template='plotly_white')
    figure.update_xaxes(tickmode='array', tickvals=list(OFFSETS))
    figure.write_html(str(OUTPUT / 'exact_hit.html'), include_plotlyjs=True)
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUTPUT / 'report.md')


if __name__ == '__main__':
    main()

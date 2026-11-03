"""Summarize matched generator and value validation without additional inference."""
import csv
import json
from collections import Counter
from itertools import combinations
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr

from carl.icl.api import write_json
from scripts.campaign_components_validation import GENERATOR, OUTPUT, VALUES

VALUE_VARIANTS = (*VALUES, 'current_frozen')


def rank_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute board-weighted rankings with ties and missing predictions explicit."""
    boards = sorted({row['board_id'] for row in rows})
    board_metrics = []
    for board in boards:
        selected = sorted((row for row in rows if row['board_id'] == board), key=lambda row: row['position'])
        valid = [row for row in selected if row['predicted_value'] is not None]
        predicted = [row['predicted_value'] for row in valid]
        reference = [row['reference_value'] for row in valid]
        rho = None
        if len(valid) >= 2 and len(set(predicted)) > 1 and len(set(reference)) > 1:
            value = float(cast(Any, spearmanr(predicted, reference)).statistic)
            rho = value if np.isfinite(value) else None
        nonterminal_valid = [row for row in valid if row['remaining'] > 0]
        nonterminal_predicted = [row['predicted_value'] for row in nonterminal_valid]
        nonterminal_reference = [row['reference_value'] for row in nonterminal_valid]
        nonterminal_rho = None
        if (len(nonterminal_valid) >= 2 and len(set(nonterminal_predicted)) > 1
                and len(set(nonterminal_reference)) > 1):
            rank = float(cast(Any, spearmanr(nonterminal_predicted, nonterminal_reference)).statistic)
            nonterminal_rho = rank if np.isfinite(rank) else None
        info: dict[str, Any] = {'board_id': board, 'family': selected[0]['family'], 'valid': len(valid),
                                'expected': len(selected), 'spearman': rho, 'nonterminal_spearman': nonterminal_rho,
                                'flat_prediction': len(valid) >= 2 and len(set(predicted)) == 1,
                                'flat_nonterminal_prediction': len(nonterminal_valid) >= 2 and len(set(nonterminal_predicted)) == 1}
        for name, relevant in [('all', selected), ('nonterminal', [row for row in selected if row['remaining'] > 0])]:
            concordant = discordant = ties = missing = 0
            for a, b in combinations(relevant, 2):
                if a['predicted_value'] is None or b['predicted_value'] is None:
                    missing += 1
                    continue
                difference = (a['predicted_value'] - b['predicted_value']) * (a['reference_value'] - b['reference_value'])
                concordant += difference > 0
                discordant += difference < 0
                ties += difference == 0
            expected = len(relevant) * (len(relevant) - 1) // 2
            info[name] = {'concordant': concordant, 'discordant': discordant, 'ties': ties,
                          'missing': missing, 'expected': expected,
                          'concordance_including_ties_and_missing': concordant / expected if expected else None,
                          'concordance_among_observed': concordant / (expected - missing) if expected > missing else None}
        board_metrics.append(info)
    scores = [item['nonterminal']['concordance_including_ties_and_missing'] for item in board_metrics]
    valid_rows = [row for row in rows if row['predicted_value'] is not None]
    rhos = [item['spearman'] for item in board_metrics if item['spearman'] is not None]
    nonterminal_rhos = [item['nonterminal_spearman'] for item in board_metrics if item['nonterminal_spearman'] is not None]
    return {'boards': board_metrics, 'observed_values': len(valid_rows), 'expected_values': len(rows),
            'errors': sum(row['error'] is not None for row in rows),
            'mae': float(np.mean([abs(row['predicted_value'] - row['reference_value']) for row in valid_rows])) if valid_rows else None,
            'mean_spearman_defined': float(np.mean(rhos)) if rhos else None, 'spearman_defined_boards': len(rhos),
            'flat_boards': sum(item['flat_prediction'] for item in board_metrics),
            'flat_nonterminal_boards': sum(item['flat_nonterminal_prediction'] for item in board_metrics),
            'nonterminal_mean_spearman_defined': float(np.mean(nonterminal_rhos)) if nonterminal_rhos else None,
            'nonterminal_spearman_defined_boards': len(nonterminal_rhos),
            'nonterminal_board_mean': float(np.mean(scores)) if scores and all(x is not None for x in scores) else None,
            'pair_counts': {name: {key: sum(item[name][key] for item in board_metrics)
                                   for key in ('concordant', 'discordant', 'ties', 'missing', 'expected')}
                            for name in ('all', 'nonterminal')}}


def main() -> None:
    root = OUTPUT / 'openai'
    generator_rows = [json.loads(path.read_text()) for path in sorted(root.glob('generator_*.json'))]
    value_rows = [json.loads(path.read_text()) for path in sorted(root.glob('value_*.json'))]
    if any(sum(row['variant'] == name for row in generator_rows) != 8 for name in GENERATOR):
        raise ValueError('Generator matrix is incomplete; do not select or freeze')
    if any(sum(row['variant'] == name for row in value_rows) != 40 for name in VALUE_VARIANTS):
        raise ValueError('Value matrix is incomplete; do not select or freeze')
    generator = {}
    for name in GENERATOR:
        rows = [row for row in generator_rows if row['variant'] == name]
        proposals = [item for row in rows for item in row['proposals']]
        bfs = Counter(item['bfs']['status'] for item in proposals)
        candidates = [item for row in rows for item in row['candidates']]
        generator[name] = {'requests': len(rows), 'reached_any': sum(row['reached_any'] for row in rows),
                           'parse_ok': sum(row['parse_ok'] for row in rows),
                           'api_or_parse_errors': sum(row['error'] is not None for row in rows),
                           'candidate_count': len(candidates),
                           'static_valid_candidates': sum(item['static_valid'] for item in candidates),
                           'static_invalid_candidates': sum(not item['static_valid'] for item in candidates),
                           'source_copies': sum(item['source_copy'] for item in candidates),
                           'duplicates': sum(item['duplicate'] for item in candidates),
                           'distinct_valid': sum(item['distinct_valid'] for item in candidates),
                           'bfs': {status: bfs[status] for status in ('reachable', 'unreachable_within_cap', 'node_limit')}, 'rollouts': len(proposals),
                           'rollout_errors': sum(item['rollout']['error'] is not None for item in proposals)}
    values = {name: rank_metrics([row for row in value_rows if row['variant'] == name]) for name in VALUE_VARIANTS}
    if any(values[name]['observed_values'] == 0 for name in VALUE_VARIANTS):
        raise ValueError('Cannot compare an entirely missing value variant')
    gen_winner = min(GENERATOR, key=lambda name: (-generator[name]['reached_any'],
                                                     -generator[name]['distinct_valid'], name))
    value_winner = min(VALUE_VARIANTS, key=lambda name: (-values[name]['nonterminal_board_mean'],
                                                values[name]['mae'] if values[name]['mae'] is not None else float('inf'), name))
    summary = {'generator': generator, 'value': values, 'generator_winner': gen_winner, 'value_winner': value_winner,
               'generator_selection': 'query-level reached-any / all 8 requests; more distinct valid proposals breaks ties, then alphabetical',
               'value_selection': 'board mean of nonterminal concordant / all expected pairs; MAE tie-break',
               'notes': ['Ties and missing pairs count against selection score and are separately reported.',
                         'BFS is bounded diagnostic only and is never input to Luna.',
                         'Reference is one verified continuation, not an optimal solution.']}
    write_json(OUTPUT / 'summary.json', summary)
    with (OUTPUT / 'value_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['variant', 'board_id', 'family', 'position', 'remaining', 'gamma', 'reference_value', 'predicted_value', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(value_rows)
    lines = ['# Generator i V: walidacja komponentów', '',
             'Generator: 8 sparowanych żądań na czterech realnych planszach, k=4/8, dwie propozycje na żądanie. '
             'CLLP ma L=8, a BFS jest wyłącznie offline. W mianowniku są także błędne i puste odpowiedzi.', '',
             '| Generator | reached-any | parsowanie | statycznie poprawne | kopie | duplikaty | poprawne odmienne | BFS |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, row in generator.items():
        lines.append(f"| {name} | {row['reached_any']}/{row['requests']} | {row['parse_ok']}/{row['requests']} | "
                     f"{row['static_valid_candidates']} | {row['source_copies']} | {row['duplicates']} | "
                     f"{row['distinct_valid']} | {row['bfs']} |")
    lines += ['', 'V: 8 plansz (4 realne i 4 syntetyczne), po 5 pozycji, porównanie tych samych stanów. '
              'Ranking referencyjny pochodzi z jednej poprawnej kontynuacji, nie z optymalnego rozwiązania. '
              'current_frozen to istniejący kontrakt remaining-actions + cechy planszy + JSON schema; '
              'pozostałe warianty różnią się wyłącznie przykładami.', '',
              '| V | nieterminalne poprawne / pary | remisy / brak | Spearman nieterm. (plansze) | płaskie nieterm. | MAE | wartości |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in values.items():
        pairs = row['pair_counts']['nonterminal']
        lines.append(f"| {name} | {pairs['concordant']}/{pairs['expected']} | {pairs['ties']}/{pairs['missing']} | "
                     f"{row['nonterminal_mean_spearman_defined']} ({row['nonterminal_spearman_defined_boards']}/8) | {row['flat_nonterminal_boards']} | "
                     f"{row['mae']} | {row['observed_values']}/{row['expected_values']} |")
    lines += ['', f"Wybór generatora: **{gen_winner}**; wybór V: **{value_winner}**.", '',
              'Spearman liczony jest wewnątrz planszy, a następnie uśredniany tylko tam, gdzie jest określony. '
              'Pary z remisem i brakującą odpowiedzią obniżają wynik wyboru. Wyniki ze stanem terminalnym '
              'i kompletne liczniki per plansza znajdują się w summary.json. Surowe odpowiedzi, '
              'osiągnięte cele i endpointy są w `summary.json` oraz plikach `openai/*.json`. '
              'Wartości dla wszystkich pozycji są w `value_predictions.csv`.']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'generator': generator, 'value_summary': {k: {key: v[key] for key in
           ('nonterminal_board_mean', 'pair_counts', 'mean_spearman_defined', 'spearman_defined_boards', 'flat_boards', 'flat_nonterminal_boards', 'nonterminal_mean_spearman_defined',
            'nonterminal_spearman_defined_boards', 'mae', 'errors')}
          for k, v in values.items()}, 'generator_winner': gen_winner, 'value_winner': value_winner}, indent=2))


if __name__ == '__main__':
    main()

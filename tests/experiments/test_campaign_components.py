"""Validation metrics preserve missing responses, ties, and terminal separation."""
from carl.icl.boards import from_rows, to_rows
from scripts.archive.campaign_20261102.campaign_components_validation import classify
from scripts.archive.campaign_20261102.report_campaign_components import rank_metrics


def test_generator_filter_stages_do_not_credit_copies_or_duplicates() -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    changed = ['#####', '#-@-#', '#-$-#', '#-.-#', '#####']
    import json
    raw = [{'output_text': json.dumps({'subgoals': [{'rows': to_rows(board)}, {'rows': changed}]})}]
    stages = classify(raw, board, False)
    assert stages['parse_ok']
    assert stages['candidates'][0]['source_copy'] and not stages['candidates'][0]['distinct_valid']
    assert stages['candidates'][1]['static_valid'] and stages['candidates'][1]['distinct_valid']
    raw = [{'output_text': json.dumps({'subgoals': [{'rows': changed}, {'rows': changed}]})}]
    stages = classify(raw, board, False)
    assert stages['candidates'][1]['duplicate'] and not stages['candidates'][1]['distinct_valid']


def test_value_ranking_counts_ties_and_missing_without_terminal_inflation() -> None:
    values = [0.1, 0.1, None, 0.4, 1.0]
    rows = [{'board_id': 'board', 'family': 'real', 'position': i, 'remaining': 4 - i, 'gamma': .9,
             'reference_value': .9 ** (4 - i), 'predicted_value': value, 'error': None if value is not None else {}}
            for i, value in enumerate(values)]
    result = rank_metrics(rows)
    pairs = result['pair_counts']['nonterminal']
    assert pairs == {'concordant': 2, 'discordant': 0, 'ties': 1, 'missing': 3, 'expected': 6}
    assert result['nonterminal_board_mean'] == 2 / 6
    assert result['pair_counts']['all']['concordant'] == 5
    assert result['observed_values'] == 4 and result['spearman_defined_boards'] == 1


def test_solved_state_does_not_hide_a_flat_nonterminal_prediction() -> None:
    values = [0.2, 0.2, 0.2, 0.2, 1.0]
    rows = [{'board_id': 'board', 'family': 'real', 'position': i, 'remaining': 4 - i, 'gamma': .9,
             'reference_value': .9 ** (4 - i), 'predicted_value': value, 'error': None}
            for i, value in enumerate(values)]
    result = rank_metrics(rows)
    assert result['flat_boards'] == 0 and result['flat_nonterminal_boards'] == 1
    assert result['spearman_defined_boards'] == 1 and result['nonterminal_spearman_defined_boards'] == 0
    assert result['pair_counts']['nonterminal']['ties'] == 6

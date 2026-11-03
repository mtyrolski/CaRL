"""Contract checks for the separate, matched prompt exploration."""
import json

from scripts.archive.campaign_20261102.campaign_generator_prompt import summarize, variant_prompt


def test_illustration_has_no_invented_scale_label() -> None:
    illustration = {'source': ['###', '#@#', '###'], 'successor': ['###', '#+#', '###'], 'action': 0}
    prompt = variant_prompt({'generator_prompt': 'Generate boards.'}, illustration)
    example = json.loads(prompt.split('answer:\n', 1)[1])
    assert example == {'source': illustration['source'], 'successor': illustration['successor'], 'action': 'UP'}
    assert 'NO percentile label' in prompt


def test_summary_keeps_bad_and_empty_responses_in_denominator() -> None:
    rows = [
        {'variant': 'baseline', 'reached_any': False, 'error': None, 'parse_ok': True, 'candidates': []},
        {'variant': 'baseline', 'reached_any': False, 'error': {'type': 'RequestFailed'}, 'parse_ok': False,
         'candidates': []},
        {'variant': 'baseline', 'reached_any': True, 'error': None, 'parse_ok': True,
         'candidates': [{'static_valid': True, 'source_copy': False, 'duplicate': False, 'distinct_valid': True}]},
    ]
    result = summarize(rows, expected=4)
    assert not result['complete']
    assert result['variants']['baseline']['requests'] == 3
    assert result['variants']['baseline']['reached_any'] == 1
    assert result['variants']['baseline']['errors'] == 1
    assert result['variants']['baseline']['empty_responses'] == 1
    assert result['variants']['baseline']['parse_failures'] == 1

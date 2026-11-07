"""Offline controls: no counted response or provider cost is fabricated."""
import json

import pytest
from pathlib import Path

from scripts.figure1_pilot.exact_token_preflight import project_count_request, quote_from_count


def request():
    return {'model': 'gpt-6.1-terra', 'input': [{'role': 'user', 'content': 'hello'}],
            'text': {'format': {'type': 'json_object'}}, 'reasoning': {'effort': 'none'},
            'tools': [], 'max_output_tokens': 256, 'stream': True, 'store': False,
            'include': [], 'service_tier': 'default'}


def price():
    return {'input_usd_per_million': '1', 'cached_input_usd_per_million': '0.05',
            'cache_write_30m_usd_per_million': '1.25', 'cache_write_12h_usd_per_million': '2.5',
            'output_usd_per_million': '5', 'max_input_tokens': 1050000, 'max_output_tokens': 128000,
            'long_context_threshold': 272000, 'regional_multiplier': '1.1'}


def test_exact_projection_includes_schema_and_reasoning():
    payload = request()
    projected = project_count_request(payload)
    assert projected == {key: payload[key] for key in ('model', 'input', 'text', 'reasoning', 'tools')}
    quote = quote_from_count(payload, projected, 2000, price())
    assert quote['generation_upper_nanos_assuming_rates'] == 6908000
    assert quote['count_endpoint_nanos_if_billed_as_input_assumption'] == 5500000
    assert quote['count_endpoint_price_verified'] is False and quote['budget_reserved'] is False
    with pytest.raises(ValueError, match='projection differs'):
        quote_from_count(payload, {**projected, 'text': {}}, 2000, price())


@pytest.mark.parametrize('field,value', [('previous_response_id', 'r'), ('conversation', 'c'),
                                          ('unreviewed_option', True), ('tools', [{'type': 'web_search'}])])
def test_context_changing_or_unknown_fields_fail_closed(field, value):
    with pytest.raises(ValueError):
        project_count_request({**request(), field: value})


def test_image_input_is_not_priced_as_text():
    payload = request()
    payload['input'] = [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'https://example.com/x'}]}]
    with pytest.raises(ValueError, match='non-text'):
        project_count_request(payload)


@pytest.mark.parametrize('value', [-1, True, 1050001, 0.5])
def test_malformed_count_is_rejected(value):
    payload = request()
    with pytest.raises(ValueError, match='Invalid provider token count'):
        quote_from_count(payload, project_count_request(payload), value, price())


def test_policy_payload_projects_with_metadata_and_schema():
    from carl.icl.boards import from_rows
    from scripts.reachability_runner import engine

    study = engine.load_study(Path('configs/icl/reachability_runner/offline_core_plan.json'))
    training = json.loads(Path('configs/icl/checkpoints/launch100/assets/data/training.json').read_text())
    context = training['contexts'][0]
    payload = engine._payload(study, study.arms['luna'], training,
                              from_rows(context['state']), from_rows(context['candidates'][0]['subgoal']))
    projected = project_count_request(payload)
    assert 'metadata' in payload and 'metadata' not in projected
    assert len(projected['input']) >= 2
    assert projected['text'] == payload['text'] and projected['reasoning'] == payload['reasoning']

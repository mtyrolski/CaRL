"""Projection and arithmetic for the provider's exact input-token counter.

This module does not call the counter or alter a ledger.  The counter endpoint's
own billing has not been verified; its use cannot establish a hard cost cap by
itself.  A count is not an authenticated statement once stored in a local file.
"""
from decimal import Decimal
from typing import Any

from carl.icl.campaign_budget import _nanos, _rate
from scripts.figure1_pilot.budget import _payload_hash

COUNT_FIELDS = frozenset({'model', 'input', 'instructions', 'reasoning', 'text',
                          'tools', 'tool_choice', 'parallel_tool_calls', 'truncation'})
# Only non-context fields supported by the actual, text-only policy request.
NON_CONTEXT_FIELDS = frozenset({'max_output_tokens', 'service_tier', 'stream', 'store', 'include', 'metadata'})


def project_count_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a stateless text-only Responses request, refusing unknown fields.

    The token endpoint supports input, reasoning and text.format, but not stream,
    output-token cap, storage, service tier or include.  Unknown fields fail
    closed because they could change the input context.  In particular, implicit
    previous responses and conversations cannot be safely sized here.
    """
    if not isinstance(payload, dict) or set(payload) - COUNT_FIELDS - NON_CONTEXT_FIELDS:
        raise ValueError('Unexpected Responses field in token count projection')
    if not isinstance(payload.get('model'), str) or not payload['model']:
        raise ValueError('A model is required')
    if not isinstance(payload.get('input'), (str, list)) or payload.get('tools', []):
        raise ValueError('Only explicit text inputs without tools are supported')
    for message in payload['input'] if isinstance(payload['input'], list) else []:
        if not isinstance(message, dict) or message.get('role') not in {'developer', 'system', 'user', 'assistant'}:
            raise ValueError('Only explicit text messages are supported')
        content = message.get('content')
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or any(not isinstance(part, dict)
                or part.get('type') not in {'input_text', 'output_text'}
                or not isinstance(part.get('text'), str) for part in content):
            raise ValueError('Images, files and non-text inputs are not supported')
    if payload.get('service_tier', 'default') != 'default':
        raise ValueError('Only the default tier is priced')
    maximum = payload.get('max_output_tokens')
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise ValueError('A positive output token cap is required')
    _payload_hash(payload)
    return {key: value for key, value in payload.items() if key in COUNT_FIELDS}


def quote_from_count(payload: dict[str, Any], count_request: dict[str, Any], count: int,
                     price: dict[str, Any]) -> dict[str, Any]:
    """Bind exact provider count to the request and quote conservative costs.

    Assumes known rates for the selected model and endpoint. The separate
    hypothetical count-endpoint charge is one maximum-rate input charge; no
    official count-endpoint price has been established, so this is an assumption,
    not a guarantee. This function is not a reservation and does not authorise a
    network request.
    """
    if count_request != project_count_request(payload):
        raise ValueError('Token count projection differs from the exact request')
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= price['max_input_tokens']:
        raise ValueError('Invalid provider token count')
    if payload['max_output_tokens'] > price['max_output_tokens']:
        raise ValueError('Output token cap exceeds the model profile')
    highest = max(_rate(price[name]) for name in ('input_usd_per_million', 'cached_input_usd_per_million',
                  'cache_write_30m_usd_per_million', 'cache_write_12h_usd_per_million'))
    output_rate = _rate(price['output_usd_per_million'])
    multiplier = _rate(price['regional_multiplier'])
    if multiplier < 1:
        raise ValueError('Regional multiplier must be conservative')
    # The existing campaign prices double input and 1.5x output at long context.
    in_factor = Decimal(2) if count > price['long_context_threshold'] else Decimal(1)
    out_factor = Decimal('1.5') if count > price['long_context_threshold'] else Decimal(1)
    input_cost = highest * count * in_factor * multiplier / Decimal(1_000_000)
    output_cost = output_rate * payload['max_output_tokens'] * out_factor * multiplier / Decimal(1_000_000)
    return {'payload_sha256': _payload_hash(payload), 'count_request_sha256': _payload_hash(count_request),
            'input_tokens': count, 'max_output_tokens': payload['max_output_tokens'],
            'generation_upper_nanos_assuming_rates': _nanos(input_cost + output_cost),
            'count_endpoint_nanos_if_billed_as_input_assumption': _nanos(input_cost),
            'count_endpoint_price_verified': False, 'budget_reserved': False}

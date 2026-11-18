"""Check response normalization without accepting unrelated text."""

import pytest

from carl.icl.intents.representation.normalization import remove_one_outer_fence


@pytest.mark.parametrize("opening", ["```json", "```"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_removes_one_complete_plain_or_json_fence(opening, newline):
    body = '{"effects":[]}'
    output = f"{opening}{newline}{body}{newline}```"
    assert remove_one_outer_fence(output) == (body, True)


@pytest.mark.parametrize(
    "output",
    [
        '{"effects":[]}',
        'Prose before\n```json\n{"effects":[]}\n```',
        '```json\n{"effects":[]}\n```\nprose after',
        '```python\n{"effects":[]}\n```',
        '```JSON\n{"effects":[]}\n```',
        '```json\n{"effects":[]}\n```\n```json\n{"effects":[]}\n```',
        '```json\n{"text":"```"}\n```',
        '```json\n{"effects":[]}\n```\n',
    ],
)
def test_does_not_modify_other_responses(output):
    assert remove_one_outer_fence(output) == (output, False)

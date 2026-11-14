import pytest

from carl.icl.intent_fence_audit import inspect_output
from carl.icl.intent_fence_audit import remove_one_outer_fence

BOARD = ["######", "#@-$.#", "#----#", "#----#", "#----#", "######"]
INTENT = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'


@pytest.mark.parametrize("opening", ["```json", "```"])
def test_accepts_only_one_complete_surrounding_fence(opening):
    output = f"{opening}\n{INTENT}\n```"
    assert remove_one_outer_fence(output) == (INTENT, True)
    audit = inspect_output(output, BOARD)
    assert audit["fence_removed"] and audit["json_valid"] and audit["valid_nontrivial"]


@pytest.mark.parametrize(
    "output",
    [
        f"Prose before\n```json\n{INTENT}\n```",
        f"```json\n{INTENT}\n```\nprose after",
        f"```python\n{INTENT}\n```",
        f"```json\n{INTENT}\n```\n```json\n{INTENT}\n```",
    ],
)
def test_does_not_extract_prose_other_languages_or_multiple_fences(output):
    assert remove_one_outer_fence(output) == (output, False)
    assert not inspect_output(output, BOARD)["json_valid"]


def test_unwrapped_json_is_unchanged_and_uses_main_intent_checker():
    assert remove_one_outer_fence(INTENT) == (INTENT, False)
    assert inspect_output(INTENT, BOARD)["valid_nontrivial"]
    already_true = '{"effects":[{"predicate":"box_at","row":1,"col":3}],"preserve":[]}'
    assert inspect_output(already_true, BOARD)["json_valid"]
    assert not inspect_output(already_true, BOARD)["valid_nontrivial"]

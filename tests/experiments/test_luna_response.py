"""Extract final text from faithful OpenAI Responses SDK objects."""

import pytest
from openai.types.responses import Response

from carl.icl.luna_response import LunaResponseError
from carl.icl.luna_response import final_assistant_text


def output_text(text):
    return {"type": "output_text", "text": text, "annotations": []}


def message(phase=None, content=None, status="completed"):
    value = {"type": "message", "id": "msg_test", "role": "assistant", "status": status,
             "content": [output_text('{"action":"UP"}')] if content is None else content}
    if phase is not None:
        value["phase"] = phase
    return value


def response(*items, status="completed", error=None):
    return Response.model_validate({
        "id": "resp_test", "created_at": 1.0, "model": "gpt-6-luna", "object": "response",
        "output": list(items), "parallel_tool_calls": False, "tool_choice": "none", "tools": [],
        "status": status, "error": error,
    })


def test_commentary_is_not_concatenated_with_final_answer():
    result = response(message("commentary", [output_text('{"action":"LEFT"}')]),
                      message("final_answer", [output_text('{"action":"RIGHT"}')]))
    assert result.output_text == '{"action":"LEFT"}{"action":"RIGHT"}'
    assert final_assistant_text(result) == '{"action":"RIGHT"}'


def test_unphased_single_message_is_supported():
    assert final_assistant_text(response(message())) == '{"action":"UP"}'


def test_one_final_message_can_have_several_ordered_text_parts():
    parts = [output_text('{"action":'), output_text('"UP"}')]
    assert final_assistant_text(response(message("final_answer", parts))) == '{"action":"UP"}'


def test_reasoning_items_are_not_mistaken_for_messages():
    reasoning = {"type": "reasoning", "id": "reason_test", "summary": []}
    assert final_assistant_text(response(reasoning, message("final_answer"))) == '{"action":"UP"}'


@pytest.mark.parametrize("items", [
    (),
    (message("commentary"),),
    (message(), message()),
    (message("commentary"), message()),
    (message("final_answer"), message("final_answer")),
])
def test_missing_or_ambiguous_final_is_rejected_without_content(items):
    with pytest.raises(LunaResponseError) as error:
        final_assistant_text(response(*items))
    assert '"action"' not in str(error.value)


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", "queued", "cancelled", None])
def test_noncompleted_response_is_rejected(status):
    with pytest.raises(LunaResponseError):
        final_assistant_text(response(message("final_answer"), status=status))


def test_incomplete_message_is_rejected_even_if_outer_response_is_completed():
    with pytest.raises(LunaResponseError):
        final_assistant_text(response(message("final_answer", status="incomplete")))


@pytest.mark.parametrize("phase", ["commentary", "final_answer", None])
def test_refusal_in_any_assistant_message_is_rejected_without_exposing_it(phase):
    refusal = message(phase, [{"type": "refusal", "refusal": "PRIVATE-REFUSAL-DETAIL"}])
    items = (refusal, message("final_answer")) if phase == "commentary" else (refusal,)
    with pytest.raises(LunaResponseError) as error:
        final_assistant_text(response(*items))
    assert "PRIVATE-REFUSAL-DETAIL" not in str(error.value)


@pytest.mark.parametrize("content", [[], [output_text("")], [output_text("   \n")]])
def test_empty_final_text_is_rejected(content):
    with pytest.raises(LunaResponseError):
        final_assistant_text(response(message("final_answer", content)))


def test_error_details_are_not_exposed():
    result = response(message("final_answer"), error={"code": "server_error", "message": "SECRET-DETAIL"})
    with pytest.raises(LunaResponseError) as error:
        final_assistant_text(result)
    assert "SECRET-DETAIL" not in str(error.value)

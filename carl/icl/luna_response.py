"""Extract one final assistant message from a Responses API result."""

from openai.types.responses import Response


class LunaResponseError(ValueError):
    """The response has no unambiguous, completed assistant answer."""


def final_assistant_text(response: Response) -> str:
    """Return only final-answer text; never merge commentary with the answer.

    If the service supplies no phase, a single assistant message is accepted.
    Multiple phase-less messages are ambiguous and are rejected. A refusal,
    incomplete result or unsupported final content is also rejected.
    Error messages contain no response text or service-provided detail.
    """
    if response.status != "completed" or response.error is not None:
        raise LunaResponseError("Response is not completed")

    messages = [item for item in response.output
                if item.type == "message" and item.role == "assistant"]
    for message in messages:
        if message.status != "completed":
            raise LunaResponseError("Assistant message is not completed")
        if any(part.type == "refusal" for part in message.content):
            raise LunaResponseError("Response contains a refusal")

    finals = [message for message in messages if message.phase == "final_answer"]
    if len(finals) == 1:
        final = finals[0]
    elif not finals and len(messages) == 1 and messages[0].phase is None:
        final = messages[0]
    else:
        raise LunaResponseError("Response has no unique final assistant message")

    if not final.content or any(part.type != "output_text" for part in final.content):
        raise LunaResponseError("Final message has no supported text")
    text = "".join(part.text for part in final.content if part.type == "output_text")
    if not text.strip():
        raise LunaResponseError("Final message has no supported text")
    return text

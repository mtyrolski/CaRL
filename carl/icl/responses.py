"""Build model requests and normalize the supported text API contracts."""
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeGuard

from carl.icl.types import JSONObject, JSONValue

if TYPE_CHECKING:
    from openai.types.responses import Response, ResponseStreamEvent

RESPONSE_CONTRACT_VERSION: Final[str] = "luna-stream-v1"
FAILED: Final[frozenset[str]] = frozenset(
    {"error", "response.failed", "response.incomplete", "response.refusal.delta"}
)
INCLUDE: Final[tuple[str, ...]] = (
    "reasoning.encrypted_content",
    "web_search_call.action.sources",
)


class ResponseStream(Protocol):
    """The iterable, closable subset of the SDK stream used by inference."""

    def __iter__(self) -> Iterator["ResponseStreamEvent"]:
        """Iterate over server events."""
        ...

    def close(self) -> None:
        """Release the underlying HTTP response."""
        ...


def text_message(
    role: Literal["developer", "user", "assistant"], text: str
) -> JSONObject:
    """Encode prompts and demonstration answers using explicit content blocks."""
    return {
        "role": role,
        "content": [
            {
                "type": "output_text" if role == "assistant" else "input_text",
                "text": text,
            }
        ],
    }


def request_options(effort: str, output_format: JSONObject) -> JSONObject:
    """Build fresh options matching the Luna streaming contract."""
    return {
        "service_tier": "default",
        "text": {"format": output_format, "verbosity": "low"},
        "reasoning": {"effort": effort, "mode": "standard", "summary": "auto"},
        "tools": [],
        "stream": True,
        "store": False,
        "include": list(INCLUDE),
    }


def is_token_count(value: JSONValue) -> TypeGuard[int]:
    """Recognize nonnegative integral usage and limit fields."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def chat_payload(payload: JSONObject) -> JSONObject:
    """Translate canonical text messages to the compatible Chat Completions API."""
    reasoning = payload.get("reasoning") or {}
    if reasoning.get("effort", "none") != "none":
        raise ValueError("Chat Completions does not support this reasoning setting")
    if payload.get("stream") or payload.get("tools") or payload.get("service_tier", "default") != "default":
        raise ValueError("Chat Completions requires non-streamed text without tools")
    if payload.get("prompt_cache_options"):
        raise ValueError("Chat Completions does not support explicit prompt cache controls")
    source = payload["input"]
    if isinstance(source, str):
        source = [{"role": "user", "content": source}]
    messages = []
    for message in source:
        role = message["role"]
        if role not in ("developer", "system", "user", "assistant"):
            raise ValueError("Unsupported Chat Completions message role")
        content = message["content"]
        if isinstance(content, list):
            if any(
                part.get("type") not in ("input_text", "output_text")
                or not isinstance(part.get("text"), str)
                or set(part) != {"type", "text"}
                for part in content
            ):
                raise ValueError("Chat Completions requires plain text content")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Chat Completions requires plain text content")
        messages.append({"role": "system" if role == "developer" else role, "content": content})
    maximum = payload["max_output_tokens"]
    if not messages or not is_token_count(maximum) or maximum < 1:
        raise ValueError("Chat Completions requires messages and a positive completion limit")
    wire = {"model": payload["model"], "messages": messages, "max_tokens": maximum}
    response_format = (payload.get("text") or {}).get("format") or {"type": "text"}
    match response_format["type"]:
        case "json_schema":
            wire["response_format"] = {
                "type": "json_schema",
                "json_schema": {key: value for key, value in response_format.items() if key != "type"},
            }
        case "json_object":
            wire["response_format"] = {"type": "json_object"}
        case "text":
            pass
        case _:
            raise ValueError("Unsupported Chat Completions response format")
    if "temperature" in payload:
        wire["temperature"] = payload["temperature"]
    return wire


def chat_output_text(response: JSONObject) -> str:
    """Accept a single complete assistant text choice from Chat Completions."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Chat Completions requires one response choice")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ValueError("Chat Completions response is incomplete")
    if message.get("role") != "assistant" or message.get("refusal") or message.get("tool_calls"):
        raise ValueError("Chat Completions did not return assistant text")
    content = message.get("content")
    if isinstance(content, list):
        if any(
            not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str)
            for part in content
        ):
            raise ValueError("Chat Completions returned non-text content")
        content = "".join(part["text"] for part in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat Completions returned empty output")
    return content


def normalize_usage(usage: JSONObject, endpoint: str) -> JSONObject:
    """Keep native usage fields and provide consistent billable token counters."""
    if endpoint == "responses":
        return usage
    result = dict(usage)
    if "prompt_tokens" in usage:
        result["input_tokens"] = usage["prompt_tokens"]
    if endpoint == "embeddings":
        result["output_tokens"] = 0
    elif "completion_tokens" in usage:
        result["output_tokens"] = usage["completion_tokens"]
    if isinstance(usage.get("prompt_tokens_details"), dict):
        result["input_tokens_details"] = usage["prompt_tokens_details"]
    return result


def collect_response(
    stream: ResponseStream,
    *,
    deadline: float,
    on_terminal: Callable[["Response"], None],
) -> "Response":
    """Require a completed terminal response; always close, never accept partial text."""
    try:
        for event in stream:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Response stream exceeded the run deadline")
            match event.type:
                case "response.completed":
                    on_terminal(event.response)
                    return event.response
                case "response.failed" | "response.incomplete":
                    on_terminal(event.response)
            if event.type in FAILED:
                # Do not expose provider messages or refusal text in logs.
                raise ValueError("Response stream failed, was refused or incomplete")
        raise ValueError("Response stream ended without a completed event")
    finally:
        stream.close()

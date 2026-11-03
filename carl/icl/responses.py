"""Shared Responses request contract and bounded stream collection."""
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Final, Literal, Protocol

from carl.icl.types import JSONObject

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

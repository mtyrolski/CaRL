"""Normalize structured model responses without extracting surrounding prose."""

import re

FENCE = re.compile(r"\A```(?:json)?\r?\n(?P<body>.*?)\r?\n```\Z", re.DOTALL)


def remove_one_outer_fence(output: str) -> tuple[str, bool]:
    """Remove one complete plain or JSON Markdown fence, if present."""
    match = FENCE.fullmatch(output)
    if match is None or "```" in match.group("body"):
        return output, False
    return match.group("body"), True

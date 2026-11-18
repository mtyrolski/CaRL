"""Sanitise service errors before writing request diagnostics."""

import re
from typing import Any

SAFE_ERROR_CODES = frozenset({
    "authentication_error", "billing_hard_limit_reached", "content_policy_violation", "insufficient_quota",
    "invalid_api_key", "invalid_prompt", "invalid_request_error", "model_not_found", "organization_deactivated",
    "permission_denied", "permission_error", "project_deactivated", "rate_limit_exceeded", "rate_limit_error",
    "server_error", "service_unavailable", "unsupported_model", "unsupported_parameter",
})


def error_diagnostics(error: BaseException) -> dict[str, Any]:
    """Keep recognised service details while redacting potentially sensitive values."""
    status = getattr(error, "status_code", None)
    result: dict[str, Any] = {"http_status": status if isinstance(status, int) and not isinstance(status, bool)
                              and 100 <= status <= 599 else None}
    for name in ("code", "type"):
        value = getattr(error, name, None)
        result[f"service_{name}"] = value if isinstance(value, str) and value in SAFE_ERROR_CODES else None
    request_id = getattr(error, "request_id", None)
    result["request_id"] = (request_id if isinstance(request_id, str) and request_id.startswith("req_")
                            and len(request_id) <= 80 and request_id[4:].isalnum() else None)
    parameter = getattr(error, "param", None)
    allowed = {"model", "input", "temperature", "reasoning", "reasoning.effort", "text", "text.format",
               "max_output_tokens", "service_tier", "tools", "store"}
    if isinstance(parameter, str) and parameter in allowed:
        result["service_param"] = parameter
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        message = re.sub(r"(?i)https?://\S+", "[redacted-url]", message)
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", message)
        message = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[redacted-key]", message)
        message = re.sub(r"(?i)(api[_ -]?key|authorization|access[_ -]?token)\s*[:=]\s*[^\s,;}]+",
                         r"\1=[redacted]", message)
        message = re.sub(r"\b[A-Za-z0-9_/-]{24,}\b", "[redacted-token]", message)
        result["service_message"] = " ".join(message.split())[:280]
    return result

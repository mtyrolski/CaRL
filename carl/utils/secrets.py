"""Helpers for safely displaying secrets in logs."""


def mask_secret(secret: str | None, *, prefix_chars: int = 4, suffix_chars: int = 4) -> str:
    """Return a log-safe representation of a secret value."""
    if prefix_chars < 0 or suffix_chars < 0:
        msg = "prefix_chars and suffix_chars must be non-negative"
        raise ValueError(msg)

    if secret is None:
        return "<missing>"

    if secret == "":
        return "<empty>"

    if len(secret) <= prefix_chars + suffix_chars:
        return "*" * len(secret)

    return f"{secret[:prefix_chars]}...{secret[-suffix_chars:]}"

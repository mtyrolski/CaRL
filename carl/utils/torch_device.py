import os

from loguru import logger
import torch

CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
_CUDA_DISABLE_VALUES = {"", "-1", "none"}


def _cuda_disabled_via_env() -> bool:
    value = os.environ.get(CUDA_VISIBLE_DEVICES_ENV)
    if value is None:
        return False
    return value.strip().lower() in _CUDA_DISABLE_VALUES


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """
    Resolve a torch.device without triggering CUDA initialization when it is explicitly disabled.
    """
    if not prefer_cuda or _cuda_disabled_via_env():
        return torch.device("cpu")

    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception as exc:
        logger.warning(f"torch.cuda.is_available() failed; falling back to CPU: {exc}")

    return torch.device("cpu")

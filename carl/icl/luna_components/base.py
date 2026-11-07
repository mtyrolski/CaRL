"""Backward-compatible imports; implementations live in ``llm_components``."""
from carl.icl.llm_components.base import *  # noqa: F401,F403
from carl.icl.llm_components.base import LLMComponentBase

LunaBase = LLMComponentBase

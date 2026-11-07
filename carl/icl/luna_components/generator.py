"""Backward-compatible imports; implementations live in ``llm_components``."""
from carl.icl.llm_components.generator import *  # noqa: F401,F403
from carl.icl.llm_components.generator import LLMSubgoalGenerator

LunaGenerator = LLMSubgoalGenerator

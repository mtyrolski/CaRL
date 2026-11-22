"""Backward-compatible imports; implementations live in ``llm_components``."""
from carl.icl.llm_components.generator import *  # noqa: F401,F403  # pylint: disable=wildcard-import,unused-wildcard-import
from carl.icl.llm_components.generator import LLMSubgoalGenerator

LunaGenerator = LLMSubgoalGenerator

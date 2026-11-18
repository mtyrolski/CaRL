"""Backward-compatible imports; implementations live in ``llm_components``."""
from carl.icl.llm_components.policy import *  # noqa: F401,F403
from carl.icl.llm_components.policy import GoalConditionedPolicy

LunaPolicy = GoalConditionedPolicy

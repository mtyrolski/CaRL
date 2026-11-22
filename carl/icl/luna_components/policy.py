"""Backward-compatible imports; implementations live in ``llm_components``."""
from carl.icl.llm_components.policy import *  # noqa: F401,F403  # pylint: disable=wildcard-import,unused-wildcard-import
from carl.icl.llm_components.policy import GoalConditionedPolicy

LunaPolicy = GoalConditionedPolicy

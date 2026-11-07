"""Model-neutral prompted generator, conditional policy and value components."""

from carl.icl.llm_components.generator import LLMSubgoalGenerator
from carl.icl.llm_components.policy import GoalConditionedPolicy
from carl.icl.llm_components.value import LLMValueFunction


class ICLComponents(LLMSubgoalGenerator, GoalConditionedPolicy, LLMValueFunction):
    """Share one configured model and inference client across the ICL roles."""

__all__ = ["ICLComponents"]

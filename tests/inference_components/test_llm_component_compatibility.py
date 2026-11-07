"""Legacy Luna names remain exact aliases of model-neutral implementations."""
import pickle

from carl.icl.components import ICLComponents
from carl.icl.llm_components.base import LLMComponentBase
from carl.icl.llm_components.formatting import task
from carl.icl.llm_components.generator import LLMSubgoalGenerator
from carl.icl.llm_components.policy import GoalConditionedPolicy
from carl.icl.llm_components.value import LLMValueFunction
from carl.icl.luna import LunaComponents
from carl.icl.luna_components.base import LunaBase
from carl.icl.luna_components.formatting import task as legacy_task
from carl.icl.luna_components.generator import LunaGenerator
from carl.icl.luna_components.policy import LunaPolicy
from carl.icl.luna_components.value import LunaValue


def test_legacy_and_neutral_imports_share_class_identity():
    assert LunaComponents is ICLComponents
    assert LunaBase is LLMComponentBase
    assert LunaGenerator is LLMSubgoalGenerator
    assert LunaPolicy is GoalConditionedPolicy
    assert LunaValue is LLMValueFunction
    assert task is legacy_task
    # Historic pickle global references must continue to load.
    assert pickle.loads(b'ccarl.icl.luna_components.policy\nLunaPolicy\n.') is GoalConditionedPolicy

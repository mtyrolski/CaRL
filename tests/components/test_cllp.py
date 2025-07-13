import pytest

from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy


class CllpMockIncremental(ConditionalLowLevelPolicy):
    def __init__(self, conditional_low_level_policy_class, path_to_conditional_low_level_policy_weights, env) -> None:
        pass

    def get_action(
        self,
        state,
        state_after_k,
    ):
        if state_after_k == state:
            return 0
        return 1 if state_after_k > state else -1

    def construct_network(self):
        pass

    def get_network(self):
        return None

    def get_component_training_module(self):
        return None


def reachability(cllp, state: int, target: int, budget: int):
    for _ in range(budget):
        if state == target:
            return True
        state += cllp.get_action(state, target)
    return state == target


def test_cllp_incremental():
    """Test the SolvingPathConditionalLowLevelPolicyReplayBuffer class."""
    cllp = CllpMockIncremental(None, None, None)

    # Check the replay buffer
    assert cllp.get_action(0, 0) == 0
    assert cllp.get_action(0, 1) == 1
    assert cllp.get_action(1, 0) == -1
    assert cllp.get_action(1, 1) == 0
    assert cllp.get_action(1, 2) == 1
    assert cllp.get_action(2, 1) == -1


@pytest.mark.parametrize('state, target, budget, expected', [
    (0, 6, 10, True),
    (0, 6, 5, False),
    (0, 6, 6, True),
    (10, 100, 90, True),
    (10, 100, 89, False),
    (10, 100, 91, True),
])
def test_reachability(state, target, budget, expected):
    """Test the reachability function."""
    cllp = CllpMockIncremental(None, None, None)
    assert reachability(cllp, state, target, budget) == expected

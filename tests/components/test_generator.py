import pytest
from carl.solver.nodes import GeneratedSubgoal, SearchTreeNode
from tests.utils import build_dummy_search_tree


class GeneratorMock:
    def __init__(
        self,
        k_distance: int = 1,
        num_subgoals: int = 3,
    ):
        self.k_distance = k_distance
        self.num_subgoals = num_subgoals

    def construct_network(self):
        pass

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        return [GeneratedSubgoal(node.state + self.k_distance + j, {}) for j in range(self.num_subgoals)]


@pytest.mark.parametrize('k, num_subgoals', [(1, 3), (2, 2), (3, 1), (5, 10)])
def test_generator(k, num_subgoals):
    """Test the SubgoalGenerator class."""

    print('Testing SubgoalGenerator...')

    _, solving_node = build_dummy_search_tree()
    generator = GeneratorMock(k, num_subgoals)

    # Check the subgoals
    subgoals = generator.get_subgoals(solving_node)
    assert len(subgoals) == num_subgoals

    for i, subgoal in enumerate(subgoals):
        assert subgoal.state == solving_node.state + k + i

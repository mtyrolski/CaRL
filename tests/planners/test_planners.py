import numpy as np
import pytest

from carl.planners.adasubs import AdasubsPlanner
from carl.planners.greedy import GreedyPlanner, GreedyPlanner, BestFSPlanner, AstarPlanner, BfsPlanner
from carl.planners.base import Experience, SearchInfo

from carl.planners.base import SearchTreeNode
from carl.solver.nodes import get_root_from_solving_node
from tests.utils import build_dummy_search_tree


@pytest.mark.parametrize('generators_k_list', [[2, 3, 4], [1, 2, 3, 4], [4, 3, 2], [3, 2, 4]])
def test_adasubs_planner_get(generators_k_list: list[int]):
    root_state = np.ones((12, 12, 7))
    planner = AdasubsPlanner(root_state, generators_k_list)

    for k in sorted(generators_k_list, reverse=True):
        node = planner.get()
        assert isinstance(node, SearchTreeNode)
        assert np.array_equal(node.state, root_state)
        assert node.parent_node is not None
        assert node.next_expand_with_k_generator == k

    assert planner.get() is None    # No more nodes to get


def test_adasubs_planner_is_seen():
    root_state = np.ones((12, 12, 7))
    generators_k_list = [2, 3, 4]
    planner = AdasubsPlanner(root_state, generators_k_list)

    state = np.ones((12, 12, 7))

    assert planner.is_seen(state) == False

    with pytest.raises(ValueError):
        planner.add(SearchTreeNode(state, 0, None, None, 2))

    root_node = planner.root_node

    planner.add(SearchTreeNode(np.ones((12, 12, 7)), 0, None, root_node, metadata={}))

    assert planner.is_seen(state) == True


def test_adasubs_planner_add():
    root_state = np.ones((12, 12, 7))
    generators_k_list = [2, 3, 4]
    planner = AdasubsPlanner(root_state, generators_k_list)
    root_node = planner.root_node
    node = SearchTreeNode(np.ones((12, 12, 7)), 0, None, root_node, metadata={})
    planner.add(node)

    assert np.array_equal(planner.get().state, node.state)


def test_adasubs_planner_get_solution_data():
    _, solving_node = build_dummy_search_tree(6)
    generators_k_list = [2, 3, 4]
    root_node = get_root_from_solving_node(solving_node)
    planner = AdasubsPlanner(root_node.state, generators_k_list)

    search_info = SearchInfo(finished_reason='solved', low_level_nodes_visited=7)
    experience: Experience = planner.get_solution_data(solving_node, search_info)

    assert experience.solution.solved == True
    assert experience.search_info.finished_reason == 'solved'
    assert experience.search_info.solving_node == solving_node
    assert experience.solution.action_path == [1, 1, 1, 1, 1, 1]


def get_root_state():
    return np.random.rand(10, 10)


def get_root_node(state):
    return SearchTreeNode(state, 0, None, None)


# Tests for GreedyPlanner
class TestGreedyPlanner:
    def test_initialization(self):
        root_state = get_root_state()
        planner = GreedyPlanner(root_state)
        assert np.array_equal(planner.root_state, root_state)

    def test_add_and_get(self):
        root_state = get_root_state()
        planner = GreedyPlanner(root_state)
        node = SearchTreeNode(root_state, 0, None, None)
        planner.add(node)
        retrieved_node = planner.get()
        assert retrieved_node is not None
        assert np.array_equal(retrieved_node.state, node.state)

    def test_sequence_of_nodes(self):
        root_state = get_root_state()
        planner = GreedyPlanner(root_state)
        nodes = []
        for i in range(18):
            node = SearchTreeNode(state=root_state + i,
                                  value=i,
                                  parent_node=None,
                                  next_expand_with_k_generator=None,
                                  low_level_path=[1])
            nodes.append(node)

        for node in nodes:
            planner.add(node)

        nodes = [planner.root_node, *nodes]

        for expected_node in nodes:
            retrieved_node = planner.get()
            assert np.array_equal(retrieved_node.state, expected_node.state)

    def test_is_seen(self):
        root_state = get_root_state()
        root_node = get_root_node(root_state)
        planner = GreedyPlanner(root_state)
        planner.add(root_node)
        assert planner.is_seen(root_state) == True

    def test_greedy_selects_highest_value(self):
        root = get_root_state()
        planner = GreedyPlanner(root)
        low = SearchTreeNode(root, 1, None, None)
        high = SearchTreeNode(root, 10, None, None)
        planner.add(low)
        planner.add(high)
        chosen = planner.get()
        assert np.array_equal(chosen.state, high.state)


# Tests for BestFSPlanner
class TestBestFSPlanner:
    def test_priority_calculation(self):
        root_state = get_root_state()
        planner = BestFSPlanner(root_state)
        node = SearchTreeNode(root_state, 10, None, None)
        assert planner.get_node_priority(node) == -10

    @pytest.mark.parametrize('N', [5, 10, 15])
    def test_sequence_of_nodes(self, N):
        root_state = get_root_state()
        planner = BestFSPlanner(root_state)
        nodes = []
        for i in range(N):
            node = SearchTreeNode(state=root_state + i,
                                  value=i / N,
                                  parent_node=None,
                                  next_expand_with_k_generator=None,
                                  low_level_path=[1])
            nodes.append(node)

        for node in nodes:
            planner.add(node)

        nodes = [*reversed(nodes), planner.root_node]

        for expected_node in nodes:
            retrieved_node = planner.get()
            assert np.array_equal(retrieved_node.state, expected_node.state)

    @pytest.mark.parametrize('N', [5, 10, 15])
    def test_sequence_of_nodes_2(self, N):
        root_state = get_root_state()
        planner = BestFSPlanner(root_state)
        nodes = []
        for i in range(N):
            node = SearchTreeNode(state=root_state + i,
                                  value=(N - i) / N,
                                  parent_node=None,
                                  next_expand_with_k_generator=None,
                                  low_level_path=[1])
            nodes.append(node)

        for node in nodes:
            planner.add(node)

        nodes = [*nodes, planner.root_node]

        for expected_node in nodes:
            retrieved_node = planner.get()
            assert np.array_equal(retrieved_node.state, expected_node.state)

    def test_bestfs_priority_negative_value(self):
        root = get_root_state()
        planner = BestFSPlanner(root)
        node = SearchTreeNode(root, 4.2, None, None)
        priority = planner.get_node_priority(node)
        assert priority == pytest.approx(-4.2)


# Tests for AstarPlanner
class TestAstarPlanner:
    def test_priority_calculation(self):
        root_state = get_root_state()
        root_node = get_root_node(root_state)
        planner = AstarPlanner(root_state, value_weight=1.0, depth_weight=1.0)
        node = SearchTreeNode(root_state, 10, None, root_node, metadata={'depth': 5})
        assert planner.get_node_priority(node) == -5.0    # -10 value and -5 for depth

    def test_sequence_of_nodes(self):
        root_state = get_root_state()
        planner = AstarPlanner(root_state, value_weight=1.0, depth_weight=1.0)
        nodes = []
        for i in range(18):
            node = SearchTreeNode(state=root_state + i,
                                  value=2 * i,
                                  parent_node=None,
                                  next_expand_with_k_generator=None,
                                  low_level_path=[1],
                                  metadata={'depth': i})
            nodes.append(node)

        for node in nodes:
            planner.add(node)

        high_to_low = list(reversed(nodes))  # 17,16,…,0
        expected = [*high_to_low, planner.root_node]

        for expected_node in expected:
            retrieved_node = planner.get()
            assert np.array_equal(retrieved_node.state, expected_node.state)

# Tests for BfsPlanner
class TestBfsPlanner:
    def test_add_and_get(self):
        root_state = get_root_state()
        planner = BfsPlanner(root_state)
        node = SearchTreeNode(root_state, 0, None, None)
        planner.add(node)
        retrieved_node = planner.get()
        assert retrieved_node is not None
        assert np.array_equal(retrieved_node.state, node.state)

    def test_sequence_of_nodes(self):
        root_state = get_root_state()
        planner = BfsPlanner(root_state)
        nodes = []
        for i in range(18):
            node = SearchTreeNode(state=root_state + i,
                                  value=i,
                                  parent_node=None,
                                  next_expand_with_k_generator=None,
                                  low_level_path=[1])
            nodes.append(node)

        for node in nodes:
            planner.add(node)

        nodes = [planner.root_node, *nodes]

        for expected_node in nodes:
            retrieved_node = planner.get()
            assert np.array_equal(retrieved_node.state, expected_node.state)

    def test_priority_is_zero(self):
        root_state = get_root_state()
        planner = BfsPlanner(root_state)
        node = SearchTreeNode(root_state, 10, None, None)
        assert planner.get_node_priority(node) == 0.0    # BFS does not use node value for priority

def test_adasubs_add_expands_all_k():
    root = np.zeros((3,3))
    ks = [1,2,3]
    planner = AdasubsPlanner(root, ks)
    # clear out the initial subgoal for setup
    _ = planner.get()
    before = planner.nodes_queue.size()  # assuming SafePriorityQueue.size()
    node = SearchTreeNode(root, 5, [], planner.root_node, None, metadata={'depth':0})
    planner.add(node)
    assert planner.nodes_queue.size() == before + len(ks)
    
    
def test_adasubs_selection_counters():
    root = np.ones((2,2))
    ks = [1,2]
    planner = AdasubsPlanner(root, ks)
    # each get() for k should bump that k’s counter
    first = planner.get()
    k1 = first.next_expand_with_k_generator
    assert planner.subgoals_selected_for_expansion[k1] == 1
    second = planner.get()
    k2 = second.next_expand_with_k_generator
    assert planner.subgoals_selected_for_expansion[k2] == 1
    
def test_adasubs_is_seen_after_add():
    root = np.zeros((4,4))
    planner = AdasubsPlanner(root, [1])
    new_state = np.ones((4,4))
    assert not planner.is_seen(new_state)
    node = SearchTreeNode(new_state, 0, [], planner.root_node, None)
    planner.add(node)
    assert planner.is_seen(new_state)
    
def test_adasubs_get_solution_data_unsolved():
    root = np.zeros((2,2))
    planner = AdasubsPlanner(root, [1], prune_search_trees=True)
    info = SearchInfo(finished_reason='timeout', low_level_nodes_visited=0)
    exp = planner.get_solution_data(None, info)
    assert exp.solution.solved is False
    assert exp.search_info.finished_reason == 'timeout'

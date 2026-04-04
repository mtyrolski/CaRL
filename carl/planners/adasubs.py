import numpy as np
from loguru import logger

from carl.planners.base import Experience
from carl.planners.base import Planner
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.planners.base import get_tree_info
from carl.solver.nodes import SafePriorityQueue
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import get_solving_path_data
from carl.solver.nodes import prune_search_tree_from_solving_node


class AdasubsPlanner(Planner):
    def __init__(self, root_state: np.ndarray, generators_k_list: list[int], prune_search_trees: bool = False) -> None:
        super().__init__(root_state)
        self.generators_k_list = generators_k_list
        self.seen_states: set[tuple[int, ...] | tuple[str, ...]] = set()

        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()
        self.root_node = SearchTreeNode(root_state, 0, None, None, metadata={'depth': -1})

        self.subgoals_added = {k: 0 for k in self.generators_k_list}
        self.subgoals_selected_for_expansion = {k: 0 for k in self.generators_k_list}
        self.prune_search_trees = prune_search_trees

        for k in self.generators_k_list:
            root_key = (
                -k,
                0,
            )    # Note: 0 is the maximum value in priority queue since this queue is min-heap
            node = SearchTreeNode(root_state, 0, None, self.root_node, k)

            self.nodes_queue.put(data=node, key=root_key)
            self.subgoals_added[k] += 1

    def get(self):
        if self.nodes_queue.empty():
            return None

        node: SearchTreeNode = self.nodes_queue.get() # type: ignore
        assert node.next_expand_with_k_generator is not None, \
            'Node during getting from queue should have next_expand_with_k_generator set to not None'
        self.subgoals_selected_for_expansion[node.next_expand_with_k_generator] += 1
        return node

    def is_seen(self, state: np.ndarray | str) -> bool:
        if isinstance(state, str):
            return tuple(state) in self.seen_states
        if isinstance(state, np.ndarray):
            return tuple(state.flatten()) in self.seen_states
        return False

    def add(self, node: SearchTreeNode) -> None:
        if node.next_expand_with_k_generator is not None:
            logger.error('Node during adding to queue should have next_expand_with_k_generator set to None')
            logger.error('Planner decides which k-distances should be expanded, not the search itself')
            raise ValueError('Node during adding to queue should have next_expand_with_k_generator set to None')

        if isinstance(node.state, str):
            self.seen_states.add(tuple(node.state))
        elif isinstance(node.state, np.ndarray):
            self.seen_states.add(tuple(node.state.flatten()))

        for k in self.generators_k_list:
            metadata = {'depth': node.parent_node.metadata['depth'] + 1}
            new_node = SearchTreeNode(node.state, node.value, node.low_level_path, node.parent_node, k, metadata)
            self.nodes_queue.put(data=new_node, key=(-k, -node.value))
            self.subgoals_added[k] += 1

    def get_solution_data(self, solving_node: SearchTreeNode | None, search_info: SearchInfo) -> Experience:
        search_info.subgoals_visited = len(self.seen_states)
        search_info.search_tree = self.root_node

        if self.prune_search_trees and solving_node is not None:
            prune_search_tree_from_solving_node(solving_node)

        search_info.solving_node = solving_node
        search_info.subgoals_added_per_k = self.subgoals_added
        search_info.subgoals_selected_for_expansion = self.subgoals_selected_for_expansion

        tree_info = get_tree_info(self.root_node, search_info)
        for k, v in tree_info.items():
            if hasattr(search_info, k):
                setattr(search_info, k, v)

        if solving_node is None:
            solution = Solution(solved=False)
            return Experience(solution=solution, search_info=search_info)

        subgoal_path, action_path, values, subgoal_distance_path, _ = get_solving_path_data(
            solving_node, include_state_path=False)
        solution = Solution(
            solved=True,
            subgoal_path=subgoal_path,
            action_path=action_path,
            subgoal_distance_path=subgoal_distance_path,
            subgoal_values=values
        )
        return Experience(solution=solution, search_info=search_info)


class UniversalPropositionalPlanner(Planner):
    """AdaSubS-style planner variant for universal propositional generators (no explicit k bank).

    Keeps the same `Planner` interface but uses a single priority queue and never assigns
    `next_expand_with_k_generator`.
    """

    def __init__(self, root_state: np.ndarray, prune_search_trees: bool = False) -> None:
        super().__init__(root_state)
        self.seen_states: set[tuple[int, ...] | tuple[str, ...]] = set()

        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()
        self.root_node = SearchTreeNode(root_state, 0, None, None, metadata={'depth': -1})

        self.subgoals_added_total = 0
        self.subgoals_selected_for_expansion_total = 0
        self.prune_search_trees = prune_search_trees

        root_frontier_node = SearchTreeNode(root_state, 0, None, self.root_node)
        self.nodes_queue.put(data=root_frontier_node, key=0.0)
        self.subgoals_added_total += 1

    def get(self) -> SearchTreeNode | None:
        if self.nodes_queue.empty():
            return None
        node = self.nodes_queue.get()
        if node is None:
            return None
        self.subgoals_selected_for_expansion_total += 1
        return node

    def is_seen(self, state: np.ndarray | str) -> bool:
        if isinstance(state, str):
            return tuple(state) in self.seen_states
        if isinstance(state, np.ndarray):
            return tuple(state.flatten()) in self.seen_states
        return False

    def add(self, node: SearchTreeNode) -> None:
        if node.next_expand_with_k_generator is not None:
            logger.error('Universal planner expects nodes with next_expand_with_k_generator=None')
            raise ValueError('Universal planner expects nodes with next_expand_with_k_generator=None')

        if isinstance(node.state, str):
            self.seen_states.add(tuple(node.state))
        elif isinstance(node.state, np.ndarray):
            self.seen_states.add(tuple(node.state.flatten()))

        # Highest value first (SafePriorityQueue is min-heap).
        self.nodes_queue.put(data=node, key=-float(node.value))
        self.subgoals_added_total += 1

    def get_solution_data(self, solving_node: SearchTreeNode | None, search_info: SearchInfo) -> Experience:
        search_info.subgoals_visited = len(self.seen_states)
        search_info.search_tree = self.root_node

        if self.prune_search_trees and solving_node is not None:
            prune_search_tree_from_solving_node(solving_node)

        search_info.solving_node = solving_node
        # Keep existing fields populated (empty-per-k for universal mode), plus universal totals in optional fields.
        search_info.subgoals_added_per_k = {}
        search_info.subgoals_selected_for_expansion = {}
        search_info.proposal_events_count = max(search_info.proposal_events_count or 0, 0)

        tree_info = get_tree_info(self.root_node, search_info)
        for k, v in tree_info.items():
            if hasattr(search_info, k):
                setattr(search_info, k, v)

        if solving_node is None:
            solution = Solution(solved=False)
            return Experience(solution=solution, search_info=search_info)

        subgoal_path, action_path, values, subgoal_distance_path, _ = get_solving_path_data(
            solving_node, include_state_path=False)
        solution = Solution(
            solved=True,
            subgoal_path=subgoal_path,
            action_path=action_path,
            subgoal_distance_path=subgoal_distance_path,
            subgoal_values=values,
        )
        return Experience(solution=solution, search_info=search_info)

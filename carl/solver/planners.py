import queue
from abc import abstractmethod
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
from enum import StrEnum

from carl.environment.sokoban.env import printable_sokoban_state
from carl.environment.utilis import DeadEndFinder
from carl.solver.nodes import SafePriorityQueue, prune_search_tree_from_solving_node
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import get_solving_path_data
from carl.solver.nodes import hashable_state

GeneratorIdx = int

@dataclass
class Solution:
    solved: bool
    subgoal_path: list[np.ndarray] | None = None
    action_path: list[int] | None = None
    subgoal_distance_path: list[int] | None = None
    
    def __post_init__(self):
        if self.solved and (self.subgoal_path is None or self.action_path is None or self.subgoal_distance_path is None):
            raise ValueError("If solved, subgoal_path, action_path and subgoal_distance_path must be provided.")
        if not self.solved and (self.subgoal_path is not None or self.action_path is not None or self.subgoal_distance_path is not None):
            raise ValueError("If not solved, subgoal_path, action_path and subgoal_distance_path must be None.")

@dataclass
class SearchInfo:
    finished_reason: str
    low_level_nodes_visited: int = 0
    high_level_nodes_valid: int = 0
    high_level_nodes_unreachable: int = 0
    subgoals_reachable_count_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_unreachable_count_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_reachable_rate_per_k: dict[int, float] = field(default_factory=dict)
    search_tree: SearchTreeNode | None = None # root node
    tree_size: int = 0
    tree_depth: int = 0
    leaf_nodes: int = 0
    branching_factor: float = 0.0
    subgoals_visited: int = 0
    solving_node: SearchTreeNode | None = None
    subgoals_added_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_selected_for_expansion: dict[int, int] = field(default_factory=dict)
    dead_ends_rate: float = 0.0
    
class FinishReason(StrEnum):
    BUDGET_EXCEEDED = 'budget_exceeded'
    NOTHING_TO_EXPAND = 'nothing_to_expand'
    SOLVED = 'solved'

@dataclass
class Experience:
    solution: Solution
    search_info: SearchInfo


class Planner:
    """
    General Planner class.

    Manages the search tree, selects consecutive nodes to expand and returns
    the final solution once the problem is solved.
    """
    @abstractmethod
    def __init__(self, root_state: np.ndarray) -> None:
        self.root_state = root_state

    @abstractmethod
    def add(self, node: SearchTreeNode) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get(self) -> SearchTreeNode | None:
        raise NotImplementedError()

    @abstractmethod
    def is_seen(self, state: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_solution_data(self,
                          solving_node: SearchTreeNode | None,
                          search_info: SearchInfo) -> Experience:
        raise NotImplementedError()

def get_branching_factor(tree_size: int, leaf_nodes: int) -> float:
    """Calculates average branching factor (excluding leaf nodes)"""
    if tree_size == leaf_nodes: # all nodes are leaf nodes i.e. single node tree
        return 0.0
    if tree_size <= 1: # no branching possible
        return 0.0
    return max((tree_size - 1) / (tree_size - leaf_nodes), 0)

def get_tree_info(root_node: SearchTreeNode | None, search_info: dict):
    # Initialize statistics
    tree_size = search_info["low_level_nodes_visited"]
    max_depth = 0
    leaf_nodes = 0
    total_children = 0

    # Queue for breadth-first traversal (node, depth)
    queue_ = [(root_node, 0)]

    while queue_:
        current_node, depth = queue_.pop(0)
        max_depth = max(max_depth, depth)

        if not current_node.children:    # Leaf node
            leaf_nodes += 1
        else:
            total_children += len(current_node.children)
            # Add children to the queue with increased depth
            queue_.extend((child, depth + 1) for child in current_node.children)

    return {
        "tree_size": tree_size,  # low-level nodes visited
        "tree_depth": max_depth,
        "leaf_nodes": leaf_nodes,
        "branching_factor": get_branching_factor(tree_size, leaf_nodes),
    }


def get_dead_end_data(dead_end_finder, seen_states_list, search_info):
    dead_ends_counter = [0, 0]

    for node in seen_states_list[::-1]:
        state_path = [node.parent_node.state]

        for action in node.low_level_path:
            state_path.append(dead_end_finder.env.next_state(state_path[-1], action))

        for state in state_path[::-1]:
            print('DEAD END SOLVING')
            print(printable_sokoban_state(state))

            is_dead_end = dead_end_finder.check_bfs(state)

            print('is_dead_end:', is_dead_end)
            print()

            dead_ends_counter[is_dead_end] += 1

    if sum(dead_ends_counter) > 0:
        search_info['dead_ends_rate'] = dead_ends_counter[True] / sum(dead_ends_counter)
    else:
        search_info['dead_ends_rate'] = 0


class GreedyPlanner(Planner):
    """Basic planner that always selects the node with somehow formulated priority.
    In default implementation, all priorities are equal thus the planner behaves like breadth-first search."""
    def __init__(self, root_state):
        super().__init__(root_state)

        self.seen_states = set()

        self.create_priority_queue()
        self.root_node = SearchTreeNode(self.root_state, 0, [], None, None, metadata={'depth': 0})
        self.add(self.root_node)

    def create_priority_queue(self):
        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()

    def get_node_priority(self, node: SearchTreeNode) -> float:
        return 0.0

    def add(self, node: SearchTreeNode):
        self.seen_states.add(hashable_state(node.state))

        if node.parent_node is not None:
            depth = node.parent_node.metadata['depth'] + 1
            node.metadata['depth'] = depth

        node_priority = self.get_node_priority(node)
        node.metadata['queue_priority'] = node_priority

        self.nodes_queue.put(data=node, key=node_priority)

    def get(self):
        if self.nodes_queue.empty():
            return None

        return self.nodes_queue.get()

    def is_seen(self, state: np.ndarray | str) -> bool | None:
        return hashable_state(state) in self.seen_states

    def get_solution_data(self, solving_node: SearchTreeNode | None, search_info: SearchInfo) -> Experience:
        search_info.low_level_nodes_visited = len(self.seen_states)
        search_info.search_tree = self.root_node

        for k, v in get_tree_info(self.root_node, search_info).items():
            assert k in search_info.__dict__, f"Key {k} not found in search_info"
            search_info.__dict__[k] = v

        if solving_node is None:
            return Solution(solved=False), search_info

        subgoal_path: list[np.ndarray]
        action_path: list[int]
        subgoal_path, action_path, subgoal_distance_path, _ = get_solving_path_data(solving_node,
                                                                                    include_state_path=False)
        
        solution = Solution(
            solved=True,
            subgoal_path=subgoal_path,
            action_path=action_path,
            subgoal_distance_path=subgoal_distance_path
        )

        return solution, search_info


class BestFSPlanner(GreedyPlanner):
    def get_node_priority(self, node: SearchTreeNode):
        return -node.value


class AstarPlanner(GreedyPlanner):
    def __init__(self, root_state, value_weight=150.0, depth_weight=1.0):
        self.value_weight = value_weight
        self.depth_weight = depth_weight

        super().__init__(root_state)

    def get_node_priority(self, node: SearchTreeNode):
        return -node.value * self.value_weight + node.metadata['depth'] * self.depth_weight


class BfsPlanner(GreedyPlanner):
    def create_priority_queue(self):
        self.nodes_queue = queue.Queue()

    def add(self, node: SearchTreeNode):
        self.seen_states.add(tuple(node.state.flatten()))
        self.nodes_queue.put(node)

    def get_node_priority(self, node: SearchTreeNode) -> float:
        return 0.0    # BFS does not use priority queue


# A subclass that keeps track of dead ends
# To use, inherit from the desired planner
class DeadEndTrackingPlanner(BestFSPlanner):    # may inherit from any planner
    def __init__(self, root_state, dead_end_finder: DeadEndFinder, **kwargs):
        self.dead_end_finder = dead_end_finder
        self.seen_states_list = []

        super().__init__(root_state, **kwargs)

    def add(self, node: SearchTreeNode):
        self.seen_states_list.append(node)

        super().add(node)

    def get_solution_data(self, solving_node: SearchTreeNode | None, search_info: dict):
        get_dead_end_data(self.dead_end_finder, self.seen_states_list, search_info)

        return super().get_solution_data(solving_node, search_info)


class AdasubsPlanner(Planner):
    def __init__(self, root_state: np.ndarray, generators_k_list: list[int], prune_search_trees: bool = False) -> None:
        super().__init__(root_state)
        self.generators_k_list = generators_k_list
        self.seen_states: set[tuple[
            np.ndarray,
        ]] = set()

        root_key = (0, 0)

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

        node = self.nodes_queue.get()
        self.subgoals_selected_for_expansion[node.next_expand_with_k_generator] += 1
        return node

    def is_seen(self, state: np.ndarray | str) -> bool | None:
        if isinstance(state, str):
            return tuple(state) in self.seen_states
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten()) in self.seen_states
        return None

    def add(self, node: SearchTreeNode) -> None:
        if node.next_expand_with_k_generator is not None:
            logger.error('Node during adding to queue should have next_expand_with_k_generator set to None')
            logger.error('Planner decides which k-distances should be expanded, not the search itself')
            raise ValueError('Node during adding to queue should have next_expand_with_k_generator set to None')

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
        search_info.low_level_nodes_visited = len(self.seen_states)

        tree_info = get_tree_info(self.root_node, search_info.__dict__)
        for k, v in tree_info.items():
            if hasattr(search_info, k):
                setattr(search_info, k, v)

        if solving_node is None:
            solution = Solution(solved=False)
            return Experience(solution=solution, search_info=search_info)

        subgoal_path, action_path, subgoal_distance_path, _ = get_solving_path_data(
            solving_node, include_state_path=False)
        solution = Solution(
            solved=True,
            subgoal_path=subgoal_path,
            action_path=action_path,
            subgoal_distance_path=subgoal_distance_path
        )
        return Experience(solution=solution, search_info=search_info)

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

import numpy as np

from carl.solver.nodes import SearchTreeNode

GeneratorIdx = int

@dataclass
class Solution:
    solved: bool
    subgoal_path: list[np.ndarray] | None = None
    action_path: list[int] | None = None
    subgoal_distance_path: list[int] | None = None
    subgoal_values: list[float] | None = None
    
    def __post_init__(self):
        if self.solved and (self.subgoal_path is None or self.action_path is None or self.subgoal_distance_path is None):
            raise ValueError("If solved, subgoal_path, action_path and subgoal_distance_path must be provided.")
        if not self.solved and (self.subgoal_path is not None or self.action_path is not None or self.subgoal_distance_path is not None):
            raise ValueError("If not solved, subgoal_path, action_path and subgoal_distance_path must be None.")

@dataclass
class SearchInfo:
    finished_reason: str
    low_level_nodes_visited: int | None = None
    high_level_nodes_valid: int | None = None
    high_level_nodes_unreachable: int | None = None
    subgoals_reachable_count_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_unreachable_count_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_reachable_rate_per_k: dict[int, float] = field(default_factory=dict)
    search_tree: SearchTreeNode | None = None # root node
    tree_size: int | None = None
    tree_depth: int | None = None
    leaf_nodes: int | None = None
    branching_factor: float | None = None
    subgoals_visited: int | None = None
    solving_node: SearchTreeNode | None = None
    subgoals_added_per_k: dict[int, int] = field(default_factory=dict)
    subgoals_selected_for_expansion: dict[int, int] = field(default_factory=dict)
    dead_ends_rate: float | None = None
    
    @property
    def is_valid_tree_search_info(self) -> bool:
        return (self.low_level_nodes_visited is not None and
                self.tree_size is not None and
                self.tree_depth is not None and
                self.leaf_nodes is not None and
                self.branching_factor is not None)
    
    @property
    def is_valid_subgoal_search_info(self) -> bool:
        return self.is_valid_tree_search_info and \
                (self.high_level_nodes_valid is not None and
                self.high_level_nodes_unreachable is not None and
                len(self.subgoals_reachable_count_per_k) > 0 and
                len(self.subgoals_unreachable_count_per_k) > 0 and
                len(self.subgoals_reachable_rate_per_k) > 0 and
                self.subgoals_visited is not None and
                self.subgoals_added_per_k is not None and
                self.subgoals_selected_for_expansion is not None)

    
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

def get_tree_info(root_node: SearchTreeNode, search_info: SearchInfo):
    # Initialize statistics
    tree_size = search_info.low_level_nodes_visited
    assert tree_size is not None, 'Low level nodes param has to be set.'
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

            is_dead_end = dead_end_finder.check_bfs(state)

            print('is_dead_end:', is_dead_end)
            print()

            dead_ends_counter[is_dead_end] += 1

    if sum(dead_ends_counter) > 0:
        search_info['dead_ends_rate'] = dead_ends_counter[True] / sum(dead_ends_counter)
    else:
        search_info['dead_ends_rate'] = 0


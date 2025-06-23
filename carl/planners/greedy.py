
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

        for k, v in  get_tree_info(self.root_node, search_info).items():
            assert k in search_info.__dict__, f"Key {k} not found in search_info"
            search_info.__dict__[k] = v

        if solving_node is None:
            return Solution(solved=False), search_info

        subgoal_path: list[np.ndarray]
        action_path: list[int]
        subgoal_path, action_path, values, subgoal_distance_path, _ = get_solving_path_data(solving_node,
                                                                                    include_state_path=False)
        
        solution = Solution(
            solved=True,
            subgoal_path=subgoal_path,
            action_path=action_path,
            subgoal_distance_path=subgoal_distance_path,
            subgoal_values=values
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


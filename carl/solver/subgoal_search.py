import sys
from collections.abc import Callable

import numpy as np
from loguru import logger

from carl.inference_components.subgoal_generator import AdaptiveSubgoalGenerator
from carl.inference_components.validator import Validator
from carl.inference_components.value import Value
from carl.planners.base import Experience
from carl.planners.base import FinishReason
from carl.planners.base import Planner
from carl.planners.base import SearchInfo
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import ValidationResult
from carl.utils.aliases import State


def ensure_high_recursion_limit() -> None:
    recursion_limit = 2147483640
    curr_recursion_limit = sys.getrecursionlimit()

    if curr_recursion_limit < recursion_limit:
        print(f'Rising recursion limit for worker from {curr_recursion_limit} to {recursion_limit}.')
        sys.setrecursionlimit(recursion_limit)


class Solver:
    def __init__(
        self,
        max_nodes: int,
        planner_class: Callable[[State], Planner],
        subgoal_generator: AdaptiveSubgoalGenerator,
        validator: Validator,
        value_function: Value,
    ) -> None:

        self.max_nodes = max_nodes
        self.planner_class = planner_class
        self.planner: Planner | None = None
        self.subgoal_generator = subgoal_generator
        self.validator = validator
        self.value_function = value_function

    def construct_networks(self) -> None:
        self.subgoal_generator.construct_network()
        self.validator.construct_network()
        self.value_function.construct_network()

    def solve(self, initial_state: State) -> Experience:
        ensure_high_recursion_limit()

        self.planner = self.planner_class(
            initial_state)    #, dead_end_finder=DeadEndFinder(self.subgoal_generator.env, 4))
        nodes_visited: int = 0
        nodes_valid: int = 0
        nodes_unreachable: int = 0
        solving_node: SearchTreeNode | None = None

        search_info: SearchInfo = SearchInfo(finished_reason=FinishReason.BUDGET_EXCEEDED.value)
        ks = self.subgoal_generator.generator_k_list
        subgoals_reachable_count_per_k: dict[int, int] = {k: 0 for k in ks}
        subgoals_unreachable_count_per_k: dict[int, int] = {k: 0 for k in ks}
        
        while nodes_visited < self.max_nodes and solving_node is None:
            current_node: SearchTreeNode | None = self.planner.get()
            if current_node is None:
                # There is nothing more to expand.
                search_info.finished_reason = FinishReason.NOTHING_TO_EXPAND.value
                break
            subgoals = self.subgoal_generator.get_subgoals(current_node)

            for subgoal, generation_metadata in subgoals:
                if self.planner.is_seen(subgoal):
                    continue

                validation: ValidationResult = self.validator.is_valid(
                    current_node.state,
                    subgoal,
                    steps_limit=current_node.next_expand_with_k_generator,
                )
                nodes_visited += validation.low_level_nodes_visited

                if not validation.is_valid:
                    nodes_unreachable += 1
                    if current_node.next_expand_with_k_generator is not None:
                        subgoals_unreachable_count_per_k[current_node.next_expand_with_k_generator] += 1
                    # Subgoal is invalid, discard it.
                    continue

                valid_subgoal: np.ndarray | str = validation.achieved_state

                nodes_valid += 1
                if current_node.next_expand_with_k_generator is not None:
                    subgoals_reachable_count_per_k[current_node.next_expand_with_k_generator] += 1

                value: float = self.value_function.get_value(valid_subgoal)

                subgoal_node: SearchTreeNode = SearchTreeNode(
                    valid_subgoal,
                    value,
                    validation.path,
                    current_node,
                    metadata=generation_metadata,
                )
                self.planner.add(subgoal_node)

                if validation.is_solved:
                    solving_node = subgoal_node
                    logger.success(f'solved with {nodes_visited} low level nodes visited')
                    search_info.finished_reason = FinishReason.SOLVED.value
                    break
        
        search_info.low_level_nodes_visited = nodes_visited
        search_info.high_level_nodes_valid = nodes_valid
        search_info.high_level_nodes_unreachable = nodes_unreachable

        for k in ks:
            search_info.subgoals_reachable_count_per_k[k] = subgoals_reachable_count_per_k[k]
            search_info.subgoals_unreachable_count_per_k[k] = subgoals_unreachable_count_per_k[k]

            if subgoals_reachable_count_per_k[k] + subgoals_unreachable_count_per_k[k] == 0:
                search_info.subgoals_reachable_rate_per_k[k] = 0
            else:
                rate = subgoals_reachable_count_per_k[k] / (subgoals_reachable_count_per_k[k] +
                                                            subgoals_unreachable_count_per_k[k])
                search_info.subgoals_reachable_rate_per_k[k] = rate

        # The computational budget is over.
        return self.planner.get_solution_data(solving_node, search_info)

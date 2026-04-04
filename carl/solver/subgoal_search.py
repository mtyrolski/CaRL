import sys
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from carl.inference_components.subgoal_generator import SubgoalGenerator
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
        subgoal_generator: AdaptiveSubgoalGenerator | SubgoalGenerator,
        validator: Validator,
        value_function: Value,
    ) -> None:

        self.max_nodes = max_nodes
        self.planner_class = planner_class
        self.planner: Planner | None = None
        self.subgoal_generator = subgoal_generator
        self.validator = validator
        self.value_function = value_function

    @staticmethod
    def _copy_state_for_logging(state: State) -> State:
        if isinstance(state, np.ndarray):
            return state.copy()
        return state

    @staticmethod
    def _hash_state(state: State) -> str:
        if isinstance(state, np.ndarray):
            return str(hash(state.tobytes()))
        return str(hash(state))

    def _mark_reached_proposal_events(self, search_info: SearchInfo, solving_node: SearchTreeNode | None) -> None:
        if solving_node is None or not search_info.proposal_events:
            return
        reached_event_ids: set[int] = set()
        current = solving_node
        while current.parent_node is not None:
            event_id = current.metadata.get("proposal_event_id")
            if isinstance(event_id, int):
                reached_event_ids.add(event_id)
            current = current.parent_node
        for event in search_info.proposal_events:
            event_id = event.get("event_id")
            if isinstance(event_id, int):
                event["reached"] = event_id in reached_event_ids

    def _compute_post_solve_metrics(self, search_info: SearchInfo, solving_node: SearchTreeNode | None) -> None:
        accepted = search_info.proposal_acceptances or 0
        rejected = search_info.proposal_rejections or 0
        denom = accepted + rejected
        search_info.validator_rejection_rate = 0.0 if denom == 0 else rejected / denom
        search_info.proposal_events_count = len(search_info.proposal_events)

        if search_info.proposal_events:
            non_duplicate_events = [e for e in search_info.proposal_events if not bool(e.get("duplicate", False))]
            keys = [str(e.get("proposed_state_hash")) for e in non_duplicate_events if e.get("proposed_state_hash") is not None]
            if keys:
                unique = len(set(keys))
                total = len(keys)
                search_info.proposal_diversity_unique_ratio = unique / total
                counts: dict[str, int] = {}
                for key in keys:
                    counts[key] = counts.get(key, 0) + 1
                probs = [count / total for count in counts.values()]
                search_info.proposal_diversity_entropy = float(-sum(p * math.log(p + 1e-12) for p in probs))
            else:
                search_info.proposal_diversity_unique_ratio = 0.0
                search_info.proposal_diversity_entropy = 0.0

        if solving_node is None:
            return

        segment_lengths: list[int] = []
        segment_budgets: list[int] = []
        current = solving_node
        while current.parent_node is not None:
            low_level_path = current.low_level_path or []
            segment_lengths.append(len(low_level_path))
            if current.next_expand_with_k_generator is not None:
                segment_budgets.append(int(current.next_expand_with_k_generator))
            current = current.parent_node
        segment_lengths.reverse()
        segment_budgets.reverse()

        search_info.realized_segment_lengths = segment_lengths
        search_info.progress_per_segment = [float(x) for x in segment_lengths]
        if segment_lengths:
            search_info.realized_segment_length_mean = float(sum(segment_lengths) / len(segment_lengths))
            search_info.progress_per_segment_mean = float(sum(segment_lengths) / len(segment_lengths))
            # Proxy detour metric: realized segment length divided by planner segment budget where available.
            if segment_budgets and len(segment_budgets) == len(segment_lengths):
                ratios = [seg / max(budget, 1) for seg, budget in zip(segment_lengths, segment_budgets)]
                search_info.detour_ratio = float(sum(ratios) / len(ratios))
            else:
                search_info.detour_ratio = 1.0
        else:
            search_info.realized_segment_length_mean = 0.0
            search_info.progress_per_segment_mean = 0.0
            search_info.detour_ratio = 0.0

        # Reconstruct low-level solving state path to estimate repeated-state backtracking.
        env = getattr(self.validator, "env", None)
        if env is None:
            return
        try:
            from carl.solver.nodes import get_solving_path_data
            _, _, _, _, state_path = get_solving_path_data(solving_node, include_state_path=True, env=env)
        except Exception:
            state_path = None

        if state_path is not None and len(state_path) > 1:
            seen: set[str] = set()
            revisits = 0
            for state in state_path:
                state_key = self._hash_state(state)
                if state_key in seen:
                    revisits += 1
                else:
                    seen.add(state_key)
            search_info.backtracking_ratio = revisits / len(state_path)
        elif state_path is not None:
            search_info.backtracking_ratio = 0.0

    def construct_networks(self) -> None:
        self.subgoal_generator.construct_network()
        self.validator.construct_network()
        self.value_function.construct_network()

    def solve(self, initial_state: State) -> Experience:
        ensure_high_recursion_limit()
        start_time = time.perf_counter()

        self.planner = self.planner_class(
            initial_state)    #, dead_end_finder=DeadEndFinder(self.subgoal_generator.env, 4))
        nodes_visited: int = 0
        nodes_valid: int = 0
        nodes_unreachable: int = 0
        solving_node: SearchTreeNode | None = None

        mode = str(getattr(self.subgoal_generator, "mode", "bank"))
        search_info: SearchInfo = SearchInfo(
            finished_reason=FinishReason.BUDGET_EXCEEDED.value,
            generator_mode=mode,
        )
        ks = list(getattr(self.subgoal_generator, "generator_k_list", []))
        subgoals_reachable_count_per_k: dict[int, int] = {k: 0 for k in ks}
        subgoals_unreachable_count_per_k: dict[int, int] = {k: 0 for k in ks}
        proposal_event_id = 0
        proposal_duplicates = 0
        proposal_acceptances = 0
        proposal_rejections = 0
        
        while nodes_visited < self.max_nodes and solving_node is None:
            current_node: SearchTreeNode | None = self.planner.get()
            if current_node is None:
                # There is nothing more to expand.
                search_info.finished_reason = FinishReason.NOTHING_TO_EXPAND.value
                break
            subgoals = self.subgoal_generator.get_subgoals(current_node)

            for subgoal, generation_metadata in subgoals:
                event: dict[str, Any] = {
                    "event_id": proposal_event_id,
                    "current_state": self._copy_state_for_logging(current_node.state),
                    "proposed_state": self._copy_state_for_logging(subgoal),
                    "current_state_hash": self._hash_state(current_node.state),
                    "proposed_state_hash": self._hash_state(subgoal),
                    "generator_mode": mode,
                    "steps_limit": current_node.next_expand_with_k_generator,
                    "validator_accept": False,
                    "validator_reject": False,
                    "reached": False,
                    "duplicate": False,
                }
                if isinstance(generation_metadata, dict):
                    for key, value in generation_metadata.items():
                        if key in {"proposal_rank", "proposal_confidence", "node_depth", "proposal_duplicate_in_decode"}:
                            event[key] = value
                proposal_event_id += 1
                if self.planner.is_seen(subgoal):
                    event["duplicate"] = True
                    search_info.proposal_events.append(event)
                    proposal_duplicates += 1
                    continue

                validation: ValidationResult = self.validator.is_valid(
                    current_node.state,
                    subgoal,
                    steps_limit=current_node.next_expand_with_k_generator,
                )
                nodes_visited += validation.low_level_nodes_visited
                event["validator_low_level_nodes_visited"] = validation.low_level_nodes_visited
                event["validator_is_solved_subgoal"] = validation.is_solved
                event["achieved_state"] = self._copy_state_for_logging(validation.achieved_state)

                if not validation.is_valid:
                    nodes_unreachable += 1
                    proposal_rejections += 1
                    event["validator_reject"] = True
                    if current_node.next_expand_with_k_generator is not None:
                        subgoals_unreachable_count_per_k[current_node.next_expand_with_k_generator] += 1
                    search_info.proposal_events.append(event)
                    # Subgoal is invalid, discard it.
                    continue

                valid_subgoal: np.ndarray | str = validation.achieved_state

                nodes_valid += 1
                proposal_acceptances += 1
                event["validator_accept"] = True
                if current_node.next_expand_with_k_generator is not None:
                    subgoals_reachable_count_per_k[current_node.next_expand_with_k_generator] += 1

                value: float = self.value_function.get_value(valid_subgoal)

                metadata = dict(generation_metadata) if isinstance(generation_metadata, dict) else {}
                metadata["proposal_event_id"] = event["event_id"]
                metadata["generator_mode"] = mode
                subgoal_node: SearchTreeNode = SearchTreeNode(
                    valid_subgoal,
                    value,
                    validation.path,
                    current_node,
                    metadata=metadata,
                )
                event["accepted_value"] = value
                search_info.proposal_events.append(event)
                self.planner.add(subgoal_node)

                if validation.is_solved:
                    solving_node = subgoal_node
                    logger.success(f'solved with {nodes_visited} low level nodes visited')
                    search_info.finished_reason = FinishReason.SOLVED.value
                    break

        search_info.low_level_nodes_visited = nodes_visited
        search_info.high_level_nodes_valid = nodes_valid
        search_info.high_level_nodes_unreachable = nodes_unreachable
        search_info.proposal_acceptances = proposal_acceptances
        search_info.proposal_rejections = proposal_rejections
        search_info.proposal_duplicates = proposal_duplicates

        for k in ks:
            search_info.subgoals_reachable_count_per_k[k] = subgoals_reachable_count_per_k[k]
            search_info.subgoals_unreachable_count_per_k[k] = subgoals_unreachable_count_per_k[k]

            if subgoals_reachable_count_per_k[k] + subgoals_unreachable_count_per_k[k] == 0:
                search_info.subgoals_reachable_rate_per_k[k] = 0
            else:
                rate = subgoals_reachable_count_per_k[k] / (subgoals_reachable_count_per_k[k] +
                                                            subgoals_unreachable_count_per_k[k])
                search_info.subgoals_reachable_rate_per_k[k] = rate

        search_info.runtime_seconds = time.perf_counter() - start_time
        search_info.solving_node = solving_node
        self._mark_reached_proposal_events(search_info, solving_node)
        self._compute_post_solve_metrics(search_info, solving_node)

        # The computational budget is over.
        return self.planner.get_solution_data(solving_node, search_info)

import os
from abc import ABC
from abc import abstractmethod
from os.path import join
from typing import Any

from loguru import logger

from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.utils.loggers import NeptuneCaRLLogger
from carl.utils.metric_logging import MetricsAccumulator

SOLVED_PREFIX = 'solved_instances'
ALL_PREFIX = 'all_instances'
AVG_PREFIX = 'average'

class ResultLogger(ABC):
    @abstractmethod
    def log_results(self, results: Any) -> None:
        raise NotImplementedError

    @property
    def node_global_id(self) -> str:
        """
        Support for heterogeneous jobs.
        """
        het_group_id = os.environ.get('CARL_HET_GROUP_ID', None)
        local_worker_id = os.environ.get('CARL_LOCAL_WORKER_ID', None)

        if het_group_id is None or local_worker_id is None:
            logger.debug('No HET_GROUP_ID or LOCAL_WORKER_ID found in the environment.')
            return ''

        return f'HG{het_group_id}_W{local_worker_id}'


class SubgoalSearchResultLogger(ResultLogger):
    def __init__(self, custom_logger: NeptuneCaRLLogger, budget_logs: list[int], problem_to_solve: int) -> None:
        self.custom_logger = custom_logger.return_logger()
        if not hasattr(self.custom_logger, 'run') or self.custom_logger.run is None or not hasattr(self.custom_logger.run, '__getitem__'):
            raise RuntimeError("Custom logger is not properly initialized: 'run' attribute is missing or not subscriptable.")
        self.completed_problems: int = 0
        self.solved_problems: int = 0
        self.problem_to_solve = problem_to_solve
        self.finished_reasons: dict[str, int] = {}
        self.budget_logs = budget_logs
        self.total_completed_problems = 0
        self.solved_stats = MetricsAccumulator()

    def log_results(self, results: list[Experience]) -> None:
        base_completed_problems = self.completed_problems
        self.completed_problems += len(results)
        self.neptune_run['total_completed_problems'].append(
            step=self.completed_problems, value=self.completed_problems)

        for task_id, experience in enumerate(results):
            solution = experience.solution
            search_info = experience.search_info
            self.solved_problems += int(solution.solved)
            for instance_log_fn in [
                self._log_solved_rate,
                self._log_solved_instance_metrics,
                self._log_general_metrics,
                self._log_budget_solved_rates,
                self._count_finished_reason
            ]:
                instance_log_fn(base_completed_problems, task_id, solution, search_info)
        self._log_final_solved_rates()
        self._log_final_finished_reasons()

    @property
    def neptune_run(self):
        if not hasattr(self.custom_logger, 'run') or self.custom_logger.run is None or not hasattr(self.custom_logger.run, '__getitem__'):
            raise RuntimeError("Custom logger is not properly initialized: 'run' attribute is missing or not subscriptable.")
        return self.custom_logger.run

    def _log_solved_rate(self, base_completed_problems: int, task_id: int, solution: Solution, search_info: SearchInfo):
        self.solved_stats.log_metric_to_average('rate/full', int(solution.solved))

    def _get_solution_lengths(self, solution: Solution):
        if not solution.solved:
            return 0, 0
        low_level_len = len(solution.action_path) if solution.action_path is not None else 0
        high_level_len = len(solution.subgoal_path) if solution.subgoal_path is not None else 0
        return low_level_len, high_level_len

    def _log_solved_instance_metrics(self, base_completed_problems: int, task_id: int, solution: Solution, search_info: SearchInfo):
        # Logs metrics only for solved instances
        low_level_len, high_level_len = self._get_solution_lengths(solution)
        if not solution.solved:
            return
        self.neptune_run[f'{SOLVED_PREFIX}__low_level_solution_len'].append(
            step=base_completed_problems + task_id, value=low_level_len + 1)
        self.neptune_run[f'{SOLVED_PREFIX}__high_level_solution_len'].append(
            step=base_completed_problems + task_id, value=high_level_len + 1)
        self.neptune_run[f'{SOLVED_PREFIX}__tree_size'].append(
            step=base_completed_problems + task_id, value=search_info.tree_size)
        self.neptune_run[f'{SOLVED_PREFIX}__tree_depth'].append(
            step=base_completed_problems + task_id, value=search_info.tree_depth)
        self.neptune_run[f'{SOLVED_PREFIX}__leaf_nodes'].append(
            step=base_completed_problems + task_id, value=search_info.leaf_nodes)
        self.neptune_run[f'{SOLVED_PREFIX}__branching_factor'].append(
            step=base_completed_problems + task_id, value=search_info.branching_factor)
        
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__low_level_solution_len', low_level_len + 1)
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__high_level_solution_len', high_level_len + 1)
        
        assert search_info.is_valid_tree_search_info, "SearchInfo must be valid for solved instance metrics logging"
        
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__tree_size', search_info.tree_size) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__tree_depth', search_info.tree_depth) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__leaf_nodes', search_info.leaf_nodes) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{SOLVED_PREFIX}__branching_factor', search_info.branching_factor) # type: ignore

        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__validator_rejection_rate',
                                           search_info.validator_rejection_rate, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__realized_segment_len_mean',
                                           search_info.realized_segment_length_mean, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__progress_per_segment_mean',
                                           search_info.progress_per_segment_mean, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__backtracking_ratio',
                                           search_info.backtracking_ratio, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__detour_ratio',
                                           search_info.detour_ratio, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__proposal_diversity_unique_ratio',
                                           search_info.proposal_diversity_unique_ratio, solved_only=True)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'solved_instances__proposal_diversity_entropy',
                                           search_info.proposal_diversity_entropy, solved_only=True)


    def _log_general_metrics(self, base_completed_problems: int, task_id: int, solution: Solution, search_info: SearchInfo):
        assert search_info.is_valid_tree_search_info
        self.neptune_run[f'{ALL_PREFIX}__tree_size'].append(
            step=base_completed_problems + task_id, value=search_info.tree_size)
        self.neptune_run[f'{ALL_PREFIX}__tree_depth'].append(
            step=base_completed_problems + task_id, value=search_info.tree_depth)
        self.neptune_run[f'{ALL_PREFIX}__leaf_nodes'].append(
            step=base_completed_problems + task_id, value=search_info.leaf_nodes)
        self.neptune_run[f'{ALL_PREFIX}__branching_factor'].append(
            step=base_completed_problems + task_id, value=search_info.branching_factor)

        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{ALL_PREFIX}__tree_size', search_info.tree_size) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{ALL_PREFIX}__tree_depth', search_info.tree_depth) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{ALL_PREFIX}__leaf_nodes', search_info.leaf_nodes) # type: ignore
        self.solved_stats.log_metric_to_average(f'{AVG_PREFIX}__{ALL_PREFIX}__branching_factor', search_info.branching_factor) # type: ignore

        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__runtime_seconds',
                                           search_info.runtime_seconds)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__validator_rejection_rate',
                                           search_info.validator_rejection_rate)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__proposal_events_count',
                                           search_info.proposal_events_count)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__proposal_duplicates',
                                           search_info.proposal_duplicates)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__proposal_diversity_unique_ratio',
                                           search_info.proposal_diversity_unique_ratio)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__proposal_diversity_entropy',
                                           search_info.proposal_diversity_entropy)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__backtracking_ratio',
                                           search_info.backtracking_ratio)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__realized_segment_len_mean',
                                           search_info.realized_segment_length_mean)
        self._log_optional_instance_metric(base_completed_problems, task_id, 'all_instances__detour_ratio',
                                           search_info.detour_ratio)

    def _log_budget_solved_rates(self, base_completed_problems: int, task_id: int, solution: Solution, search_info: SearchInfo):
        for budget in self.budget_logs:
            is_solved_in_budget = solution.solved and (search_info.low_level_nodes_visited is not None and search_info.low_level_nodes_visited <= budget)
            self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', is_solved_in_budget)
            self.solved_stats.log_metric_to_average(f'rate/{budget}_subgoals', solution.solved and (search_info.subgoals_visited is not None and search_info.subgoals_visited <= budget))
            
    def _count_finished_reason(self, base_completed_problems: int, task_id: int, solution: Solution, search_info: SearchInfo):
        finished_reason = search_info.finished_reason
        self.finished_reasons[finished_reason] = self.finished_reasons.get(finished_reason, 0) + 1

    def _log_optional_instance_metric(
        self,
        base_completed_problems: int,
        task_id: int,
        name: str,
        value: Any,
        solved_only: bool = False,
    ) -> None:
        if value is None:
            return
        self.neptune_run[name].append(step=base_completed_problems + task_id, value=value)
        avg_key = f'{AVG_PREFIX}__{name}'
        self.solved_stats.log_metric_to_average(avg_key, value)

    def _log_final_solved_rates(self):
        self.neptune_run[join('problems/solved', self.node_global_id)].append(
            step=self.completed_problems, value=self.solved_problems)
        self.neptune_run[join('solved', self.node_global_id)].append(
            step=self.completed_problems, value=self.solved_stats.return_scalars())

    def _log_final_finished_reasons(self):
        for finished_reason, count in self.finished_reasons.items():
            self.neptune_run[join(f'finished_reasons/{finished_reason}/rate', self.node_global_id)].append(
                step=self.completed_problems, value=count / self.completed_problems)

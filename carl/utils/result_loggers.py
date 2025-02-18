from abc import ABC
from abc import abstractmethod
from os.path import join
from typing import Any
import os
from loguru import logger
import numpy as np
from carl.utils.loggers import NeptuneCaRLLogger
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.utils.metric_logging import MetricsAccumulator


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
            logger.debug('No HET_GROUP_ID or LOCAL_WORKER_ID found in the environment. It be w.')
            return ''

        return f'HG{het_group_id}_W{local_worker_id}'


class SubgoalSearchResultLogger(ResultLogger):
    def __init__(self, custom_logger: NeptuneCaRLLogger, budget_logs: list[int], problem_to_solve: int) -> None:
        # TODO: Check type of custom_logger.
        self.custom_logger = custom_logger.return_logger()
        self.completed_problems: int = 0
        self.solved_problems: int = 0
        self.problem_to_solve = problem_to_solve
        self.finished_reasons: dict[str, int] = {}
        self.budget_logs = budget_logs
        self.total_completed_problems = 0
        self.solved_stats = MetricsAccumulator()

    def log_results(self, results) -> None:
        base_completed_problems = self.completed_problems
        self.completed_problems = self.completed_problems + len(results)
        self.custom_logger.run[join('total_completed_problems',
                                    self.node_global_id)].append(step=self.completed_problems,
                                                                 value=self.completed_problems)
        SokobanEnv(tokenizer=SokobanTokenizer())

        for task_id, (solution, search_info) in enumerate(results):
            # Log the main solved rate metric and the solution.
            self.solved_stats.log_metric_to_average('rate/full', solution['solved'])
            self.solved_problems += solution['solved']
            solution_length: int = 0
            path_length_all_nodes: int = 0
            if solution['solved']:
                self.custom_logger.run['solved_instances_solution_path_length_all_nodes'].append(
                    step=base_completed_problems + task_id, value=path_length_all_nodes + 1)
                self.custom_logger.run['solved_instances_solution_path_length'].append(step=base_completed_problems +
                                                                                       task_id,
                                                                                       value=solution_length)
                self.custom_logger.run['solved_instances_tree_size'].append(step=base_completed_problems + task_id,
                                                                            value=search_info['tree_size'])
                self.custom_logger.run['solved_instances_tree_depth'].append(step=base_completed_problems + task_id,
                                                                             value=search_info['tree_depth'])
                self.custom_logger.run['solved_instances_leaf_nodes'].append(step=base_completed_problems + task_id,
                                                                             value=search_info['leaf_nodes'])
                self.custom_logger.run['solved_instances_branching_factor'].append(
                    step=base_completed_problems + task_id, value=search_info['branching_factor'])

                self.solved_stats.log_metric_to_average("average_solved_instances_solution_path_length",
                                                        solution_length)
                self.solved_stats.log_metric_to_average("average_solved_instances_tree_size", search_info['tree_size'])
                self.solved_stats.log_metric_to_average("average_solved_instances_tree_depth",
                                                        search_info['tree_depth'])
                self.solved_stats.log_metric_to_average("average_solved_instances_leaf_nodes",
                                                        search_info['leaf_nodes'])
                self.solved_stats.log_metric_to_average("average_solved_instances_branching_factor",
                                                        search_info['branching_factor'])
                self.solved_stats.log_metric_to_average("average_solved_instances_solution_path_length_all_nodes",
                                                        path_length_all_nodes)

            self.custom_logger.run['solution_path_length'].append(step=base_completed_problems + task_id,
                                                                  value=solution_length)
            # Accumulate the length of the solution path, tree size, tree depth, leaf nodes, and branching factor.
            self.solved_stats.log_metric_to_average("average_solution_path_length", solution_length)
            self.solved_stats.log_metric_to_average("average_tree_size", search_info['tree_size'])
            self.solved_stats.log_metric_to_average("average_tree_depth", search_info['tree_depth'])
            self.solved_stats.log_metric_to_average("average_leaf_nodes", search_info['leaf_nodes'])
            self.solved_stats.log_metric_to_average("average_branching_factor", search_info['branching_factor'])
            self.solved_stats.log_metric_to_average("average_solved_solution_path_length_all_nodes",
                                                    path_length_all_nodes)

            # Log solved rates for selected budgets.
            for budget in self.budget_logs:
                is_solved_in_budget = solution['solved'] and (search_info['low_level_nodes_visited'] <= budget)
                self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', is_solved_in_budget)

                # # High Level budget
                # is_solved_in_high_level_budget = solution['solved'] and (search_info['subgoals_visited'] <= budget)
                # self.solved_stats.log_metric_to_average(f'rate/{budget}_subgoals', is_solved_in_high_level_budget)

            # Log additional search metrics.
            for metric, value in search_info.items():
                if isinstance(value, int | float):
                    self.custom_logger.run[join(metric,
                                                self.node_global_id)].append(step=base_completed_problems + task_id,
                                                                             value=value)

            # Log fraction of reachable nodes and unreachable nodes.
            # fraction_of_reachable_nodes: float = search_info['nodes_valid'] / (search_info['nodes_valid'] +
            #                                                                    search_info['nodes_unreachable'])
            # fraction_of_unreachable_nodes: float = search_info['nodes_unreachable'] / (search_info['nodes_valid'] +
            #                                                                            search_info['nodes_unreachable'])

            # self.custom_logger.run['fraction_of_reachable_nodes'].append(step=base_completed_problems + task_id,
            #                                                              value=fraction_of_reachable_nodes)
            # self.custom_logger.run['fraction_of_unreachable_nodes'].append(step=base_completed_problems + task_id,
            #                                                                value=fraction_of_unreachable_nodes)

            # Count the finished reasons.
            finished_reason = search_info['finished_reason']
            self.finished_reasons[finished_reason] = (self.finished_reasons.get(finished_reason, 0) + 1)

        # Log the solved rate metrics.
        self.custom_logger.run[join('problems/solved', self.node_global_id)].append(step=self.completed_problems,
                                                                                    value=self.solved_problems)

        self.custom_logger.run[join('solved', self.node_global_id)].append(step=self.completed_problems,
                                                                           value=self.solved_stats.return_scalars())

        # Log the finished reasons.
        for finished_reason, count in self.finished_reasons.items():
            self.custom_logger.run[join(f'finished_reasons/{finished_reason}/rate',
                                        self.node_global_id)].append(step=self.completed_problems,
                                                                     value=count / self.completed_problems)

        if self.completed_problems == self.problem_to_solve:
            for budget in self.budget_logs:
                self.custom_logger.run['success_rate/budget'].append(
                    step=budget, value=self.solved_stats.get_value(f'rate/{budget}_nodes'))
                self.custom_logger.run['success_rate/budget_logscale'].append(
                    step=int(np.log(budget) * 100),
                    value=self.solved_stats.get_value(f'rate/{budget}_nodes'),
                )


class MCTSResultLogger(ResultLogger):
    def __init__(self, custom_logger: NeptuneCaRLLogger, budget_logs: list[int], problem_to_solve: int) -> None:
        self.custom_logger = custom_logger.return_logger()
        self.completed_problems: int = 0
        self.solved_problems: int = 0
        self.problem_to_solve = problem_to_solve
        self.finished_reasons: dict[str, int] = {}
        self.budget_logs = budget_logs
        self.solved_stats: MetricsAccumulator = MetricsAccumulator()

    def log_results(self, results) -> None:
        base_completed_problems = self.completed_problems
        self.completed_problems = self.completed_problems + len(results)
        self.custom_logger.run[join('total_completed_problems',
                                    self.node_global_id)].append(step=self.completed_problems,
                                                                 value=self.completed_problems)

        for task_id, search_info in enumerate(results):

            if search_info['done']:
                self.custom_logger.run["solved_instances_solution_path_length"].append(
                    step=base_completed_problems + task_id, value=search_info['solution_length'])
                self.custom_logger.run['solved_instances_tree_size'].append(step=base_completed_problems + task_id,
                                                                            value=search_info['nodes'])
                self.custom_logger.run['solved_instances_inner_nodes'].append(step=base_completed_problems + task_id,
                                                                              value=search_info['inner_nodes'])
                self.custom_logger.run['solved_instances_leaf_nodes'].append(step=base_completed_problems + task_id,
                                                                             value=search_info['leaves'])
                self.custom_logger.run['solved_instances_branching_factor'].append(
                    step=base_completed_problems + task_id,
                    value=(search_info['nodes'] - 1) / (search_info["nodes"] - search_info["leaves"]))

                self.solved_stats.log_metric_to_average("average_solved_instances_tree_size", search_info['nodes'])
                self.solved_stats.log_metric_to_average("average_solved_instances_inner_nodes",
                                                        search_info['inner_nodes'])
                self.solved_stats.log_metric_to_average("average_solved_instances_leaf_nodes", search_info['leaves'])
                self.solved_stats.log_metric_to_average("average_solved_instances_solution_path_length",
                                                        search_info['solution_length'])

            self.solved_stats.log_metric_to_average("average_solution_path_length", search_info['solution_length'])
            self.solved_stats.log_metric_to_average("average_tree_size", search_info['nodes'])
            self.solved_stats.log_metric_to_average("average_leaf_nodes", search_info['leaves'])
            self.solved_stats.log_metric_to_average("average_inner_nodes", search_info['inner_nodes'])
            self.solved_stats.log_metric_to_average("average_solved_solution_path_length",
                                                    search_info['solution_length'])

            # Log solved rates for selected budgets.
            for budget in self.budget_logs:
                is_solved_in_budget_nodes = search_info['done'] and (search_info['nodes'] <= budget)
                self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', is_solved_in_budget_nodes)
                is_solved_in_budget_leaves = search_info['done'] and (search_info['leaves'] <= budget)
                self.solved_stats.log_metric_to_average(f'rate/{budget}_leaves', is_solved_in_budget_leaves)
                is_solved_in_budget_inner_nodes = search_info['done'] and (search_info['inner_nodes'] <= budget)
                self.solved_stats.log_metric_to_average(f'rate/{budget}_inner_nodes', is_solved_in_budget_inner_nodes)

                self.custom_logger.run[f'rate/{budget}_nodes'].append(
                    step=base_completed_problems + task_id, value=self.solved_stats.get_value(f'rate/{budget}_nodes'))

            for metric, value in search_info.items():
                if isinstance(value, int | float):
                    self.custom_logger.run[join(metric,
                                                self.node_global_id)].append(step=base_completed_problems + task_id,
                                                                             value=value)

        # Log the solved rate metrics.
        self.custom_logger.run[join('problems/solved', self.node_global_id)].append(step=self.completed_problems,
                                                                                    value=self.solved_problems)

        if self.completed_problems == self.problem_to_solve:
            for budget in self.budget_logs:
                self.custom_logger.run['success_rate/budget'].append(
                    step=budget, value=self.solved_stats.get_value(f'rate/{budget}_nodes'))
                self.custom_logger.run['success_rate/budget_logscale'].append(
                    step=int(np.log(budget) * 100),
                    value=self.solved_stats.get_value(f'rate/{budget}_nodes'),
                )
                self.custom_logger.run['success_rate/budget_leaves'].append(
                    step=budget, value=self.solved_stats.get_value(f'rate/{budget}_leaves'))
                self.custom_logger.run['success_rate/budget_logscale_leaves'].append(
                    step=int(np.log(budget) * 100),
                    value=self.solved_stats.get_value(f'rate/{budget}_leaves'),
                )
                self.custom_logger.run['success_rate/budget_inner_nodes'].append(
                    step=budget, value=self.solved_stats.get_value(f'rate/{budget}_inner_nodes'))
                self.custom_logger.run['success_rate/budget_logscale_inner_nodes'].append(
                    step=int(np.log(budget) * 100),
                    value=self.solved_stats.get_value(f'rate/{budget}_inner_nodes'),
                )

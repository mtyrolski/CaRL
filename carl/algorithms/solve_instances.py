import numpy as np
import torch
from carl.environment.instance_generator import BasicInstanceGenerator
from carl.utils.result_loggers import ResultLogger
from carl.solver.subgoal_search import Solver
from joblib import Parallel, delayed, dump
from loguru import logger
from pickle import HIGHEST_PROTOCOL
import os
from carl.env import CUDA_VISIBLE_DEVICES__ENV_VAR

class SolveInstances:
    def __init__(
        self,
        solver: Solver,
        data_loader: BasicInstanceGenerator,
        result_logger: ResultLogger,
        problems_to_solve: int,
        n_parallel_workers: int,
        dump_solved=False,
    ):
        self.solver = solver
        self.data_loader = data_loader
        self.result_logger = result_logger

        self.problems_to_solve = problems_to_solve

        self.completed_problems: int = 0
        self.n_parallel_workers = n_parallel_workers
        self.dump_solved = dump_solved

        if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' and self.n_parallel_workers > 1:
            logger.warning(
                'CUDA_VISIBLE_DEVICES is not set. Reducing n_parallel_workers to 1.'
            )
            # self.n_parallel_workers = 1
        logger.info(f'Using {self.n_parallel_workers} parallel workers')

    def run(self) -> None:
        logger.warning('Running solve_instances.py')

        self.solver.construct_networks()

        all_experiences: list[tuple[dict, dict]] = []

        for batch, problems in enumerate(self.data_loader.reset_dataloader()):
            # TODO: convert to numpy array is not elegant. Change this.
            conv_problems: list[str] | np.ndarray
            if isinstance(problems[0], np.ndarray | torch.Tensor):
                conv_problems: np.ndarray = problems.numpy()

            else:
                conv_problems: list[str] = problems

            results: list[tuple[dict, dict]] = []

            if self.problems_to_solve <= self.completed_problems:
                logger.info(f'problems/completed: {self.completed_problems}')
                break

            if self.n_parallel_workers == 1:
                for num, problem in enumerate(conv_problems):
                    logger.info(f'Batches: {batch + 1}, Problem: {num + 1} of {len(conv_problems)}')
                    result: tuple[dict, dict] = self.solver.solve(problem)
                    results.append(result)
                    self.completed_problems += 1
            else:
                logger.info(f'Batches: {batch + 1}, Problems: {len(conv_problems)}')
                results = Parallel(n_jobs=self.n_parallel_workers, verbose=100)(
                    delayed(self.solver.solve)(problem) for problem in conv_problems)

            self.result_logger.log_results(results)
            all_experiences.append(results)

        if self.dump_solved:
            dump(all_experiences, 'solved_problems.joblib', protocol=HIGHEST_PROTOCOL)

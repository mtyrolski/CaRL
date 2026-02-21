import os
import shutil
import time
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import EvalPrediction
from transformers import PreTrainedModel
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments

from carl.algorithms import CARL_ALL_NODES_COUNT
from carl.algorithms import CARL_WORKER_LOCAL_ID
from carl.algorithms.algorithm import Algorithm
from carl.dataloader.game_data_module import UntokenizedTrajectory
from carl.dataloader.game_dataset import GameDataset
from carl.environment.env import GameEnv
from carl.environment.instance_generator import BasicInstanceGenerator
from carl.environment.instance_generator import GeneralIterableDataLoader
from carl.environment.instance_generator import InstanceGenerator
from carl.inference_components.component import InferenceComponent
from carl.inference_components.component import TrainingModule
from carl.inference_components.conditional_low_level_policy import TransformerConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import AdaptiveSubgoalGenerator
from carl.inference_components.subgoal_generator import TransformerSubgoalGenerator
from carl.inference_components.value import TransformerValue
from carl.memory.replay_buffer import OfflineReplayBuffer
from carl.memory.replay_buffer import SolvingPathGeneratorReplayBuffer
from carl.memory.replay_buffer import SimpleUniversalReplayBuffer
from carl.planners.base import Experience
from carl.planners.base import Solution
from carl.solver.nodes import prune_experiences
from carl.solver.subgoal_search import Solver
from carl.utils.resources import dump_resource
from carl.utils.resources import exists_resource
from carl.utils.resources import get_latest_file
from carl.utils.resources import read_resource_and_delete
from carl.utils.resources import stop_signal
from carl.utils.result_loggers import SubgoalSearchResultLogger
from carl.utils.training_metrics import MetricsHF


def solved_count(results: list[Solution]) -> int:
    return sum(int(x.solved) for x in results)


class TrainingLoopHF(Algorithm):
    """
    A training loop for hierarchical reinforcement learning framework, integrating various components and strategies.

    Args:
        n_iterations (int): Number of iterations to run the training loop.
        evaluate_every (int): Frequency of evaluation, i.e., perform evaluation after every 'evaluate_every' iterations.
        problem_to_evaluate (int): Number of problems to evaluate in each evaluation phase.
        solver (Solver): Solver containing the inference components.
        env (GameEnv): Environment to solve.
        replay_buffer (SimpleUniversalReplayBuffer): Replay buffer for storing experiences.
        generated_data_path (str): Path to the generated data for instance generation.
        instance_generator_batch_size (int): Batch size for the instance generator.
        eval_data_path (str): Path to the evaluation data.
        eval_batch_size (int): Batch size for evaluation.
        weights_dump_path (str): Path to dump the weights of the components.
        custom_logger (NeptuneCaRLLogger): Custom logger for logging, currently only NeptuneCaRLLogger is supported.
        replay_buffer_data_split (float): Split ratio of the replay buffer data for training and validation.
        path_to_data_used_for_testing_cllp (str | None, optional): Path to the data used for
            testing the conditional low-level policy.
        number_of_trajectories_to_test_cllp (int | None, optional): Number of trajectories to
            test the conditional low-level policy.
        path_to_data_used_for_components_training (str | None, optional):
            Path to the data used for training components.
        initialization_buffer_with_data_used_for_components_training (bool):
            Flag to initialize the replay buffers with data used for training components.
        subgoal_generator_trainer_class (type[HFTrainer] | None, optional):
            Class of the trainer for the subgoal generator.
        subgoal_generator_trainer_args (Callable[..., TrainingArguments] | None, optional):
            Arguments for the subgoal generator trainer.
        metric_for_subgoal_generator (MetricsHF | None, optional): Metric to use for the subgoal generator.
        value_function_trainer_class (type[HFTrainer] | None, optional): Class of the trainer for the value function.
        value_function_trainer_args (Callable[..., TrainingArguments] | None, optional): Arguments
            for the value function trainer.
        metric_for_value_function (MetricsHF | None, optional): Metric to use for the value function.
        cllp_trainer_class (type[HFTrainer] | None, optional): Class of the trainer for the conditional
        low-level policy.
        cllp_trainer_args (Callable[..., TrainingArguments] | None, optional): Arguments for
            the conditional low-level policy trainer.
        metric_for_cllp (MetricsHF | None, optional): Metric to use for the conditional low-level policy.
        add_to_replay_buffer (bool, optional): Flag to add new experiences to the replay buffer during training.
        train_first (bool): Flag if loop should train first or solve first.

    The class integrates various components such as instance generators, replay buffers, and trainers for different
    components in the hierarchical framework. It also handles logging, evaluation, and data preparation for
    training/testing.
    """
    def __init__(
        self,
        # General Loop Flow Parameters
        n_iterations: int,
        evaluate_every: int,
        problem_to_evaluate: int,
        min_count_of_new_solved_boards_to_next_train_iteration: int,
        limit_of_data_used_for_components_training: int,

        # Env-Related Parameters.
        solver: Solver,    # Contains the inference components.
        env: GameEnv,
        replay_buffer: SimpleUniversalReplayBuffer,

        # Instance Generator Parameters
        generated_data_path: str,
        instance_generator_batch_size: int,

        # Component training and testing paramaters
        eval_data_path: str,
        eval_batch_size: int,
        weights_dump_path: str,
        result_logger: SubgoalSearchResultLogger,
        replay_buffer_data_split: float,
        path_to_data_used_for_testing_cllp: str | None = None,
        number_of_trajectories_to_test_cllp: int | None = None,
        path_to_data_used_for_components_training: str | None = None,
        initialization_buffer_with_data_used_for_components_training: bool = False,
        workdir_to_import_weights_and_experiences_from: str | None = None,
        add_to_replay_buffer: bool = True,
        n_jobs: int = 8,    # Number of jobs for parallelization in sequential CPU solver. Ignored for batched solver.
        n_jobs_factor_wrt_cores: float | None = None,    # Factor to determine the number of jobs for parallelization in sequential CPU solver. Ignored for batched solver.
        train_first: bool = False,
    ):
        super().__init__()

        logger.warning('Only NeptuneCaRLLogger is supported as a custom logger.')

        self.n_iterations = n_iterations
        self.evaluate_every = evaluate_every
        self.problem_to_evaluate = problem_to_evaluate
        self.env = env
        self.weights_dump_path = weights_dump_path
        self.solver = solver
        self.n_jobs = n_jobs
        self.n_jobs_factor_wrt_cores = n_jobs_factor_wrt_cores
        self.min_count_of_new_solved_boards_to_next_train_iteration = (
            min_count_of_new_solved_boards_to_next_train_iteration)
        self.solved_boards_since_last_training = 0
        self.limit_of_data_used_for_components_training = limit_of_data_used_for_components_training
        self.workdir_to_import_weights_and_experiences_from = workdir_to_import_weights_and_experiences_from
        self.random_int = int(time.time())
        self.train_first = train_first

        if self.n_jobs_factor_wrt_cores is not None:
            n_cores = os.cpu_count()
            if n_cores is None:
                logger.warning('Could not detect CPU core count; keeping default n_jobs.')
            else:
                logger.info(
                    f'Detected {n_cores} cores. Setting n_jobs to {int(n_cores * self.n_jobs_factor_wrt_cores)}')
                self.n_jobs = int(n_cores * self.n_jobs_factor_wrt_cores)

        logger.info(f'Number of jobs for parallelization in sequential CPU solver: {self.n_jobs}')

        self.components: dict[str, InferenceComponent] = {
            'generator': solver.subgoal_generator,
            'value': solver.value_function,
            'cllp': solver.validator.cllp,
        }

        self.path_to_data_used_for_components_training = path_to_data_used_for_components_training
        self.initialization_buffer_with_data_used_for_components_training = (
            initialization_buffer_with_data_used_for_components_training)

        if self.initialization_buffer_with_data_used_for_components_training:
            assert self.path_to_data_used_for_components_training is not None, (
                'Path to data used for components training must be provided if initialization buffer with data used '
                'for components training is set to True.')
            self.path_to_data_for_init_buffers: Path = Path(self.path_to_data_used_for_components_training)

        self.replay_buffer = replay_buffer

        self.replay_buffer_data_split = replay_buffer_data_split

        self.instance_generator: InstanceGenerator = BasicInstanceGenerator(
            GeneralIterableDataLoader(generated_data_path), instance_generator_batch_size)
        self.eval_instance_generator: InstanceGenerator = BasicInstanceGenerator(
            GeneralIterableDataLoader(eval_data_path), eval_batch_size)

        if not any(component.is_trainable() for component in self.components.values()):
            raise ValueError('At least one trainer must be specified')

        self.result_logger = result_logger
        self.neptune_callback = self.result_logger.custom_logger
        run = self.neptune_callback.run
        if run is None:
            raise RuntimeError('Neptune logger is not initialized.')
        self.neptune_run: Any = run

        self.path_to_data_used_for_testing_cllp = path_to_data_used_for_testing_cllp
        self.data_to_test_cllp: list[UntokenizedTrajectory] | None = None
        self.number_of_trajectories_to_test_cllp = number_of_trajectories_to_test_cllp
        self.add_to_replay_buffer = add_to_replay_buffer

    @staticmethod
    def get_stats_after_one_iteration(stats: list[dict[str, float | int]]) -> dict[str, float | int]:

        stats_dict: dict[str, list[tuple[float | int, float | int]]] = {}
        stats_after_one_iteration: dict[str, float | int] = {}

        for d in stats:
            step = d.get('step')
            if step is None:
                continue
            for k, v in d.items():
                if k in ('step', 'epoch'):
                    continue
                stats_dict.setdefault(k, []).append((step, v))

        for k, v in stats_dict.items():
            stats_after_one_iteration[f'after_iterations_{k}'] = sorted(v, key=lambda x: x[1])[0][1]

        return stats_after_one_iteration

    def log_stats_to_neptune_logger(
        self,
        stats: list[dict[str, float | int]],
        iteration: int,
        component_name: str,
        buffer_length: int,
    ) -> None:
        stats_after_one_iteration: dict[str, float | int]

        stats_after_one_iteration = self.get_stats_after_one_iteration(stats)

        for k, v in stats_after_one_iteration.items():
            k = component_name + '_' + k
            self.neptune_run[k].append(v)

        self.neptune_run[component_name + '_buffer_length'].append(buffer_length)

    # TODO: change this to extract_components_for_training i zrobic iteracje po tym
    def prepare_component_for_training(self,
                                       name: str) -> tuple[PreTrainedModel | None, OfflineReplayBuffer | None]:
        model: PreTrainedModel | None = None
        buffer: OfflineReplayBuffer | None = None

        if name.startswith('generator'):
            model_storage = self.solver.subgoal_generator.get_network()
            if isinstance(model_storage, PreTrainedModel):
                model = model_storage
                buffer_candidate = self.replay_buffer.get_buffer_for_generator(None)
                if isinstance(buffer_candidate, dict):
                    raise RuntimeError('Expected a single generator buffer for non-adaptive generator.')
                buffer = cast(OfflineReplayBuffer, buffer_candidate)
            else:
                assert isinstance(model_storage, dict)
                k = int(name.split('/')[-1])
                logger.info(f'Using adaptive generator with k={k}')
                model = model_storage[k]
                buffer = cast(OfflineReplayBuffer, self.replay_buffer.get_buffer_for_generator(k))
        if name == 'value':
            model_storage = self.solver.value_function.get_network()
            assert isinstance(model_storage, PreTrainedModel)
            model = model_storage
            buffer = self.replay_buffer.get_buffer_for_value()
        if name == 'cllp':
            model_storage = self.solver.validator.get_network()
            assert isinstance(model_storage, PreTrainedModel)
            model = model_storage
            buffer = self.replay_buffer.get_buffer_for_policy()

        return model, buffer

    def prepare_data_for_testing_cllp(self) -> None:
        data: list[UntokenizedTrajectory] = []
        temp: dict[int, UntokenizedTrajectory] = {}
        assert self.path_to_data_used_for_testing_cllp is not None
        path_to_data: Path = Path(self.path_to_data_used_for_testing_cllp)

        for file in tqdm(path_to_data.iterdir()):
            logger.info(f'Loading data from file {file} to test cllp.')
            part_dict: dict[int, UntokenizedTrajectory] = joblib.load(file)
            temp.update(part_dict)

        for _, trajectory in temp.items():
            if not trajectory:
                continue
            if isinstance(trajectory[0], np.ndarray):
                data.append(trajectory)
            else:
                logger.warning('Skipping non-array trajectory when preparing CLLP testing data.')

        if self.number_of_trajectories_to_test_cllp is None:
            self.data_to_test_cllp = data
        else:
            self.data_to_test_cllp = data[:self.number_of_trajectories_to_test_cllp]

    def initialize_buffers_with_training_data(self) -> None:
        logger.info('Initializing buffers with data used for components training')
        path_to_data: Path = self.path_to_data_for_init_buffers

        if not os.path.exists(path_to_data):
            logger.warning(f'Path {path_to_data} does not exist. Skipping.')
            logger.warning('Trying to fallback locally')
            path_to_data = Path(os.path.basename(path_to_data))
            if not os.path.exists(path_to_data):
                logger.error(f'Path {path_to_data} does not exist. Skipping.')
                return
        left = self.limit_of_data_used_for_components_training
        fs = os.listdir(path_to_data)
        for f in fs:
            if left <= 0:
                break

            logger.info(f'Loading data from file {f} to initialize buffers.')
            data: dict = joblib.load(os.path.join(path_to_data, f))
            trajectories = list(data.values())

            if len(trajectories) > left:
                trajectories = trajectories[:left]
                left = 0
            else:
                left -= len(trajectories)

            logger.info(f'Adding {len(trajectories)} trajectories to replay buffer.')
            self.replay_buffer.add_from_trajectories(trajectories)

            # Logging buffer sizes
            self.neptune_run['replay_buffer_len_after_filling_buffer/value'].append(
                len(self.replay_buffer.value_buffer.buffer))

            self.neptune_run['replay_buffer_len_after_filling_buffer/cllp'].append(
                len(self.replay_buffer.cllp_buffer.buffer))

            if isinstance(self.replay_buffer.generator_buffer, dict):
                for k, v in self.replay_buffer.generator_buffer.items():
                    self.neptune_run[f'replay_buffer_len_after_filling_buffer/generator/{k}'].append(
                        len(v.buffer))
            else:
                generator_buffer = cast(OfflineReplayBuffer, self.replay_buffer.generator_buffer)
                self.neptune_run['replay_buffer_len_after_filling_buffer/generator'].append(
                    len(generator_buffer.buffer))

    def prepare_data_for_training_component(
        self,
        data_x_y: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[GameDataset, GameDataset]:
        train_data_x_y: list[tuple[torch.Tensor, torch.Tensor]]
        validation_data_x_y: list[tuple[torch.Tensor, torch.Tensor]]

        train_data_x_y, validation_data_x_y = train_test_split(data_x_y, test_size=self.replay_buffer_data_split)

        x_train: torch.Tensor = torch.cat([x for x, _ in train_data_x_y])
        y_train: torch.Tensor = torch.cat([y for _, y in train_data_x_y])
        x_validation: torch.Tensor = torch.cat([x for x, _ in validation_data_x_y])
        y_validation: torch.Tensor = torch.cat([y for _, y in validation_data_x_y])

        train_dataset: GameDataset = GameDataset(x_train, y_train)
        validation_dataset: GameDataset = GameDataset(x_validation, y_validation)

        return train_dataset, validation_dataset

    def solve_one_rl_iteration(self, batch: list[np.ndarray], iteration: int) -> list[Experience]:
        num_solved_in_batch: float = 0.0
        logger.info(f'Solving {len(batch)} boards')
        logger.info(f'SolveOneIteration call. Iteration: {iteration}; Batch size: {len(batch)};')

        time_start = time.time()
        timeout = 99999
        outputs_all_at_once = list(
            joblib.Parallel(n_jobs=self.n_jobs, verbose=100, timeout=timeout)(
                joblib.delayed(self.solver.solve)(initial_state) for initial_state in batch
            )
        )
        outputs_all_at_once = cast(list[Experience], outputs_all_at_once)

        self.result_logger.log_results(outputs_all_at_once)

        time_end = time.time()

        for new_solution in outputs_all_at_once:
            num_solved_in_batch += int(new_solution.solution.solved)
            if self.add_to_replay_buffer:
                self.replay_buffer.add(new_solution)

        num_solved_in_batch_rate = num_solved_in_batch / len(batch)

        self.neptune_run['num_solved_in_batch_during_one_rlloop'].append(num_solved_in_batch_rate)
        self.neptune_run['num_solved_in_batch_during_one_rlloop_count'].append(num_solved_in_batch)
        self.neptune_run['time_batch_per_iteration'].append(time_end - time_start)

        return outputs_all_at_once

    def test_cllp_on_trajectory(self, trajectory: UntokenizedTrajectory) -> float:

        if not trajectory or not isinstance(trajectory[0], np.ndarray):
            return 0.0
        trajectory_arrays = cast(list[np.ndarray], trajectory)
        trajectory_length: int = len(trajectory_arrays)
        if isinstance(self.replay_buffer.generator_buffer, dict):
            if not self.replay_buffer.generator_buffer:
                return 0.0
            distance_range = next(iter(self.replay_buffer.generator_buffer.values())).distance_range
        elif isinstance(self.replay_buffer.generator_buffer, SolvingPathGeneratorReplayBuffer):
            distance_range = self.replay_buffer.generator_buffer.distance_range
        else:
            raise RuntimeError('Unexpected generator buffer type for CLLP testing.')
        cllp_achieved_goals: int = 0

        for i in range(trajectory_length - 1):
            for dist in distance_range:
                inner_dist: int = min(dist, trajectory_length - 1 - i)
                x: np.ndarray
                y: np.ndarray

                x = trajectory_arrays[i]
                y = trajectory_arrays[i + inner_dist]

                cllp_achieved_goals += int(self.solver.validator.is_valid(x, y).is_valid)

                if i + inner_dist >= len(trajectory_arrays) - 1:
                    # don't add more than one copy of the last subgoal
                    break

        mean_cllp_achieved_goals: float = cllp_achieved_goals / trajectory_length

        return mean_cllp_achieved_goals

    def test_cllp(self, iteration: int) -> None:
        cllp_achieved_goals: float = 0.0

        if self.data_to_test_cllp is None:
            logger.warning('No data available to test CLLP.')
            return

        for trajectory in tqdm(self.data_to_test_cllp):
            cllp_achieved_goals += self.test_cllp_on_trajectory(trajectory)

        self.neptune_run['cllp_achieved_goals'].append(cllp_achieved_goals / len(self.data_to_test_cllp))

    def iterate_trainers(self,) -> Iterable[tuple[str, type[HFTrainer], Callable[..., TrainingArguments], MetricsHF]]:
        for name, component in self.components.items():
            component: InferenceComponent
            if component.is_trainable():
                training_module = component.get_component_training_module()
                logger.info(f'Component {name} is trainable, iterating over training modules.')

                if isinstance(training_module, TrainingModule):
                    yield (name, training_module.trainer_class, training_module.trainer_args,
                           training_module.metrics_for_component)
                else:
                    assert isinstance(training_module, dict)
                    logger.info(
                        f'Component {name} is nested, iterating over training modules: {training_module.keys()}.')
                    for k, v in training_module.items():
                        yield os.path.join(name, str(k)), v.trainer_class, v.trainer_args, v.metrics_for_component
            else:
                logger.info(f'Component {name} is not trainable, skipping.')

    def train_prepare_components_one_rl_iteration(self, iteration: int) -> None:
        for name, trainer, args, metrics in self.iterate_trainers():
            # Model Preparation
            logger.info(f'Training component: {name}')
            model_to_train: PreTrainedModel | None
            buffer: OfflineReplayBuffer | None
            model_to_train, buffer = self.prepare_component_for_training(name)

            if model_to_train is None or buffer is None:
                logger.warning(f'Component {name} has no model or buffer available for training.')
                continue

            logger.info(f'Replay buffer size for {name} is {len(buffer.buffer)}')
            if len(buffer.buffer) == 0:
                logger.warning('No data to train on, select better checkpoints.')
                continue

            # Prepare data
            train_dataset: GameDataset
            eval_dataset: GameDataset

            train_dataset, eval_dataset = self.prepare_data_for_training_component(buffer.buffer)

            save_dir: str = self.get_save_dir(name)
            training_args: TrainingArguments = args(output_dir=save_dir)

            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
            compute_metrics: Callable[[EvalPrediction], dict[str, float]] | None

            preprocess_logits_for_metrics, compute_metrics = metrics.get_metrics()
            if preprocess_logits_for_metrics is None or compute_metrics is None:
                preprocess_logits_for_metrics = None
                compute_metrics = None

            ready_trainer: HFTrainer = trainer(
                model=model_to_train,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_collector,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )
            ready_trainer.train()

            stats: list[dict[str, float | int]] = ready_trainer.state.log_history
            self.log_stats_to_neptune_logger(
                stats=stats,
                iteration=iteration,
                component_name=name,
                buffer_length=len(buffer.buffer),
            )

    def create_directories_for_weights(self) -> None:
        for name, _, _, _ in self.iterate_trainers():
            save_dir: str = self.get_save_dir(name)
            logger.info(f'Creating directory for weights: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

    def create_or_import_directories_for_weights(self) -> None:
        logger.info('Preparing directories for weights')
        for name, _, _, _ in self.iterate_trainers():
            save_dir: str = self.get_save_dir(name)
            logger.info(f'Creating directory for weights: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

        if self.workdir_to_import_weights_and_experiences_from is None or not os.path.exists(
                self.workdir_to_import_weights_and_experiences_from):
            logger.info('No directory to import weights and experiences from.')
            return

        experiences_fs = [
            f for f in os.listdir(self.workdir_to_import_weights_and_experiences_from) if f.startswith('experience')
        ]
        for f in experiences_fs:
            # cp
            logger.info(f'Copying from {f} from {self.weights_dump_path} to local')
            shutil.copyfile(os.path.join(self.workdir_to_import_weights_and_experiences_from, f),
                            os.path.join(self.weights_dump_path, f))

    def get_save_dir(self, name: str) -> str:
        save_dir: str = os.path.join(self.weights_dump_path, str(self.random_int), name)
        return save_dir

    @staticmethod
    def data_collector(xy: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([x for x, _ in xy]),
            'labels': torch.stack([y for _, y in xy]),
        }

    def run(self) -> None:
        logger.info('Running RL LOOP')

        self.create_or_import_directories_for_weights()
        logger.info(f'Solved boards since last training: {self.solved_boards_since_last_training} [START]')

        self.solver.construct_networks()

        iterations_with_training = 0

        if self.initialization_buffer_with_data_used_for_components_training:
            logger.info('Initializing buffers with data used for components training')
            self.initialize_buffers_with_training_data()

        if self.path_to_data_used_for_testing_cllp is not None:
            logger.info('Initializing data used for testing CLLP')
            self.prepare_data_for_testing_cllp()

        if self.initialization_buffer_with_data_used_for_components_training:
            logger.info('Initializing buffers with data used for components training')
            self.initialize_buffers_with_training_data()

        if self.path_to_data_used_for_testing_cllp is not None:
            logger.info('Initializing data used for testing CLLP')
            self.prepare_data_for_testing_cllp()

        for _iteration, batch_of_instances in zip(range(self.n_iterations), self.instance_generator.reset_dataloader()):
            # batch_of_instances tensor of shape (batch_size, *state_dim)
            if not self.train_first:
                batch_tensor = cast(torch.Tensor, batch_of_instances)
                batch_of_instances = list(batch_tensor.cpu().numpy())

                logger.info(f'Current (RL LOOP) iteration: {_iteration + 1}. Solving {len(batch_of_instances)} boards')

                solutions_and_search_infos = self.solve_one_rl_iteration(batch_of_instances, _iteration)
                solutions = [experience.solution for experience in solutions_and_search_infos]
                self.solved_boards_since_last_training += solved_count(solutions)

            if (self.solved_boards_since_last_training >= self.min_count_of_new_solved_boards_to_next_train_iteration):
                self.train_prepare_components_one_rl_iteration(_iteration)
                self.solved_boards_since_last_training -= (self.min_count_of_new_solved_boards_to_next_train_iteration)
                iterations_with_training += 1
                self.neptune_run['iterations_with_training'].append(iterations_with_training)
            else:
                logger.info(
                    f'Not enough new boards solved, skipping training. Solved boards: {self.solved_boards_since_last_training}'
                )

            if self.train_first:
                batch_tensor = cast(torch.Tensor, batch_of_instances)
                batch_of_instances = list(batch_tensor.cpu().numpy())

                logger.info(f'Current (RL LOOP) iteration: {_iteration + 1}. Solving {len(batch_of_instances)} boards')

                solutions_and_search_infos = self.solve_one_rl_iteration(batch_of_instances, _iteration)
                solutions = [experience.solution for experience in solutions_and_search_infos]
                self.solved_boards_since_last_training += solved_count(solutions)

            self.neptune_run['finished_iterations'].append(_iteration)


class DistributedSolverWorker(TrainingLoopHF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_models_ckpts_if_available(self) -> None:
        for name, component in self.components.items():
            logger.info(f'Updating component {name}')
            if isinstance(component, AdaptiveSubgoalGenerator):
                for k, v in component.subgoal_generators.items():
                    save_dir = self.get_save_dir(os.path.join(name, str(k)))
                    logger.info(f'Looking for checkpoints in {save_dir}')
                    if not os.path.exists(save_dir):
                        logger.warning(f'No checkpoints found for {name}/{k}.')
                        continue
                    fs = os.listdir(save_dir)
                    logger.info(f'Found {save_dir} with files: {fs}')
                    fs = list(filter(lambda x: x.startswith('checkpoint'), fs))
                    fs = [os.path.join(save_dir, file) for file in fs]
                    if len(fs) == 0:
                        logger.warning(f'No checkpoints found for {name}/{k}. Skipping')
                        continue
                    latest_ckpt_folder = get_latest_file(fs)
                    logger.info(f'Latest checkpoint folder for {name}/{k}: {latest_ckpt_folder}')
                    if latest_ckpt_folder is None:
                        continue
                    if isinstance(v, TransformerSubgoalGenerator):
                        v.sub_generator = cast(PreTrainedModel, v.instantiate_network(v.generator, latest_ckpt_folder))
                continue
            save_dir = self.get_save_dir(name)
            if not os.path.exists(save_dir):
                logger.warning(f'No checkpoints found for {name}. Skipping.')
                continue
            fs = os.listdir(save_dir)
            logger.info(f'Found {save_dir} with files: {fs}')
            fs = list(filter(lambda x: x.startswith('checkpoint'), fs))
            fs = [os.path.join(save_dir, file) for file in fs]
            if len(fs) == 0:
                logger.warning(f'No checkpoints found for {name}. Skipping.')
                continue
            latest_ckpt_folder = get_latest_file(fs)
            logger.info(f'Latest checkpoint folder for {name}: {latest_ckpt_folder}')
            if isinstance(component, TransformerSubgoalGenerator):
                if latest_ckpt_folder is not None:
                    component.sub_generator = cast(
                        PreTrainedModel,
                        component.instantiate_network(component.generator, latest_ckpt_folder),
                    )
            elif isinstance(component, TransformerValue):
                if latest_ckpt_folder is not None:
                    component.value_network = cast(
                        PreTrainedModel,
                        component.instantiate_network(component.value_network_class, latest_ckpt_folder),
                    )
            elif isinstance(component, TransformerConditionalLowLevelPolicy):
                if latest_ckpt_folder is not None:
                    component.cllp = cast(
                        PreTrainedModel,
                        component.instantiate_network(component.conditional_low_level_policy_class, latest_ckpt_folder),
                    )
            else:
                logger.error(f'Unknown component type: {type(component)}. Add it to _update_models_ckpts_if_available.')
                raise ValueError(f'Unknown component type: {type(component)}.')

    def run(self) -> None:
        logger.info('Running RL LOOP: Worker solving boards')
        workerid = int(os.environ.get(CARL_WORKER_LOCAL_ID, 0))
        all_workers = int(os.environ.get(CARL_ALL_NODES_COUNT, 1))

        logger.info(f'Worker id: {workerid}')
        logger.info(f'All workers: {all_workers}')

        self.solver.construct_networks()

        while True:
            logger.info(f'Worker {workerid} starting new iteration over all instances')
            data_loader = self.instance_generator.reset_dataloader()
            for _iteration, batch_of_instances in enumerate(data_loader):
                if _iteration % all_workers != workerid:
                    logger.info(f'Worker {workerid} skipping iteration {_iteration}, since its other workers job')
                    this_process_finished_iterations: int = _iteration // all_workers
                    self.neptune_run[f'solver_{workerid}/finished_iterations'].append(
                        this_process_finished_iterations)
                    continue

                if exists_resource(stop_signal):
                    logger.info(f'Worker {workerid} stopping signal found. Stopping.')
                    break

                # update models if needed
                logger.info(f'Worker {workerid} updating models')
                self._update_models_ckpts_if_available()

                # batch_of_instances tensor of shape (batch_size, *state_dim)
                batch_tensor = cast(torch.Tensor, batch_of_instances)
                batch_of_instances = list(batch_tensor.cpu().numpy())

                logger.info(f'Current (RL LOOP) iteration: {_iteration + 1}. Solving {len(batch_of_instances)} boards')

                solutions_and_search_infos = self.solve_one_rl_iteration(batch_of_instances, _iteration)

                logger.info(
                    f'Worker {workerid} dumping solutions_and_search_infos (count: {len(solutions_and_search_infos)})')

                solved_solution_and_search_infos = [
                    experience for experience in solutions_and_search_infos if experience.solution.solved
                ]

                reduced_size_solutions_and_search_infos = prune_experiences(solved_solution_and_search_infos)

                dump_resource(reduced_size_solutions_and_search_infos, 'experience')


class DistributedTrainerWorker(TrainingLoopHF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        logger.info('Running RL LOOP')

        self.create_or_import_directories_for_weights()

        self.solver.construct_networks()

        if self.initialization_buffer_with_data_used_for_components_training:
            logger.info('Initializing buffers with data used for components training')
            self.initialize_buffers_with_training_data()

        iterations_with_training = 0
        iterations_without_training = 0
        gathered_solved_boards = 0
        self.neptune_run['trainer/iterations_with_training'].append(iterations_with_training)
        all_workers = int(os.environ.get(CARL_ALL_NODES_COUNT, 1))

        for _iteration, batch_of_instances in enumerate(self.instance_generator.reset_dataloader()):
            if iterations_with_training >= self.n_iterations:
                break

            logger.info(f'Worker Trainer iteration: {iterations_with_training}')
            solutions_and_search_infos_batches = read_resource_and_delete(
                'experience',
                flatten=False,
                parallel=True,
                limit_resources_to_read=10,
            )
            solutions_and_search_infos_batches = [
                cast(list[Experience], batch)
                for batch in solutions_and_search_infos_batches
                if batch is not None
            ]

            # Experimental solving every each iteration of trainer
            if ((iterations_with_training + iterations_without_training) % all_workers == 0
                    and self.solved_boards_since_last_training
                    < self.min_count_of_new_solved_boards_to_next_train_iteration):
                logger.info(f'Worker Trainer solver iteration {_iteration} started')
                batch_tensor = cast(torch.Tensor, batch_of_instances)
                batch_of_instances = list(batch_tensor.cpu().numpy())
                solutions_and_search_infos = self.solve_one_rl_iteration(batch_of_instances, _iteration)
                solutions_and_search_infos_batches += [solutions_and_search_infos]

            count = 0

            if len(solutions_and_search_infos_batches) == 0:
                self.neptune_run['trainer/boards_receved_from_solver_nodes'].append(0)
                self.neptune_run['trainer/solved_boards_receved_from_solver_nodes'].append(0)
            else:
                for sasi_batch in solutions_and_search_infos_batches:

                    self.neptune_run['trainer/boards_receved_from_solver_nodes'].append(len(sasi_batch))

                    for solution_and_search_info in sasi_batch:
                        self.replay_buffer.add(solution_and_search_info)
                        count += 1
                    solutions = [experience.solution for experience in sasi_batch]

                    solved_boards_in_batch = solved_count(solutions)

                    self.neptune_run['trainer/solved_boards_receved_from_solver_nodes'].append(
                        solved_boards_in_batch)

                    self.solved_boards_since_last_training += solved_boards_in_batch
                    gathered_solved_boards += solved_boards_in_batch

            logger.info(f'Worker Trainer read {count} solutions')
            logger.info(f'Worker Trainer solved boards since last training: {self.solved_boards_since_last_training}')
            self.neptune_run['trainer/gathered_solved_boards'].append(gathered_solved_boards)
            self.neptune_run['trainer/solved_boards_since_last_training'].append(
                self.solved_boards_since_last_training)

            if (self.solved_boards_since_last_training >= self.min_count_of_new_solved_boards_to_next_train_iteration):
                iterations_with_training += 1
                self.neptune_run['trainer/iterations_with_training'].append(iterations_with_training)
                self.train_prepare_components_one_rl_iteration(iterations_with_training)
                self.solved_boards_since_last_training -= (self.min_count_of_new_solved_boards_to_next_train_iteration)
            else:
                logger.info(
                    f'Not enough new boards solved, skipping training. Solved: {self.solved_boards_since_last_training}'
                )
                logger.info('Sleeping for 5 seconds')
                iterations_without_training += 1
                self.neptune_run['trainer/iterations_without_training'].append(iterations_without_training)
                time.sleep(5)

        dump_resource(stop_signal, stop_signal)

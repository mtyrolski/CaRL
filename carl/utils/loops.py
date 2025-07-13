import os
from dataclasses import dataclass
from dataclasses import field
from os.path import isdir
from os.path import join
from typing import Generator, Iterator

import joblib
from loguru import logger
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from carl.algorithms.training_loop.flow_control import LoopControl
from carl.environment.instance_generator import InstanceGenerator
from carl.inference_components.component import InferenceComponent
from carl.inference_components.component import TrainingModule
from carl.memory.replay_buffer import OfflineReplayBuffer
from carl.memory.replay_buffer import SimpleUniversalReplayBuffer
from carl.planners.base import Experience
from carl.solver.subgoal_search import Solver
from carl.utils.training import iterate_networks_for_training
from carl.utils.training_metrics import extract_metrics_from_buffer_logs
from carl.utils.training_metrics import extract_metrics_from_experiences


class ComponentCollection:
    def __init__(self, **components_kwargs: InferenceComponent) -> None:
        self.components: dict[str, InferenceComponent] = {
            name: component for name, component in components_kwargs.items()
        }

    @classmethod
    def from_solver(cls, solver: Solver) -> 'ComponentCollection':
        return cls(subgoal_generator=solver.subgoal_generator,
                   validator=solver.validator,
                   value_function=solver.value_function)

    def iterate_training_modules(self) -> Iterator[tuple[str, TrainingModule]]:
        for name, component in self.components.items():
            for training_module in component.get_component_training_module():
                yield name, training_module


@dataclass
class Logs:
    prefix: str = ""
    content: dict[str, str | float | int] = field(default_factory=dict)

    def update(self, metrics: dict[str, str | float | int]) -> 'Logs':
        self.content.update({join(self.prefix, key): value for key, value in metrics.items()})

        return self


ProblemInstance = ndarray


def assert_list_of_type(lst: list, type_: type):
    assert all(isinstance(x, type_) for x in lst)


def _load_offline_data_file(path_to_offline_trajectories):
    content = joblib.load(path_to_offline_trajectories)

    if isinstance(content, dict):
        trajectories: list[ndarray] = list(content.values())
        assert_list_of_type(trajectories, ndarray)
        return trajectories
    elif isinstance(content, list):
        assert_list_of_type(content, ndarray)
        trajectories: list[ndarray] = content
        return trajectories

    raise ValueError(f"Unsupported type of offline data file: {type(content)}")


def iterate_offline_data(path_to_offline_trajectories: str, limit: int) -> Generator[list[ndarray], None, None]:
    already_loaded_trajectories = 0
    if isdir(path_to_offline_trajectories):
        chunks_of_data = [f for f in os.listdir(path_to_offline_trajectories) if f.endswith(('.pkl', '.joblib'))]

        for f in chunks_of_data:
            partial_data = _load_offline_data_file(join(path_to_offline_trajectories, f))

            if len(partial_data) == 0:
                continue

            num_to_load = min(limit - already_loaded_trajectories, len(partial_data))
            if num_to_load == 0:
                break

            already_loaded_trajectories += num_to_load
            yield partial_data[:num_to_load]


def get_next_batch(online_loader_generator: DataLoader, loop_control: LoopControl) -> list[ProblemInstance]:
    raw_batch: ndarray = next(online_loader_generator)
    batch_of_instances = list(raw_batch.cpu().numpy())

    if len(batch_of_instances) + loop_control.already_attempted_problems > loop_control.num_online_trajectories:
        batch_of_instances = batch_of_instances[:loop_control.num_online_trajectories -
                                                loop_control.already_attempted_problems]

    loop_control.already_attempted_problems += len(batch_of_instances)
    logger.info(
        f"Loaded {len(batch_of_instances)} instances for solving, {loop_control.already_attempted_problems} in total")
    return batch_of_instances


def log_step_info(loop_control: LoopControl, *logs: Logs) -> None:
    for log in logs:
        for key, value in log.content.items():
            logger.info(f"[Step {loop_control.current_iteration}] {key}: {value}")


def solve_trajectories(solver: Solver, online_boards: list[ndarray], n_jobs: int) -> tuple[list[Experience], Logs]:
    outputs_all_at_once: list[Experience] = joblib.Parallel(n_jobs=n_jobs, verbose=100)(
        joblib.delayed(solver.solve)(initial_state) for initial_state in online_boards)

    logs = Logs('solve').update(extract_metrics_from_experiences(outputs_all_at_once))
    return outputs_all_at_once, logs


def update_buffer_with_experiences(replay_buffer: SimpleUniversalReplayBuffer, experiences: list[Experience]) -> Logs:
    buffer_logs: list[dict[str, str | float | int]] = []
    for experience in tqdm(experiences,
                           desc="Adding experiences to replay buffer",
                           unit="experience",
                           total=len(experiences)):
        update_log = replay_buffer.add(experience)
        buffer_logs.append(update_log)

    logs = Logs('replay_buffer').update(extract_metrics_from_buffer_logs(buffer_logs))
    return logs


def train_components(components: ComponentCollection, replay_buffer: SimpleUniversalReplayBuffer,
                     loop_control: LoopControl) -> Logs:

    for component_name, component in components.components.items():
        logger.info(f"Training component {component_name}")
        training_module = component.get_component_training_module()
        network_container, buffer_container = iterate_networks_for_training(component, replay_buffer)
        if isinstance(network_container, dict):
            assert isinstance(buffer_container, dict) and isinstance(training_module, dict)
            for k, v in network_container.items():
                logger.info(f"Training component {component_name}/{k}")
        else:
            assert isinstance(network_container, InferenceComponent) and isinstance(buffer_container,
                                                                                    OfflineReplayBuffer)
            logger.info(f"Training component {component_name}")

    logs = Logs()
    return logs


def evaluate_components(components: ComponentCollection, eval_instance_generator: InstanceGenerator | None) -> Logs:
    if eval_instance_generator is None:
        return Logs()

    all_outputs: list[Experience] = []
    for batch_of_instances in eval_instance_generator.reset_dataloader():
        outputs = joblib.Parallel(n_jobs=1, verbose=100)(
            joblib.delayed(components.component.solve)(initial_state) for initial_state in batch_of_instances)
        all_outputs.extend(outputs)

    return Logs('eval').update(extract_metrics_from_experiences(all_outputs))

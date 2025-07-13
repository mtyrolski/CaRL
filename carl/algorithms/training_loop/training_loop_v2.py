from itertools import cycle

from loguru import logger

import carl.utils.loops as loop_utils
from carl.algorithms.algorithm import Algorithm
from carl.algorithms.training_loop.flow_control import LoopControl
from carl.environment.instance_generator import InstanceGenerator
from carl.memory.replay_buffer import SimpleUniversalReplayBuffer
from carl.planners.base import Experience
from carl.solver.subgoal_search import Solver
from carl.utils.loops import ComponentCollection
from carl.utils.loops import Logs
from carl.utils.loops import ProblemInstance


class TrainingLoop(Algorithm):
    def __init__(
        self,
        replay_buffer: SimpleUniversalReplayBuffer,
        loop_control: LoopControl,
        solver: Solver,
        instance_generator: InstanceGenerator,
        eval_instance_generator: InstanceGenerator | None = None,
        path_to_offline_trajectories: str = None,
        n_solving_jobs: int = 1,
    ):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.solver = solver
        self.components: ComponentCollection = ComponentCollection.from_solver(solver)
        self.loop_control = loop_control
        self.instance_generator = instance_generator
        self.eval_instance_generator = eval_instance_generator
        self.path_to_offline_trajectories = path_to_offline_trajectories
        self.n_solving_jobs = n_solving_jobs

    def initialize_buffers_from_offline_data(self) -> None:
        for experiences in loop_utils.iterate_offline_data(self.path_to_offline_trajectories,
                                                           self.loop_control.num_offline_trajectories):
            experiences: list[Experience]
            logger.debug(f"Adding offline experience to replay buffer: {experiences}")
            self.replay_buffer.add_from_trajectories(experiences)

    def init_components_weights(self) -> None:
        for component in self.components.components.values():
            component.construct_network()

    def serialize_full_loop_state(self) -> None:
        pass

    def run(self) -> None:
        self.init_components_weights()
        self.initialize_buffers_from_offline_data()
        logger.info("Starting training loop.")

        online_loader_generator = cycle(self.instance_generator.reset_dataloader())

        while self.loop_control.is_active():
            loop_step: int = self.loop_control.current_iteration
            logger.info(f"Starting iteration {loop_step}")

            online_boards: list[ProblemInstance] = loop_utils.get_next_batch(online_loader_generator, self.loop_control)

            # Solution and enhancing replay buffer with new experiences
            experiences, solve_logs = loop_utils.solve_trajectories(self.solver, online_boards, self.n_solving_jobs)
            buffer_update_logs: Logs = loop_utils.update_buffer_with_experiences(self.replay_buffer, experiences)
            loop_utils.log_step_info(self.loop_control, solve_logs, buffer_update_logs)

            # Updating components
            train_logs: Logs = loop_utils.train_components(self.components, self.replay_buffer, self.loop_control)
            loop_utils.log_step_info(self.loop_control, train_logs)

            # Evaluating components
            eval_logs: Logs = loop_utils.evaluate_components(self.components, self.eval_instance_generator)
            loop_utils.log_step_info(self.loop_control, eval_logs)

            self.loop_control.complete_iteration()

        self.serialize_full_loop_state()

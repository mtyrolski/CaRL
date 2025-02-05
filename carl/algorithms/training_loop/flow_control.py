from loguru import logger


class LoopControl:
    def __init__(self, n_iterations_limit: int, num_online_trajectories: int, num_offline_trajectories: int) -> None:
        self._n_iterations_limit = n_iterations_limit
        self._total_completed_iterations: int = 0
        self._active = self._total_completed_iterations <= n_iterations_limit    # is iteration active
        self.already_attempted_problems: int = 0    # Number of boards released by the instance generator (for solving)
        self.num_online_trajectories: int = num_online_trajectories    # Number of online trajectories to load in lifespan of the loop
        self.num_offline_trajectories: int = num_offline_trajectories    # Number of offline trajectories to load in lifespan of the loop

    def is_active(self) -> bool:
        return self._active

    def complete_iteration(self) -> None:

        if self._active is False:
            raise ValueError("Cannot complete an iteration that has not been started.")
        logger.info(f"Iteration {self.current_iteration} completed")
        self._total_completed_iterations += 1
        self._active = (self._total_completed_iterations
                        < self._n_iterations_limit) and (self.already_attempted_problems < self.num_online_trajectories)

    @property
    def current_iteration(self) -> int:
        """
        Iteration steps (indexed from 0)
        """
        return self._total_completed_iterations

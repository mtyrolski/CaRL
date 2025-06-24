import os

from loguru import logger

from carl.environment.env import GameEnv


class CreateDataset:
    def __init__(
        self,
        env: GameEnv,
        number_of_trajectories: int,
        max_moves: int,
        path_to_save: str | None = None,
        save_after: int = 1,
        noisy_reverse_prob: float = 0.0,
    ):
        self.env = env
        self.number_of_trajectories = number_of_trajectories
        self.max_moves = max_moves
        self.path_to_save = path_to_save
        self.save_after = save_after
        self.noisy_reverse_prob = noisy_reverse_prob

    def run(self) -> None:
        logger.warning('Building dataset')

        os.system(f'mkdir -p {self.path_to_save}')

        # For rubik only (from already implemented code)
        self.env.generate_training_data(
            number_of_trajectories=self.number_of_trajectories,
            max_moves=self.max_moves,
            path_to_save=self.path_to_save,
            save_after=self.save_after,
            noisy_reverse_prob=self.noisy_reverse_prob,
        )

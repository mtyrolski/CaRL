# Generation of data for the n-puzzle problem; size of the board is 5x5.
from carl.environment.n_puzzle.env import NPuzzleCore


class GenerateDataNPuzzle:
    def __init__(
        self,
        env: NPuzzleCore,
        n_training_samples: int,
        n_evaluation_samples: int,
        n_training_steps: int,
        n_eval_steps: int,
        path_to_save_offline_data: str,
        path_to_save_online_data: str,
        save_after_each: int,
    ):
        self.env = env
        self.n_training_samples = n_training_samples
        self.n_evaluation_samples = n_evaluation_samples
        self.n_training_steps = n_training_steps
        self.n_eval_steps = n_eval_steps
        self.path_to_save_offline_data = path_to_save_offline_data
        self.path_to_save_online_data = path_to_save_online_data
        self.save_after_each = save_after_each

    def run(self):
        self.env.generate_random_unique_dataset_with_solution(
            n_training_samples=self.n_training_samples,
            n_evaluation_samples=self.n_evaluation_samples,
            n_training_steps=self.n_training_steps,
            n_eval_steps=self.n_eval_steps,
            path_to_save_offline_data=self.path_to_save_offline_data,
            path_to_save_online_data=self.path_to_save_online_data,
            save_after_each=self.save_after_each,
        )


generation = GenerateDataNPuzzle(
    env=NPuzzleCore(size_of_board=(5, 5)),
    n_training_samples=1,
    n_evaluation_samples=1001,
    n_training_steps=1,
    n_eval_steps=1000,
    path_to_save_offline_data=None,
    path_to_save_online_data='/n_puzzle/progress/5x5/shuffle_1000',
    save_after_each=1000,
)

generation.run()

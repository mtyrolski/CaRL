from dataclasses import dataclass
from os import listdir
from os.path import exists
from os.path import join
from pickle import HIGHEST_PROTOCOL
from random import choices
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from joblib import dump
from joblib import load
from loguru import logger
from numpy import ndarray
from tqdm import tqdm
from transformers import BertForSequenceClassification

from carl.algorithms.algorithm import Algorithm
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.inference_components.value import TransformerValue
from carl.inference_components.value import Value

HIGHEST_K: int = 10


@dataclass
class OptimalSolution:
    problem_instance: str
    solution: list[int]


@dataclass
class StateOnSolutionPath:
    state: ndarray
    value: float


def rollout_trajectory(solution: OptimalSolution, env: SokobanEnv) -> list[ndarray]:
    """Converts single state (M, M, N_boxes) into list of states."""
    env.restore_full_state_from_np_array_version(solution.problem_instance)
    all_states: list[ndarray] = [solution.problem_instance]
    for action in solution.solution:
        state, _, _, _ = env.step(action)
        all_states.append(state)
    return all_states


def split_to_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def load_optimal_solutions(fs: list[str]) -> list[OptimalSolution]:
    optimal_data = []
    for f in fs:
        potential_path: str = join('optimal', f, 'xy_solutions.joblib')
        if exists(potential_path):
            board_solution_index_tuples = load(potential_path)

            for (board, solution, _) in board_solution_index_tuples:
                optimal_data.append(OptimalSolution(problem_instance=board, solution=solution))
    return optimal_data


def instantiate_value(path_to_value_function_weights, env: SokobanEnv):
    value_function: Value = TransformerValue(
        value_network_class=BertForSequenceClassification.from_pretrained,
        path_to_value_network_weights=path_to_value_function_weights,
        env=env,
        type_of_evaluation='regression',
    )

    value_function.construct_network()

    return value_function


def iterate_batches(full_trajectory: list[ndarray], k: int):
    for i in range(0, len(full_trajectory) - k):
        yield full_trajectory[i:i + k], full_trajectory[i + k]


def generate_state_subgoal_pairs(rollouted_trajectories: list[list[StateOnSolutionPath]], k: int, sample_limit=100_000):
    state_subgoal_pairs = []
    for trajectory in rollouted_trajectories:
        for i in range(0, len(trajectory) - k):
            state_subgoal_pairs.append((trajectory[i], trajectory[i + k]))
    if len(state_subgoal_pairs) > sample_limit:
        logger.warning(f'Sampling {sample_limit} state subgoal pairs from {len(state_subgoal_pairs)}')
        state_subgoal_pairs = choices(state_subgoal_pairs, k=sample_limit)

    return state_subgoal_pairs


class ValidSolution(Algorithm):
    def __init__(self,):
        super().__init__()

    def run(self) -> None:
        env = SokobanEnv(SokobanTokenizer(None, size_of_board=(12, 12)), num_boxes=4)
        optimal_trajectories: list[OptimalSolution] = load_optimal_solutions(listdir('optimal'))
        logger.info(f'Optimal trajectories {len(optimal_trajectories)} loaded')
        value_function: TransformerValue = instantiate_value(
            path_to_value_function_weights='validation/sokoban/components/full_data/value/checkpoint-1343100', env=env)
        figs = []

        cache_file_name: str = 'rollouted_trajectories.joblib'
        # logger.warning('Limiting to 50 optimal trajectories')
        # optimal_trajectories = optimal_trajectories[]
        rollouted_trajectories: list[list[StateOnSolutionPath]] = []

        # Rolling out trajectories
        if exists(cache_file_name):
            logger.info('Rollouted trajectories already exist, loading them')
            rollouted_trajectories = load(cache_file_name)
        else:
            rollouted_trajectories: list[list[StateOnSolutionPath]] = []

            for trajectory in tqdm(optimal_trajectories,
                                   desc='Rolling out trajectories',
                                   total=len(optimal_trajectories)):
                full_trajectory: list[ndarray] = rollout_trajectory(trajectory, env)
                states_on_path: list[StateOnSolutionPath] = []
                for state in full_trajectory:
                    value = value_function.get_value(state)
                    states_on_path.append(StateOnSolutionPath(state=state, value=value))
                rollouted_trajectories.append(states_on_path)

            # Caching states with assigned value
            dump(rollouted_trajectories, cache_file_name, protocol=HIGHEST_PROTOCOL)

        # Distribution of optimal solutions lengths in sokoban 12x12x4
        lengths = [len(t.solution) for t in optimal_trajectories]
        figs.append((px.histogram(x=lengths, title='Distribution of optimal solutions lengths',
                                  labels={'x': 'length'}), 'optimal_solutions_lengths'))

        # Probability of good ordering of value on solving trajectories for k=1,2,3,4,5,...
        # What stats do we want to gather?
        # - How many times the value of the next subgoal is higher than the current state?
        # - What is real value difference value(next_subgoal) - value(current_state)?

        # Bucketing the value differences of sampled pairs of states (and theirs subgoals)
        value_differences = {k: [] for k in range(1, HIGHEST_K + 1)}

        for k in range(1, HIGHEST_K + 1):
            state_subgoal_pairs = generate_state_subgoal_pairs(rollouted_trajectories, k)
            for state_subgoal_pair in state_subgoal_pairs:
                current_state, next_subgoal = state_subgoal_pair
                value_differences[k].append(next_subgoal.value - current_state.value)

        # Plotting the distribution of value differences
        MIN_ARG = -0.1
        MAX_ARG = 0.1

        common_histogram_with_overlay = go.Figure()

        # Add a non-linear trendline to the histogram
        colors_palette = px.colors.qualitative.Dark24
        BINS = 50
        for k in range(1, HIGHEST_K + 1):
            items = value_differences[k]
            ys_values = [0 for _ in range(0, BINS)]
            for item in items:
                index = int((item - MIN_ARG) / (MAX_ARG - MIN_ARG) * BINS)
                index = min(max(index, 0), BINS - 1)
                ys_values[index] += 1
            ys_values = [y / len(items) for y in ys_values]

            common_histogram_with_overlay.add_trace(
                go.Scatter(x=[MIN_ARG + ((MAX_ARG - MIN_ARG) / BINS) * i for i in range(0, BINS)],
                           y=ys_values,
                           mode='lines',
                           name=f'k={k} trendline',
                           line=dict(color=colors_palette[k], width=2)))

        # add x=0 line
        common_histogram_with_overlay.add_trace(
            go.Scatter(x=[0, 0], y=[0, 0.25], mode='lines', name='x=0', line=dict(color='black', width=2)))

        common_histogram_with_overlay.update_layout(
            title='Distribution of value differences for k=1,2,3,4,5,...',
            xaxis_title='value difference',
            yaxis_title='count',
            barmode='overlay',
        )
        figs.append((common_histogram_with_overlay, 'value_differences'))

        # How many times the value of the next subgoal is higher than the current state?
        accuracy_per_k = {k: 0 for k in range(1, HIGHEST_K + 1)}
        for k in range(1, HIGHEST_K + 1):
            state_subgoal_pairs = generate_state_subgoal_pairs(rollouted_trajectories, k)
            for state_subgoal_pair in state_subgoal_pairs:
                current_state, next_subgoal = state_subgoal_pair
                if next_subgoal.value > current_state.value:
                    accuracy_per_k[k] += 1
            accuracy_per_k[k] /= len(state_subgoal_pairs)

        # Plotting the accuracy per k
        figs.append((
            px.bar(
                x=list(accuracy_per_k.keys()),
                y=list(accuracy_per_k.values()),
                title='Accuracy per k',
                labels={
                    'x': 'k',
                    'y': 'accuracy'
                },
        # value over bars:
                text=list(accuracy_per_k.values()),
            ),
            'accuracy_per_k'))

        # dump to html
        for fig, name in figs:
            fig.write_html(f'{name}.html')
            logger.success(f'{name} saved to {name}.html')


if __name__ == '__main__':
    ValidSolution().run()

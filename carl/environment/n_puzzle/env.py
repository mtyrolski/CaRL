import copy
import os

import numpy as np
from joblib import dump
from loguru import logger
from sympy import plot
from torch import Tensor

from carl.environment.env import GameEnv
from carl.environment.env import ReadableReprT
from carl.environment.env import RepresentationType
from carl.environment.n_puzzle.tokenizer import NPuzzleTokenizer
from carl.environment.tokenizer import GameTokenizer
from carl.environment.utilis import HashableState

import numpy as np
import plotly.graph_objects as go

from carl.utils.aliases import State

def plot_n_puzzle(state: np.ndarray) -> go.Figure:
    """
    Plots the N-Puzzle board using Plotly given the state as a 1D ndarray of shape (N*N,).
    Only one number per tile, using only annotations. The empty cell (value 0) is left blank.
    """
    size = int(np.sqrt(state.size))
    board = state.reshape((size, size))

    # Dummy heatmap for grid background only
    fig = go.Figure(
        data=go.Heatmap(
            z=np.ones_like(board),
            colorscale=[[0, "white"], [1, "white"]],
            showscale=False,
            hoverinfo="skip",
            text=None,
            texttemplate=None
        )
    )
    fig.update_layout(
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, scaleanchor="y", constrain="domain"
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, autorange='reversed', scaleanchor="x"
        ),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        width=80*size,
        height=80*size,
    )

    for i in range(size):
        for j in range(size):
            value = board[i, j]
            if value != 0:
                fig.add_annotation(
                    x=j, y=i,
                    text=f"<b>{value}</b>",
                    showarrow=False,
                    font=dict(size=34, color="black", family="Arial Black"),
                    xanchor="center", yanchor="middle"
                )

    # Draw grid lines
    for i in range(size+1):
        # Horizontal lines
        fig.add_shape(type="line",
                      x0=-0.5, x1=size-0.5, y0=i-0.5, y1=i-0.5,
                      line=dict(color="black", width=3))
        # Vertical lines
        fig.add_shape(type="line",
                      x0=i-0.5, x1=i-0.5, y0=-0.5, y1=size-0.5,
                      line=dict(color="black", width=3))

    fig.update_xaxes(range=[-0.5, size-0.5])
    fig.update_yaxes(range=[-0.5, size-0.5])

    return fig


class NPuzzleCore:
    """
    This class is used to implement the core logic of the n-puzzle game.

    Moves are represented as integers:
    0: Left
    1: Right
    2: Up
    3: Down
    """
    def __init__(self, size_of_board: tuple[int, int] = (5, 5)) -> None:
        self._size_of_board = size_of_board
        assert self._size_of_board[0] == self._size_of_board[1], 'Board must be square'
        self._goal_state: np.ndarray = np.array(list(range(1, self._size_of_board[0]**2)) + [0])

    @property
    def size_of_board(self) -> tuple[int, int]:
        return self._size_of_board

    def available_actions(self, state: np.ndarray) -> list[int]:
        """Returns a list of available actions for a given state."""

        empty_space: int = int(np.where(state == 0)[0][0])
        n: int = self._size_of_board[0]
        moves: list[int] = [0, 1, 2, 3]

        if empty_space % n == 0:
            moves.remove(0)
        if empty_space % n == n - 1:
            moves.remove(1)
        if empty_space - n < 0:
            moves.remove(2)
        if empty_space + n > n * n - 1:
            moves.remove(3)

        return moves

    def next_step(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Returns the next state given a state and an action.

        If the action is not valid, the state is returned unchanged.
        """

        assert action in [0, 1, 2, 3], f'Invalid action: {action}, must be in [0, 1, 2, 3]'

        next_state: np.ndarray = copy.deepcopy(state)
        empty_space: int = int(np.where(state == 0)[0][0])
        n: int = self._size_of_board[0]

        valid_actions: list[int] = self.available_actions(state)

        if action in valid_actions:

            if action == 0:
                next_state[empty_space], next_state[empty_space - 1] = (
                    next_state[empty_space - 1],
                    next_state[empty_space],
                )
            elif action == 1:
                next_state[empty_space], next_state[empty_space + 1] = (
                    next_state[empty_space + 1],
                    next_state[empty_space],
                )
            elif action == 2:
                next_state[empty_space], next_state[empty_space - n] = (
                    next_state[empty_space - n],
                    next_state[empty_space],
                )
            else:
                next_state[empty_space], next_state[empty_space + n] = (
                    next_state[empty_space + n],
                    next_state[empty_space],
                )

        return next_state

    def next_unique_step(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Returns the next state given a state and an action.

        This method not allow for duplicate states.
        """

        assert action in self.available_actions(state), 'Invalid action'

        next_state: np.ndarray = copy.deepcopy(state)
        empty_space: int = int(np.where(state == 0)[0][0])
        n: int = self._size_of_board[0]

        if action == 0:
            next_state[empty_space], next_state[empty_space - 1] = (
                next_state[empty_space - 1],
                next_state[empty_space],
            )
        elif action == 1:
            next_state[empty_space], next_state[empty_space + 1] = (
                next_state[empty_space + 1],
                next_state[empty_space],
            )
        elif action == 2:
            next_state[empty_space], next_state[empty_space - n] = (
                next_state[empty_space - n],
                next_state[empty_space],
            )
        else:
            next_state[empty_space], next_state[empty_space + n] = (
                next_state[empty_space + n],
                next_state[empty_space],
            )

        return next_state

    def is_solved(self, state: np.ndarray) -> bool:
        """Returns True if the given state is solved, False otherwise."""

        return np.array_equal(state, self._goal_state)

    def generate_random_state_with_solution(self,
                                            number_of_moves: int) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
        """
        Generates a random state and its solution.

        args: number_of_moves: number of moves to generate the state.
        returns: state: the generated state. solution: the solution to the generated state, i.e. the sequence of
        moves to reach the goal state. trajectory: the trajectory of the generated state, i.e. the sequence of states
        generated by the sequence of moves.
        """

        state: np.ndarray = self._goal_state
        solution: list[int] = []
        trajectory: list[np.ndarray] = [state]

        for _ in range(number_of_moves):
            action: int = np.random.choice(self.available_actions(state))
            state = self.next_unique_step(state, action)
            solution.append(action)
            trajectory.append(state)

        trajectory.reverse()

        return state, solution, trajectory

    def generate_random_unique_dataset_with_solution(
        self,
        n_training_samples: int,
        n_evaluation_samples: int,
        n_training_steps: int,
        n_eval_steps: int,
        path_to_save_offline_data: str | None = None,
        path_to_save_online_data: str | None = None,
        save_after_each: int | None = None,
    ) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
        """
        Generates a dataset of random states with their solutions and data for online training.

        args: n_training_samples: number of training samples to generate.
        n_evaluation_samples: number of evaluation samples to generate.
        n_training_steps: number of steps to generate the training samples.
        n_eval_steps: number of steps to generate the evaluation samples.
        path_to_save_offline_data: path to save the offline data.
        path_to_save_online_data: path to save the online data.
        save_after_each: save the data after each sample set.

        returns: training_data: the training data and data for online training.
        """

        states: set[HashableState] = set()
        trajectories: dict[int, list[np.ndarray]] = {}
        staring_states: list[np.ndarray] = []
        offline_part: int = 0
        online_part: int = 0

        if save_after_each is None and (path_to_save_offline_data or path_to_save_online_data is not None):
            logger.info('Saving after each sample set to 1.')
            save_after_each = 1

        while len(states) < n_training_samples:
            state, _, trajectory = self.generate_random_state_with_solution(n_training_steps)
            state_hash: HashableState = HashableState(state, None, None)

            if state_hash not in states:
                states.add(state_hash)
                trajectories[len(states) - 1] = trajectory

            if path_to_save_offline_data is not None and len(trajectories) == save_after_each:
                logger.info(f'Saving offline dataset to {path_to_save_offline_data}.')
                logger.info(f'Generated {len(trajectories)} trajectories.')
                dump(
                    trajectories,
                    os.path.join(
                        path_to_save_offline_data,
                        f'offline_n_puzzle_trajectories_{offline_part}.pkl',
                    ),
                )
                trajectories.clear()
                offline_part += 1

        training_instances_size: int = len(states)

        while len(states) < n_evaluation_samples + training_instances_size:
            state, _, _ = self.generate_random_state_with_solution(n_eval_steps)
            state_hash: HashableState = HashableState(state, None, None)

            if state_hash not in states:
                states.add(state_hash)
                staring_states.append(state)

            if path_to_save_online_data is not None and len(staring_states) == save_after_each:
                logger.info(f'Saving online dataset to {path_to_save_online_data}.')
                logger.info(f'Generated {len(staring_states)} unique states for online training.')
                dump(
                    staring_states,
                    os.path.join(path_to_save_online_data, f'online_n_puzzle_trajectories_{online_part}.pkl'),
                )
                staring_states.clear()
                online_part += 1

        return trajectories, staring_states

def is_ndarray_state(state: State) -> bool:
    """
    Checks if the given state is a numpy ndarray.
    
    Args:
        state (State): The state to check.
        
    Returns:
        bool: True if the state is a numpy ndarray, False otherwise.
    """
    return isinstance(state, np.ndarray)

class NPuzzleEnv(GameEnv):
    @property
    def name(self) -> str:
        return 'n_puzzle'

    def __init__(self, tokenizer: NPuzzleTokenizer) -> None:
        self._tokenizer = tokenizer
        self.core = NPuzzleCore(size_of_board=tokenizer.size_of_board)
        self.internal_state: np.ndarray | None = None

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    def detect_action(self, board_before: State, board_after: State) -> int:
        """Detects the action that was taken to go from board_before to board_after."""

        empty_before: int = int(np.where(board_before == 0)[0][0])
        empty_after: int = int(np.where(board_after == 0)[0][0])
        n: int = self.core.size_of_board[0]

        if empty_before == empty_after + 1:
            return 0
        elif empty_before == empty_after - 1:
            return 1
        elif empty_before == empty_after + n:
            return 2
        else:
            return 3

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        """Converts a distribution to an action."""

        return int(distribution.argmax().item())

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        """
        Performs an action and returns the next state, the reward, whether the game is over and some info.
        """

        self.internal_state = self.next_state(self.internal_state, action)
        reward: float = 0.0
        done: bool = self.is_solved(self.internal_state)
        info: dict = {}

        return self.internal_state, reward, done, info

    def restore_full_state_from_np_array_version(self, state: State) -> None:
        assert is_ndarray_state(state), 'State must be a numpy array'
        self.internal_state = state # type: ignore

    def next_state(self, state: State, action: int) -> State:
        assert is_ndarray_state(state), 'State must be a numpy array'
        return self.core.next_step(state, action) # type: ignore

    def get_state(self) -> State:
        assert self.internal_state is not None, 'Internal state is not set'
        return self.internal_state # type: ignore

    def is_solved(self, board: State) -> bool:
        assert is_ndarray_state(board), 'Board must be a numpy array'
        return self.core.is_solved(board) # type: ignore

    def state_to_repr(
        self,
        state: State,
        title: str | None = None,
        repr_type: RepresentationType = RepresentationType.STR,
    ) -> ReadableReprT:
        _supported_repr_types: list[RepresentationType] = [RepresentationType.STR, RepresentationType.GO_FIGURE]
        logger.warning(f'Representation type: {repr_type} with value {repr_type.value}')
        logger.warning(f'Supported representation types: {_supported_repr_types} with values {[x.value for x in _supported_repr_types]}')
        
        if repr_type.value not in map(lambda x: x.value, _supported_repr_types):
            logger.warning(f'Only {_supported_repr_types} are supported, not {repr_type}')
            repr_type = RepresentationType.STR

        if repr_type.value == RepresentationType.GO_FIGURE.value:
            return plot_n_puzzle(state) # type: ignore
        else:
            assert repr_type.value == RepresentationType.STR.value
            return f'{title}: {state}' if title else str(state) # type: ignore
        

    def many_states_to_repr(
        self,
        states: list[State],
        titles: list[str],
        repr_type: RepresentationType = RepresentationType.STR,
    ) -> ReadableReprT:
        _supported_repr_types: list[RepresentationType] = [RepresentationType.STR, RepresentationType.GO_FIGURE]
        
        if repr_type.value not in map(lambda x: x.value, _supported_repr_types):
            logger.warning(f'Only {_supported_repr_types} are supported, not {repr_type}')
            repr_type = RepresentationType.STR


        if repr_type.value == RepresentationType.GO_FIGURE.value:
            go_figures: list[go.Figure] = [plot_n_puzzle(state) for state in states] # type: ignore
            return go.Figure(data=go_figures) # type: ignore
        else:
            assert repr_type.value == RepresentationType.STR.value
            return '\n'.join(f'{title}: {state}' for state, title in zip(states, titles)) # type: ignore

    def set_state(self, state: State) -> None:
        assert isinstance(state, np.ndarray), 'State must be a numpy array'
        assert state.shape == self.core.size_of_board, f'State must be of shape {self.core.size_of_board}, not {state.shape}'
        self.internal_state = state

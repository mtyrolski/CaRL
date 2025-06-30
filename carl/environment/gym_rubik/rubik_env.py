import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from loguru import logger
from torch import Tensor

from carl.environment.env import GameEnv
from carl.environment.env import RepresentationType, ReadableReprT
from carl.environment.gym_rubik.converter import CubeConverter
from carl.environment.gym_rubik.cube import Actions
from carl.environment.gym_rubik.cube import Cube
from carl.environment.gym_rubik.rubik_solver import Move as rubik_solver_moves
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.environment.tokenizer import GameTokenizer


class DebugLevel(Enum):
    WARNING = (0,)
    INFO = (1,)
    VERBOSE = 2


ACTION_LOOKUP = {
    0: Actions.U,
    1: Actions.U_1,
    2: Actions.D,
    3: Actions.D_1,
    4: Actions.F,
    5: Actions.F_1,
    6: Actions.B,
    7: Actions.B_1,
    8: Actions.R,
    9: Actions.R_1,
    10: Actions.L,
    11: Actions.L_1,
}


# Refactor RubikEnv to inherit from GameEnv and implement required interface
class RubikEnv(GameEnv):

    def __init__(self, tokenizer: RubikCubeTokenizer = None, step_limit=100000000000000000000000, shuffles=50, obs_type='basic'):
        super().__init__()
        self._tokenizer = tokenizer
        self.cube = Cube(3, whiteplastic=False)
        self.solved_state = self.cube.get_state()
        self.obs_type = obs_type
        self.converter = None
        self.scramble = []
        self.debugLevel = DebugLevel.WARNING
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = False
        self.scrambleSize = shuffles
        self.num_steps = 0
        self.step_limit = step_limit
        self.config()
        self.internal_state = self._get_state()

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self._take_action(action)
        reward = -1
        self.num_steps += 1
        self.internal_state = self._get_state()
        solved = np.array_equal(self.cube.get_state(), self.solved_state)
        if solved:
            reward = 0
        episode_over = solved
        return self.internal_state, reward, episode_over, {}

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        self.load_state(state)
        self._take_action(action)
        return self._get_state()

    def restore_full_state_from_np_array_version(self, state: np.ndarray) -> None:
        self.load_state(state)

    def set_state(self, state: np.ndarray) -> None:
        self.load_state(state)

    def get_state(self) -> np.ndarray:
        return self._get_state()

    def is_solved(self, board: np.ndarray) -> bool:
        return np.array_equal(board, self.solved_state)

    def detect_action(self, board_before: np.ndarray, board_after: np.ndarray) -> int | None:
        # Try all actions and see which one leads from board_before to board_after
        for action in range(12):
            self.load_state(board_before)
            new_obs, _, _, _ = self.step(action)
            if np.array_equal(new_obs, board_after):
                return action
        return None

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return int(distribution.argmax())

    def state_to_repr(
        self,
        state: np.ndarray,
        title: str | None = None,
        repr_type: RepresentationType = RepresentationType.STR,
    ) -> ReadableReprT:
        # Only string representation for now
        if repr_type == RepresentationType.STR:
            return str(state)
        else:
            logger.warning(f'Only {RepresentationType.STR} is supported, not {repr_type}')
            return str(state)

    def many_states_to_repr(
        self,
        states: list[np.ndarray],
        titles: list[str],
        repr_type: RepresentationType = RepresentationType.STR,
    ) -> ReadableReprT:
        if repr_type == RepresentationType.STR:
            return '\n'.join(f'{title}: {state}' for state, title in zip(states, titles))
        else:
            logger.warning(f'Only {RepresentationType.STR} is supported, not {repr_type}')
            return '\n'.join(f'{title}: {state}' for state, title in zip(states, titles))

    def config(
        self,
        debug_level=DebugLevel.WARNING,
        render_cube=False,
        scramble_size=None,
        render_views=True,
        render_flat=True,
        step_limit=None,
    ):
        self = self if hasattr(self, 'cube') else None  # for staticmethod compatibility
        if self is not None:
            self.debugLevel = debug_level
            self.renderCube = render_cube
            if scramble_size is not None:
                self.scrambleSize = scramble_size
            if step_limit is not None:
                self.step_limit = step_limit
            self.renderViews = render_views
            self.renderFlat = render_flat
            if self.renderCube:
                plt.ion()
                plt.show()

    def render(self, mode='human', close=False):
        if self.renderCube:
            if hasattr(self, 'fig') and self.fig:
                plt.clf()
            self.fig = self.cube.render(self.fig if hasattr(self, 'fig') else None, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.001)

    def _take_action(self, action):
        self.cube.move_by_action(ACTION_LOOKUP[int(action)])

    @staticmethod
    def action_name(action):
        return ACTION_LOOKUP[int(action)].name

    def get_scramble(self):
        return self.scramble

    def valid_scramble_action(self, action, previous_actions):
        num_previous_actions = len(previous_actions)
        if (num_previous_actions > 2
                and previous_actions[num_previous_actions - 1] == previous_actions[num_previous_actions - 2]
                and action.name == previous_actions[num_previous_actions - 1]):
            return False
        if num_previous_actions > 1 and self.cube.opposite_actions(previous_actions[num_previous_actions - 1], action):
            return False
        return True

    def randomize(self, number):
        t = 0
        while t < number:
            action = ACTION_LOOKUP[np.random.randint(len(ACTION_LOOKUP.keys()))]
            if self.valid_scramble_action(action, self.scramble):
                self.scramble.append(action.name)
                self.cube.move_by_action(action)
                t += 1

    def _get_state(self):
        raw_state = self.cube.get_state()
        if self.obs_type == 'basic':
            state = (np.arange(6) == raw_state[..., np.newaxis]).astype(int)
        else:
            if self.converter is None:
                self.converter = CubeConverter()
            state = self.converter.convert_basic_to_reduced(raw_state)
        return state

    @property
    def state(self):
        return self._get_state()

    def load_state(self, desired_state):
        assert desired_state.shape == (6, 3, 3)
        self.cube.stickers = desired_state

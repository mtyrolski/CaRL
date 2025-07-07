import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from carl.environment.env import GameEnv
from carl.environment.gym_rubik.converter import CubeConverter
from carl.environment.gym_rubik.cube import Actions, Cube
from carl.environment.gym_rubik.rubik_solver import Move as rubik_solver_moves
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.environment.tokenizer import GameTokenizer
from joblib import dump
from loguru import logger
from torch import Tensor


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


class RubikEnv(GameEnv):
    def __init__(self, tokenizer: RubikCubeTokenizer, step_limit=100000000000000000000000, shuffles=50, obs_type='basic'):
        self._tokenizer = tokenizer
        self.cube = Cube(3, whiteplastic=False)
        self.solved_state = 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'
        self.obs_type = obs_type
        self.converter = CubeConverter() if obs_type != 'basic' else None
        self.scramble = []
        self.debugLevel = DebugLevel.WARNING
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = False
        self.scrambleSize = shuffles
        self.num_steps = 0
        self.step_limit = step_limit

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    @property
    def name(self) -> str:
        return "rubik_cube"

    def step(self, action: int) -> tuple[str, float, bool, dict]:
        self._take_action(action)
        reward = -1
        self.num_steps += 1
        obs = self.cube.get_state()
        obs_str = self.cube_state_to_str(obs)
        solved = self.is_solved(obs_str)
        if solved:
            reward = 0
        episode_over = solved
        return obs_str, reward, episode_over, {}

    def next_state(self, state: str, action: int) -> str:
        # Set the cube to the given state, take the action, and return the new state as string
        self.set_state(state)
        obs_str, _, _, _ = self.step(action)
        return obs_str

    def reset(self) -> str:
        self.cube = Cube(3, whiteplastic=False)
        self.scramble = []
        if self.scrambleSize > 0:
            if self.debugLevel == DebugLevel.INFO:
                print('scramble ' + str(self.scrambleSize) + ' moves')
            self.randomize(self.scrambleSize)
        self.num_steps = 0
        obs = self.cube.get_state()
        obs_str = self.cube_state_to_str(obs)
        return obs_str

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
        if (
            num_previous_actions > 2
            and previous_actions[num_previous_actions - 1]
            == previous_actions[num_previous_actions - 2]
            and action.name == previous_actions[num_previous_actions - 1]
        ):
            return False
        if num_previous_actions > 1 and self.cube.opposite_actions(
            previous_actions[num_previous_actions - 1], action
        ):
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
        return self.cube.get_state()

    def get_state(self) -> str:
        cube_ndarray_state: np.ndarray = self.cube.get_state()
        return self.cube_state_to_str(cube_ndarray_state)

    def set_state(self, state: str) -> None:
        self.cube.stickers = self.cube_str_to_state(state)

    def restore_full_state_from_np_array_version(self, state: str) -> None:
        self.set_state(state)

    def is_solved(self, board: str) -> bool:
        # Accepts a string, converts to ndarray for comparison
        return board == self.solved_state

    def detect_action(self, board_before: str, board_after: str) -> int | None:
        # Try all actions and see which one leads from board_before to board_after
        for action in range(12):
            self.set_state(board_before)
            obs_str, _, _, _ = self.step(action)
            if obs_str == board_after:
                return action
        return None

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return int(distribution.argmax())

    @staticmethod
    def cube_state_to_str(state: np.ndarray) -> str:
        # Converts a (6,3,3) ndarray to a canonical string of 54 chars
        # Use the same face order and rotation as in cube_str_to_state
        ordered_faces = [state[i] for i in [0, 5, 2, 4, 3, 1]]
        aligned_faces = [np.rot90(face, k=-1, axes=(0, 1)) for face in ordered_faces]
        flat = np.concatenate([face.flatten() for face in aligned_faces])
        return ''.join(['ywrogb'[int(label)] for label in flat])

    @staticmethod
    def cube_str_to_state(string_obs: str) -> np.ndarray:
        assert len(string_obs) == 54, f'Expected string of length 54, got {len(string_obs)} characters: {string_obs}'
        stickers = [RubikEnv.reverse_cube_labels()[x] for x in string_obs]
        indexes = np.eye(6)[stickers]
        faces = indexes.reshape((6, 3, 3, 6))
        # Reverse the rotation applied in cube_state_to_str
        aligned_faces = [np.rot90(face, k=1, axes=(0, 1)) for face in faces]
        # Reorder faces back to original order
        # The order [0, 5, 2, 4, 3, 1] was used for serialization, so we need to invert it
        # Find the mapping from [0, 5, 2, 4, 3, 1] to [0, 1, 2, 3, 4, 5]
        face_order = [0, 5, 2, 4, 3, 1]
        inverse_order = [face_order.index(i) for i in range(6)]
        ordered_faces = [aligned_faces[i] for i in inverse_order]
        return np.argmax(np.array(ordered_faces), axis=-1)

    @staticmethod
    def reverse_cube_labels() -> dict[str, int]:
        return {'y': 0, 'w': 1, 'r': 2, 'o': 3, 'g': 4, 'b': 5}

    @staticmethod
    def cube_labels() -> str:
        return 'ywrogb'

    def state_to_repr(self, state: str, title: str | None = None, repr_type=None):
        # Minimal: just return the string, optionally with a title
        if title:
            return f"{title}: {state}"
        return state

    def many_states_to_repr(self, states: list[str], titles: list[str], repr_type=None):
        # Minimal: join all states with their titles
        return '\n'.join(f"{t}: {s}" for s, t in zip(states, titles))

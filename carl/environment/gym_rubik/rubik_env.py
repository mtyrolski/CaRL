from enum import Enum

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from carl.environment.env import GameEnv
from carl.environment.gym_rubik.converter import CubeConverter
from carl.environment.gym_rubik.cube import Actions, Cube
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.utils.aliases import State
from carl.environment.tokenizer import GameTokenizer
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
    def __init__(self, tokenizer: RubikCubeTokenizer, step_limit=100000000000000000000000, shuffles=50, obs_type='basic', target_solved_state: str | None = None) -> None:
        self._tokenizer = tokenizer
        self.cube = Cube(3, whiteplastic=False)
        self.target_solved_state = target_solved_state or 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'
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
        self.fig: Figure | None = None

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    @property
    def name(self) -> str:
        return "rubik_cube"

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        observation, reward, episode_over, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :get
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = -1
        self.num_steps += 1

        observation = self._get_state()
        solved = self.is_solved(observation)

        if solved:
            reward = 0

        episode_over = solved

        return observation, reward, episode_over, {}

    def next_state(self, state: State, action: int) -> State:
        # Set the cube to the given state, take the action, and return the new state as ndarray
        self.set_state(state)
        obs, _, _, _ = self.step(action)
        # logger.debug(f'Next state after action {action}: {obs.shape} {obs.dtype}')

        if isinstance(obs, np.ndarray):
            # logger.debug(f'Next state is ndarray with shape {obs.shape} and dtype {obs.dtype}')
            # logger.debug(f'Input state was ndarray?: {isinstance(state, np.ndarray)}')
            return obs if isinstance(state, np.ndarray) else self.cube_bin_to_str(obs)
        else:
            assert isinstance(obs, str)
            # logger.debug(f'Next state is string with length {len(obs)}')
            # logger.debug(f'Input state was string?: {isinstance(state, str)}')
            # If the output is a string, convert it to ndarray if the input state was ndarray
            return self.cube_str_to_state(obs) if isinstance(state, np.ndarray) else obs
        
        # if isinstance(state, str) and isinstance(obs, np.ndarray):
        #     # If the input state was a string, convert the output to string as well
        #     return self.cube_bin_to_str(obs)
        # return obs

    def reset(self) -> np.ndarray:
        self.cube = Cube(3, whiteplastic=False)
        self.scramble = []
        if self.scrambleSize > 0:
            if self.debugLevel == DebugLevel.INFO:
                print('scramble ' + str(self.scrambleSize) + ' moves')
            self.randomize(self.scrambleSize)

        self.num_steps = 0
        return self._get_state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            if self.fig is not None:
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

    def get_state(self) -> State:
        cube_ndarray_state: np.ndarray = self.cube.get_state()
        return cube_ndarray_state

    def set_state(self, state: State) -> None:
        self.cube.stickers = state if isinstance(state, np.ndarray) else self.cube_str_to_state(state)

    def restore_full_state_from_np_array_version(self, state: np.ndarray) -> None:
        self.set_state(state)

    
    def is_solved(self, board: State) -> bool:
        # Here, we optionally support also str without conversion for compatibility
        if isinstance(board, str):
            # If board is already a string, we can directly check if it is solved
            return board == self.target_solved_state
        state_str = self.cube_bin_to_str(board)
        return state_str == self.target_solved_state

    def detect_action(self, board_before: State, board_after: State) -> int | None:
        # Try all actions and see which one leads from board_before to board_after
        for action in range(12):
            self.set_state(board_before)
            obs_str, _, _, _ = self.step(action)
            if obs_str == board_after:
                return action
        logger.warning(f"Could not detect single action from {board_before} to {board_after}")
        return None

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return int(distribution.argmax().item())

    def cube_str_to_state(self, string_obs: str) -> np.ndarray:
        return np.argmax(self.cube_str_to_bin(string_obs), axis=-1)

    def cube_str_to_bin(self, string_obs: str) -> np.ndarray:
        assert len(string_obs) == 54, f"Expected 54 stickers, got {len(string_obs)} in {string_obs}"

        stickers: list[int] = [self.reverse_cube_labels()[x] for x in string_obs]
        indexes: np.ndarray = np.eye(6)[stickers]
        faces: np.ndarray = indexes.reshape((6, 3, 3, 6))
        aligned_faces: np.ndarray = np.array([np.rot90(face, k=-1, axes=(0, 1)) for face in faces])
        ordered_faces: list[np.ndarray] = [aligned_faces[i] for i in [0, 5, 2, 4, 3, 1]]

        return np.array(ordered_faces)

    def cube_bin_to_str(self, stickers: np.ndarray) -> str:
        """
        stickers: ndarray of shape (6,3,3), with values 0–5 for ['y','w','r','o','g','b']
        returns: 54-char string in face-order [0,5,2,4,3,1], each face rotated back.
        """
        # 1) reorder faces
        face_order = [0, 5, 2, 4, 3, 1]
        ordered = [stickers[i] for i in face_order]

        # 2) rotate each face back to match cube_str_to_bin’s rotation
        aligned = [np.rot90(face, k=1, axes=(0,1)) for face in ordered]

        # 3) flatten to a single list of 54 IDs
        ids = np.concatenate([face.flatten() for face in aligned])

        # 4) map IDs→chars
        labels = self.cube_labels()  # should be 'ywrogb'
        return ''.join(labels[i] for i in ids)

    @staticmethod
    def reverse_cube_labels() -> dict[str, int]:
        return {'y': 0, 'w': 1, 'r': 2, 'o': 3, 'g': 4, 'b': 5}

    @staticmethod
    def cube_labels() -> str:
        return 'ywrogb'

    def state_to_repr(self, state: str, title: str | None = None, repr_type=None):  # type: ignore
        # Minimal: just return the string, optionally with a title
        if title:
            return f"{title}: {state}"
        return state

    def many_states_to_repr(self, states: list[str], titles: list[str], repr_type=None): # type: ignore
        # Minimal: join all states with their titles
        return '\n'.join(f"{t}: {s}" for s, t in zip(states, titles))

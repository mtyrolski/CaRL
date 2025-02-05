import os
from enum import Enum

import gym
import matplotlib.pyplot as plt
import numpy as np
from carl.environment.env import GameEnv
from carl.environment.gym_rubik.converter import CubeConverter
from carl.environment.gym_rubik.cube import Actions, Cube
from carl.environment.gym_rubik.rubik_solver import Move as rubik_solver_moves
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.environment.tokenizer import GameTokenizer
from gym import spaces
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


class RubikEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, step_limit=100000000000000000000000, shuffles=50, obs_type='basic'):
        self.cube = Cube(3, whiteplastic=False)
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.fig = None
        self.solved_state = self.cube.get_state()

        self.observation_space = None
        self.obs_type = obs_type
        self.converter = None
        self.create_observation_space()

        self.scramble = []

        self.debugLevel = DebugLevel.WARNING
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = False
        self.scrambleSize = shuffles

        self.num_steps = 0
        self.step_limit = step_limit

        self.config()

    def config(
        self,
        debug_level=DebugLevel.WARNING,
        render_cube=False,
        scramble_size=None,
        render_views=True,
        render_flat=True,
        step_limit=None,
    ):
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

    def create_observation_space(self):
        if self.obs_type == 'basic':
            self.observation_space = spaces.Box(low=0, high=1, shape=(6, 3, 3, 6), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(20, 24), dtype=np.float32)
            self.converter = CubeConverter()

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
        solved = np.array_equal(self.cube.get_state(), self.solved_state)

        if solved:
            reward = 0

        episode_over = solved

        return observation, reward, episode_over, {}

    def reset(self):
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
            if self.fig:
                plt.clf()
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
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
            state = self.converter.convert_basic_to_reduced(raw_state)
        return state

    @property
    def state(self):
        return self._get_state()

    def load_state(self, desired_state):
        assert desired_state.shape == (6, 3, 3)
        self.cube.stickers = desired_state


class RubikCubeEnv(GameEnv):
    name: str = 'rubik_cube'

    def __init__(self, tokenizer: RubikCubeTokenizer) -> None:
        self._tokenizer = tokenizer
        self.core: RubikEnv = RubikEnv()

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state: np.ndarray
        reward: float
        done: bool
        info: dict
        action = action % 12
        state, reward, done, info = self.core.step(action)

        return state, reward, done, info

    def next_state(self, state_before: str, action: int) -> tuple[str, float, bool, dict]:
        self.core.load_state(self.cube_str_to_state(state_before))
        state_after: np.ndarray
        reward: float
        done: bool
        info: dict
        action = action % 12
        state_after, reward, done, info = self.core.step(action)
        return self.cube_bin_to_str(state_after), reward, done, info

    def detect_action(self, board_before: str, board_after: str) -> int | None:
        """
        Detects the action that was taken to go from board_before to board_after.
        """
        assert len(board_before) == len(board_after) and len(board_before) == 54

        try:
            for action in range(12):
                self.core.load_state(self.cube_str_to_state(board_before))
                new_obs, _, _, _ = self.core.step(action)
                new_obs_str = self.cube_bin_to_str(new_obs)

                if new_obs_str == board_after:
                    return action
        except:
            logger.error(f'board_before: {board_before}')

    def cube_str_to_state(self, string_obs: str) -> np.ndarray:
        return np.argmax(self.cube_str_to_bin(string_obs), axis=-1)

    def cube_str_to_bin(self, string_obs: str) -> np.ndarray:
        assert len(string_obs) == 54

        stickers: list[int] = [self.reverse_cube_labels()[x] for x in string_obs]
        indexes: np.ndarray = np.eye(6)[stickers]
        faces: np.ndarray = indexes.reshape((6, 3, 3, 6))
        aligned_faces: np.ndarray = np.array([np.rot90(face, k=-1, axes=(0, 1)) for face in faces])
        ordered_faces: list[np.ndarray] = [aligned_faces[i] for i in [0, 5, 2, 4, 3, 1]]

        return np.array(ordered_faces)

    def cube_bin_to_str(self, binary_obs) -> str:
        ordered_faces = [binary_obs[i] for i in [0, 5, 2, 4, 3, 1]]
        aligned_faces: np.ndarray = np.array([np.rot90(face, axes=(0, 1)) for face in ordered_faces])
        sticker_list: np.ndarray = aligned_faces.reshape((-1, 6))
        string_obs: str = ''.join([self.cube_labels()[label] for label in np.where(sticker_list)[1]])
        return string_obs

    @staticmethod
    def reverse_cube_labels() -> dict[str, int]:
        return {'y': 0, 'w': 1, 'r': 2, 'o': 3, 'g': 4, 'b': 5}

    @staticmethod
    def cube_labels() -> str:
        return 'ywrogb'

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        """
        Converts a distribution to an action.
        """

        return distribution.argmax().item()

    def restore_full_state_from_np_array_version(self, state: str) -> None:
        self.core.load_state(self.cube_str_to_state(state))

    def get_state(self) -> str:
        return self.cube_bin_to_str(self.core.state)

    def is_solved(self, board: str) -> bool:
        solution: str = 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'
        return board == solution

    def state_to_repr(self, state: np.ndarray, title: str | None = None, file_name: str | None = None) -> None:
        return self.cube_str_to_bin(state)

    def many_states_to_repr(self, states: list[np.ndarray], titles: list[str]):
        pass

    def set_state(self, state: np.ndarray) -> None:
        pass

    def generate_training_data(
        self,
        number_of_trajectories: int,
        max_moves: int,
        path_to_save: str | None = None,
        save_after: int = 1,
        noisy_reverse_prob: float = 0.0,
    ) -> dict[int, list[str]]:
        observation_type: str = 'basic'

        env: RubikEnv = RubikEnv(step_limit=100, shuffles=0, obs_type=observation_type)

        trajectories: dict[int, list[str]] = {}
        part: int = 0
        num_of_trajectory: int = 0
        current_num_of_trajectories: int = 0

        while True:
            obs = env.reset()
            episode: list[str] = [self.cube_bin_to_str(obs)]
            solution = [
                self.move_to_action(move) for move in self.quarterize(
                    self.normalize_sequence(self.get_noisy_solution(max_moves, noisy_reverse_prob)))
            ]

            new_obs: np.ndarray
            rew: float
            done: bool
            info: dict

            for m in solution:
                new_obs, rew, done, info = env.step(m)
                obs_string: str = self.cube_bin_to_str(new_obs)
                episode.append(obs_string)

            episode: list[str] = list(reversed(episode))

            trajectories[num_of_trajectory] = episode
            num_of_trajectory += 1

            if current_num_of_trajectories >= number_of_trajectories:
                break

            if path_to_save is not None and len(trajectories) % save_after == 0:
                logger.info(f'Saving rubik_offline_data_part_{part}')
                dump(
                    trajectories,
                    os.path.join(path_to_save, f'rubik_offline_data_part_{part}.pkl'),
                    5,
                )
                part += 1
                current_num_of_trajectories += len(trajectories)
                trajectories = {}

        return trajectories

    def generate_eval_data(
        self,
        number_of_instances_to_eval: int,
        max_moves: int,
        path_to_save: str | None = None,
        save_after: int = 1,
        noisy_reverse_prob: float = 0.0,
    ) -> list[str]:
        observation_type: str = 'basic'

        env: RubikEnv = RubikEnv(step_limit=100, shuffles=0, obs_type=observation_type)

        data_to_eval: list[str] = []
        part: int = 0
        num_of_trajectory: int = 0
        current_num_of_trajectories: int = 0

        while True:
            obs = env.reset()
            episode: list[str] = [self.cube_bin_to_str(obs)]
            solution = [
                self.move_to_action(move) for move in self.quarterize(
                    self.normalize_sequence(self.get_noisy_solution(max_moves, noisy_reverse_prob)))
            ]

            new_obs: np.ndarray
            rew: float
            done: bool
            info: dict

            for m in solution:
                new_obs, rew, done, info = env.step(m)
                obs_string: str = self.cube_bin_to_str(new_obs)
                episode.append(obs_string)

            episode: list[str] = list(reversed(episode))

            data_to_eval.append(episode[0])
            num_of_trajectory += 1

            if current_num_of_trajectories >= number_of_instances_to_eval:
                break

            if path_to_save is not None and len(data_to_eval) % save_after == 0:
                logger.info(f'Saving rubik_eval_data_part_shuffle_{max_moves}_{part}')
                dump(
                    data_to_eval,
                    os.path.join(path_to_save, f'rubik_eval_data_part_shuffle_{max_moves}_{part}.pkl'),
                    5,
                )
                part += 1
                current_num_of_trajectories += len(data_to_eval)
                data_to_eval = []

        return data_to_eval

    def move_to_action(self, move) -> int:
        move_to_action_lookup: dict[str, int] = {v: k for k, v in self.get_action_to_move().items()}
        return move_to_action_lookup[str(move)]

    def normalize_sequence(self, sequence):
        # Remove X,Y,Z moves.
        res = []

        while len(sequence) > 0:
            move = sequence[0]
            sequence = sequence[1:]

            if move.face in ['X', 'Y', 'Z']:
                sequence = self.rotate_sequence(sequence, move)
            else:
                res.append(move)

        return res

    def get_raw_solution(self, max_moves: int):
        return [self.random_move() for _ in range(max_moves)]

    def get_noisy_solution(self, max_moves: int, noisy_reverse_prob: float):
        actions = []
        move_stack = []

        while len(move_stack) < max_moves:
            actions.append(np.random.choice(list(self.get_action_to_move().keys())))
            move_stack.append(actions[-1])

            while len(move_stack) > 0:
                if np.random.rand() < noisy_reverse_prob:
                    actions.append(self.reverse_move(move_stack[-1]))
                    move_stack.pop()
                else:
                    break

        return [rubik_solver_moves.Move(self.get_action_to_move()[action]) for action in actions]

    def random_move(self):
        sample: int = np.random.choice(list(self.get_action_to_move().values()))
        return rubik_solver_moves.Move(sample)

    @staticmethod
    def get_action_to_move() -> dict[int, str]:
        return {
            0: 'U',
            1: "U'",
            2: 'D',
            3: "D'",
            4: 'F',
            5: "F'",
            6: 'B',
            7: "B'",
            8: 'R',
            9: "R'",
            10: 'L',
            11: "L'",
        }

    def reverse_move(self, move: int):
        return move + (1 - 2 * (move % 2))

    @staticmethod
    def quarterize(moves):
        quarter_moves = []

        for move in moves:
            if move.double:
                move.double = False
                quarter_moves.append(move)
                quarter_moves.append(move)
            else:
                quarter_moves.append(move)

        return quarter_moves

    @staticmethod
    def rotate_sequence(sequence, move):
        change_move = {
            'X': {
                'U': 'F',
                'D': 'B',
                'L': 'L',
                'R': 'R',
                'F': 'D',
                'B': 'U',
                'X': 'X',
                'Y': 'Z',
                'Z': 'YP',
            },
            'Y': {
                'U': 'U',
                'D': 'D',
                'L': 'F',
                'R': 'B',
                'F': 'R',
                'B': 'L',
                'X': 'ZP',
                'Y': 'Y',
                'Z': 'X',
            },
            'Z': {
                'U': 'L',
                'D': 'R',
                'L': 'D',
                'R': 'U',
                'F': 'F',
                'B': 'B',
                'X': 'Y',
                'Y': 'XP',
                'Z': 'Z',
            },
        }

        def raw_rotate(sequence, face):
            res = []

            for move in sequence:
                if move.face in ['X', 'Y', 'Z']:
                    change = change_move[face][move.face]
                    move.face = change[0]
                    if 'p' in change:
                        move = move.reverse()
                else:
                    move.face = change_move[face][move.face]
                res.append(move)

            return res

        if move.double:
            return raw_rotate(raw_rotate(sequence, move.face), move.face)

        if move.counterclockwise:
            return raw_rotate(raw_rotate(raw_rotate(sequence, move.face), move.face), move.face)

        return raw_rotate(sequence, move.face)

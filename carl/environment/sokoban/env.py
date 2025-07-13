import itertools

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from torch import Tensor

from carl.environment.env import GameEnv
from carl.environment.env import ReadableReprT
from carl.environment.env import RepresentationType
from carl.environment.sokoban.core import _SokobanEnvCore
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.tokenizer import GameTokenizer
from carl.utils.aliases import State

class SokobanEnv(GameEnv):
    @property
    def name(self) -> str:
        return 'sokoban'

    def __init__(self, tokenizer: SokobanTokenizer, num_boxes=4) -> None:
        self._tokenizer = tokenizer
        self.dim_room = tokenizer.size_of_board
        self.num_boxes = num_boxes
        self.core = _SokobanEnvCore(dim_room=self.dim_room, num_boxes=num_boxes)
        self.done = False

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        state, reward, done, info = self.core.step(action)
        return state, reward, done, info

    def next_state(self, state: State, action: int) -> State:
        self.core.restore_full_state_from_np_array_version(state)
        state, _, done, _ = self.core.step(action)
        self.done = done
        return state

    def restore_full_state_from_np_array_version(self, state) -> None:
        self.core.restore_full_state_from_np_array_version(state)

    def set_state(self, state: State) -> None:
        self.core.restore_full_state_from_np_array_version(state)

    def get_state(self):
        return self.core.get_state()

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    def detect_action(self, board_before: State, board_after: State) -> int:
        x_before: int
        y_before: int
        x_after: int
        y_after: int
        x_before, y_before = self.get_agent_position(board_before)
        x_after, y_after = self.get_agent_position(board_after)
        delta_x: int = x_after - x_before
        delta_y: int = y_after - y_before

        return self.agent_coordinates_to_action(delta_x, delta_y)

    def get_agent_position(self, board: State) -> tuple[int, int]:
        width: int
        height: int
        assert isinstance(board, np.ndarray), 'Board must be a numpy array'
        width, height, _ = board.shape
        for xy in itertools.product(list(range(width)), list(range(height))):
            x: int
            y: int
            x, y = xy
            obj: str = self.get_field_name_from_index(int(np.argmax(board[x][y])))

            if obj == 'agent':
                return x, y

            if obj == 'agent_on_goal':
                return x, y

        raise AssertionError('No agent on the board')

    def num_boxes_on_target(self, state: State) -> int:
        box_on_target: int = 0
        assert isinstance(state, np.ndarray), 'State must be a numpy array'
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                if np.argmax(state[x][y]) == self.get_field_index_from_name('box_on_goal'):
                    box_on_target += 1

        return box_on_target

    def is_solved(self, board: State) -> bool:
        box_on_target: int = self.num_boxes_on_target(board)
        return box_on_target == self.num_boxes

    @staticmethod
    def get_field_name_from_index(x: int) -> str:
        objects: dict[int, str] = {
            0: 'wall',
            1: 'empty',
            2: 'goal',
            3: 'box_on_goal',
            4: 'box',
            5: 'agent',
            6: 'agent_on_goal',
        }
        return objects[x]

    @staticmethod
    def agent_coordinates_to_action(delta_x: int, delta_y: int) -> int:
        """
        Returns action corresponding to given change of agent's coordinates (-1, 0 or 1).

        Correspondence was taken from
        here: https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py#L166
        """
        assert delta_x in (-1, 0, 1), 'Wrong value for delta_x argument'
        assert delta_y in (-1, 0, 1), 'Wrong value for delta_y argument'

        translation: dict[tuple[int, int], int] = {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3,
        }
        translation_key: tuple[int, int] = (delta_x, delta_y)

        assert translation_key in translation, 'Action should consists of exactly one move'

        return translation[translation_key]

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return int(distribution.argmax(dim=-1))

    @staticmethod
    def get_field_index_from_name(x: str) -> int:
        objects_class = {
            'wall': 0,
            'empty': 1,
            'goal': 2,
            'box_on_goal': 3,
            'box': 4,
            'agent': 5,
            'agent_on_goal': 6,
        }
        return objects_class[x]

    def _state_to_rgb(self, state: State) -> State:
        self.core.restore_full_state_from_np_array_version(state)
        return self.core.render(mode='rgb_array').astype(int)

    def state_to_repr(
        self,
        state: State,
        title: str | None = None,
        repr_type: RepresentationType = RepresentationType.PLT_FIGURE,
    ) -> ReadableReprT:

        if repr_type != RepresentationType.PLT_FIGURE:
            logger.warning(f'Only {RepresentationType.PLT_FIGURE} is supported, not {repr_type}')
            return "" # type: ignore

        pic = self._state_to_rgb(state)
        fig, ax = plt.subplots()
        ax.imshow(pic)
        ax.axis('off')

        if title is not None:
            ax.set_title(title)

        # Clear the current figure to avoid automatic display
        plt.close(fig)

        return fig # type: ignore

    def many_states_to_repr(
        self,
        states: list[State],
        titles: list[str],
        repr_type: RepresentationType = RepresentationType.PLT_FIGURE,
    ) -> ReadableReprT:
        if repr_type != RepresentationType.PLT_FIGURE:
            logger.warning(f'Only {RepresentationType.PLT_FIGURE} is supported, not {repr_type}')
            return ""

        def draw_and_describe(ax, state, title):
            pic = self._state_to_rgb(state)
            ax.set_title(title)
            ax.imshow(pic)
            ax.axis('off')

        # Create a figure and subplots
        n_states = len(states)
        fig, axes = plt.subplots(1, n_states, figsize=(3 * n_states, 3))

        # If there's only one state, `axes` will not be an array, so we wrap it in a list
        if n_states == 1:
            axes = [axes]

        # Draw each state on its corresponding subplot
        for idx, ax in enumerate(axes):
            draw_and_describe(ax, states[idx], titles[idx])

        # Close the figure to prevent automatic display in the notebook
        plt.close(fig)

        return fig # type: ignore


### Start Core Sokoban - ported and adapted to carl from https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py
import enum
import numpy as np
import importlib.resources as resources
from carl.environment.utilis import HashableState
from PIL import Image

RENDERING_MODES = ['one_hot', 'rgb_array', 'tiny_rgb_array']


def load_surfaces():
    # Necessarily keep the same order as in FieldStates
    assets_file_name = [
        'wall.png',
        'floor.png',
        'box_target.png',
        'box_on_target.png',
        'box.png',
        'player.png',
        'player_on_target.png',
    ]
    sizes = ['8x8pixels', '16x16pixels']

    # resource_package = __name__.replace('.core', '.env')
    resource_package = '.'.join(__name__.split('.')[:-1])
    print(f"Loading assets from package: {resource_package}")
    surfaces = {}
    for size in sizes:
        surfaces[size] = []
        for asset_file_name in assets_file_name:
            # Use importlib.resources to get the path to the asset
            with resources.path(f"{resource_package}.surface.{size}", asset_file_name) as asset_path:
                asset_np_array = np.array(Image.open(asset_path))
                surfaces[size].append(asset_np_array)
        surfaces[size] = np.stack(surfaces[size])

    return surfaces


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes, dtype=np.uint8)[a.reshape(-1)]).reshape(a.shape + (num_classes,))


class _SokobanEnvCore:
    metadata = {'render.modes': RENDERING_MODES}

    def __init__(
        self,
        dim_room=(10, 10),
        max_steps=np.inf,
        num_boxes=4,
        num_gen_steps=None,
        mode='one_hot',
        fast_state_eq=False,
        penalty_for_step=-0.1,
        reward_box_on_target=1,
        reward_finished=10,
        seed=None,
        load_boards_from_file=None,
        load_boards_lazy=True,
    ):
        self._seed = seed
        self.mode = mode
        self.num_gen_steps = num_gen_steps
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes

        # Penalties and Rewards
        self.penalty_for_step = penalty_for_step
        self.reward_box_on_target = reward_box_on_target
        self.reward_finished = reward_finished

        self._internal_state = None
        self.fast_state_eq = fast_state_eq
        self._surfaces = load_surfaces()
        self.initial_internal_state_hash = None
        self.load_boards_from_file = load_boards_from_file
        self.boards_from_file = None
        if not load_boards_lazy:
            self.boards_from_file = np.load(self.load_boards_from_file)

    def step(self, action):
        raw_state, rew, done = self._step(
            self._internal_state.get_raw(),
            action,
            self.penalty_for_step,
            self.reward_box_on_target,
            self.reward_finished,
        )
        self._internal_state = HashableState(*raw_state, fast_eq=self.fast_state_eq)
        return self._internal_state.one_hot, rew, done, {'solved': done}

    def render(self, mode='one_hot'):
        assert mode in RENDERING_MODES, f'Only {RENDERING_MODES} are supported, not {mode}'
        if mode == 'one_hot':
            return self._internal_state.one_hot
        render_surfaces = None
        if mode == 'rgb_array':
            render_surfaces = self._surfaces['16x16pixels']
        if mode == 'tiny_rgb_array':
            render_surfaces = self._surfaces['8x8pixels']

        size_x = self._internal_state.one_hot.shape[0] * render_surfaces.shape[1]
        size_y = self._internal_state.one_hot.shape[1] * render_surfaces.shape[2]

        res = np.tensordot(self._internal_state.one_hot, render_surfaces, (-1, 0))
        res = np.transpose(res, (0, 2, 1, 3, 4))
        return np.reshape(res, (size_x, size_y, 3))

    def clone_full_state(self):
        internal_state = self._internal_state
        internal_state._initial_state_hash = self.initial_internal_state_hash
        return internal_state

    def get_state(self):
        return self._internal_state.get_np_array_version()

    def restore_full_state(self, state):
        self._internal_state = state
        self.initial_internal_state_hash = state._initial_state_hash

    def restore_full_state_from_np_array_version(self, state_np, quick=False):
        if (state_np > 255).any() or (state_np < 0).any():
            raise ValueError(f'restore_full_state_from_np_array_version() got '
                             f'data out of range 0-255 {state_np}')
        if quick:
            agent_pos = None
            unmatched_boxes = None
        else:
            shape = state_np.shape[:2]
            agent_pos = np.unravel_index(
                np.argmax(state_np[..., _SokobanEnvCore.FieldStates.player] +
                          state_np[..., _SokobanEnvCore.FieldStates.player_target]),
                shape=shape,
            )
            unmatched_boxes = int(np.sum(state_np[..., _SokobanEnvCore.FieldStates.box]))
        self._internal_state = HashableState(state_np, agent_pos, unmatched_boxes, fast_eq=self.fast_state_eq)

    class FieldStates(enum.IntEnum):
        wall = 0
        empty = 1
        target = 2
        box_target = 3
        box = 4
        player = 5
        player_target = 6

    def _step(self, state, action, penalty_for_step, reward_box_on_target, reward_finished):
        empty = 1
        target = 2
        box_target = 3
        box = 4
        player = 5
        player_target = 6

        delta_x, delta_y = None, None
        if action == 0:
            delta_x, delta_y = -1, 0
        elif action == 1:
            delta_x, delta_y = 1, 0
        elif action == 2:
            delta_x, delta_y = 0, -1
        elif action == 3:
            delta_x, delta_y = 0, 1

        one_hot_repr, agent_pos, unmatched_boxes = state

        arena = np.zeros(shape=(3,), dtype=np.uint8)
        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y
            if index_x < one_hot_repr.shape[0] and index_y < one_hot_repr.shape[0]:
                arena[i] = np.where(one_hot_repr[index_x, index_y, :] == 1)[0][0]

        new_unmatched_boxes_ = unmatched_boxes
        new_agent_pos = agent_pos
        new_arena = np.copy(arena)

        box_moves = (arena[1] == box or arena[1] == box_target) and (arena[2] == empty or arena[2] == 2)

        agent_moves = arena[1] == empty or arena[1] == target or box_moves

        if agent_moves:
            targets = ((arena == target).astype(np.int8) + (arena == box_target).astype(np.int8) +
                       (arena == player_target).astype(np.int8))
            if box_moves:
                last_field = box - 2 * targets[2]    # Weirdness due to inconsistent target non-target
            else:
                last_field = arena[2] - targets[2]

            new_arena = np.array([empty, player, last_field]).astype(np.uint8) + targets.astype(np.uint8)
            new_agent_pos = (agent_pos[0] + delta_x, agent_pos[1] + delta_y)

            if box_moves:
                new_unmatched_boxes_ = int(unmatched_boxes - (targets[2] - targets[1]))

        new_one_hot = np.copy(one_hot_repr)
        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y
            if index_x < one_hot_repr.shape[0] and index_y < one_hot_repr.shape[0]:
                one_hot_field = np.zeros(shape=7)
                one_hot_field[new_arena[i]] = 1
                new_one_hot[index_x, index_y, :] = one_hot_field

        done = new_unmatched_boxes_ == 0
        reward = penalty_for_step - reward_box_on_target * (float(new_unmatched_boxes_) - float(unmatched_boxes))
        if done:
            reward += reward_finished

        new_state = (new_one_hot, new_agent_pos, new_unmatched_boxes_)

        return new_state, reward, done

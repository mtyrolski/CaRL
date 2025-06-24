from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import matplotlib.pyplot as plt_m
import numpy as np
from crafter import constants
from crafter import objects
from crafter.engine import ItemView
from crafter.engine import LocalView
from crafter.engine import SemanticView
from crafter.engine import World
from crafter.env import Env as CrafterEnv
from crafter.objects import Player
from matplotlib import figure as plt
from plotly import graph_objects as go
from torch import Tensor

from carl.environment.env import GameEnv
from carl.environment.env import ReadableReprT
from carl.environment.tokenizer import GameTokenizer

"""
(64, 64, 3)
(64, 64, 3) 0.0 False {'inventory': {'health': 9, 'food': 9, 'drink': 9, 'energy': 9, 'sapling': 0, 'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0, 'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0, 'wood_sword': 0, 'stone_sword': 0, 'iron_sword': 0}, 'achievements': {'collect_coal': 0, 'collect_diamond': 0, 'collect_drink': 0, 'collect_iron': 0, 'collect_sapling': 0, 'collect_stone': 0, 'collect_wood': 0, 'defeat_skeleton': 0, 'defeat_zombie': 0, 'eat_cow': 0, 'eat_plant': 0, 'make_iron_pickaxe': 0, 'make_iron_sword': 0, 'make_stone_pickaxe': 0, 'make_stone_sword': 0, 'make_wood_pickaxe': 0, 'make_wood_sword': 0, 'place_furnace': 0, 'place_plant': 0, 'place_stone': 0, 'place_table': 0, 'wake_up': 0}, 'discount': 1.0, 'semantic': array([[ 2,  2, 14, ...,  3,  3,  3],
       [ 2,  2,  2, ...,  3,  3,  3],
       [ 2,  2,  2, ...,  3,  3,  4],
       ...,
       [ 1,  1,  1, ...,  1,  1,  2],
       [ 1,  1,  1, ...,  5,  2,  2],
       [ 1,  1,  1, ...,  6,  2,  6]], dtype=uint8), 'player_pos': array([31, 32]), 'reward': 0.0}
"""

ReadableReprT = TypeVar('RepresentationT', str, go.Figure, plt.Figure)
 
class CrafterTarget(StrEnum):
    COLLECT_COAL = "collect coal"
    COLLECT_DIAMOND = "collect diamond"
    COLLECT_SAPLING = "collect sapling"
    COLLECT_STONE = "collect stone"
    COLLECT_DRINK = "collect drink"
    COLLECT_WOOD = "collect wood"
    COLLECT_ALL = "collect all"
    DEFEAT_SKELETON = "defeat skeleton"
    DEFEAT_ZOMBIE = "defeat zombie"
    DEFEAT_ALL = "defeat all"
    EAT_COW = "eat cow"
    EAT_PLANT = "eat plant"
    EAT_ALL = "eat all"
    MAKE_IRON_PICKAXE = "make iron pickaxe"
    MAKE_IRON_SWORD = "make iron sword"
    MAKE_STONE_PICKAXE = "make stone pickaxe"
    MAKE_STONE_SWORD = "make stone sword"
    MAKE_WOOD_PICKAXE = "make wood pickaxe"
    MAKE_WOOD_SWORD = "make wood sword"
    MAKE_ALL = "make all"
    PLACE_FURNACE = "place furnace"
    PLACE_PLANT = "place plant"
    PLACE_STONE = "place stone"
    PLACE_TABLE = "place table"
    PLACE_ALL = "place all"
    WAKE_UP = "wake up"
    
Pos2D = tuple[int, int]

@dataclass
class CrafterState:
    world: World
    player: Player
    episode: int
    step: int
    seed: int
    unlocked: set[str]
    last_health: int
    daylight: float
    inventory: dict
    achievements: dict
    player_pos: tuple[int, int]
    facing: tuple[int, int]
    action: str
    sleeping: bool
    hunger: int
    thirst: int
    fatigue: int
    recover: int


class CrafterCarlEnv(GameEnv):
    """
    Crafter (https://github.com/danijar/crafter) integrated into the CARL GameEnv interface.
    The 'target' parameter specifies a certain achievement or set of achievements that marks
    the environment as solved.
    
    CARL env interface models crafter problem instance as following state:    
    """

    def __init__(self, 
                 target: CrafterTarget,
                 tokenizer: GameTokenizer,
                 **crafter_kwargs):
        self.core_env = CrafterEnv(**crafter_kwargs)
        self.target = target
        self.core_env._tokenizer = tokenizer
        self.core_env._last_info = {}  # We store the most recent info dict here
        
    def get_state(self) -> CrafterState:
        return CrafterState(
            world=self.core_env._world,
            player=self.core_env._player,
            episode=self.core_env._episode,
            step=self.core_env._step,
            seed=self.core_env._seed,
            unlocked=self.core_env._unlocked,
            last_health=self.core_env._last_health,
            daylight=self.core_env._world.daylight,
            inventory=self.core_env._player.inventory.copy(),
            achievements=self.core_env._player.achievements.copy(),
            player_pos=self.core_env._player.pos,
            facing=self.core_env._player.facing,
            action=self.core_env._player.action,
            sleeping=self.core_env._player.sleeping,
            hunger=self.core_env._player._hunger,
            thirst=self.core_env._player._thirst,
            fatigue=self.core_env._player._fatigue,
            recover=self.core_env._player._recover
        )

    @property
    def name(self) -> str:
        return f"Crafter-{self.target}"

    @property
    def tokenizer(self) -> GameTokenizer:
        # If a tokenizer is needed, return it here. Otherwise return None or a stub.
        return self.core_env._tokenizer

    def detect_action(self, board_before: np.ndarray, board_after: np.ndarray) -> int | None:
        """
        Currently not implemented; returns None.
        """
        raise NotImplementedError("Action detection is not implemented for Crafter.")

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        """
        Takes the argmax of the distribution as the action.
        """
        return distribution.argmax().item()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Step the underlying Crafter environment with the given action.
        Provide a reward of 1.0 only if the environment is now solved, otherwise 0.0.
        """
        next_board, _, _, info = self.core_env.step(action)
        self.core_env._last_info = info

        done = self.is_solved(next_board)
        reward = 1.0 if done else 0.0
        return next_board, reward, done, info

    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Return a hypothetical next state if we took 'action' in the current environment.
        This uses the underlying environment's 'step' but doesn't revert afterwards,
        so it modifies the environment state. Use with caution if you need a pure function.
        """
        
        next_board, _, _, _ = self.core_env.step(action)
        return next_board

    def is_solved(self, board: np.ndarray) -> bool:
        """
        Check the stored achievements in self.core_env._last_info to decide if the target is fulfilled.
        'board' is unused here because Crafter achievements are stored in info.
        """
        achievements = self.core_env._last_info.get("achievements", {})
        required = self._targets_for_crafter(self.target)
        return all(achievements.get(key, 0) > 0 for key in required)

    def state_to_repr(self, state: CrafterState, title: str | None = None) -> ReadableReprT:
        """
        Return a matplotlib Figure showing the state image with optional title.
        """
        self.set_state(state)
        default_size = (600, 600)
        image = self.core_env.render(size=default_size)  # Returns a NumPy array (H,W,3)

        fig = plt_m.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
            
        # Clear the current figure to avoid automatic display
        plt_m.close(fig)

        return fig
    

    def many_states_to_repr(self, states: list[CrafterState], title: str | None = None) -> ReadableReprT:
        """
        Return a single matplotlib Figure showing multiple states side by side.
        """
        if not states:
            return "No states provided."

        fig, axes = plt_m.subplots(1, len(states), figsize=(6*len(states), 6))
        if title is not None:
            fig.suptitle(title)

        if len(states) == 1:
            axes = [axes]  # Ensure it's iterable

        default_size = (600, 600)
        for i, state in enumerate(states):
            self.set_state(state)
            image = self.core_env.render(size=default_size)
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"State {i}")

        return fig


    def set_state(self, state: CrafterState):
        """
        Set the environment state to match the given CrafterState object.
        """
        self.core_env._world = state.world
        self.core_env._episode = state.episode
        self.core_env._step = state.step
        self.core_env._seed = state.seed
        self.core_env._unlocked = state.unlocked
        self.core_env._last_health = state.last_health
        self.core_env._world.daylight = state.daylight
        self.core_env._player.inventory = state.inventory
        self.core_env._player.achievements = state.achievements
        self.core_env._player.pos = state.player_pos
        self.core_env._player.facing = state.facing
        self.core_env._player.action = state.action
        self.core_env._player.sleeping = state.sleeping
        self.core_env._player._hunger = state.hunger
        self.core_env._player._thirst = state.thirst
        self.core_env._player._fatigue = state.fatigue
        self.core_env._player._recover = state.recover

        view = self.core_env._view

        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self.core_env._local_view = LocalView(world=state.world,
                                              textures=self.core_env._textures,
                                              grid=[view[0], view[1] - item_rows])
        
        self.core_env._item_view = ItemView(self.core_env._textures, [view[0], item_rows])
        self.core_env._sem_view = SemanticView(self.core_env._world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant])

        # Debugging output
        print("State set:")
        print(f"Player position: {self.core_env._player.pos}")
        print(f"Player health: {self.core_env._player.health}")
        print(f"Achievements: {self.core_env._player.achievements}")
        print(f"World step: {self.core_env._step}")

    def _targets_for_crafter(self, target: CrafterTarget) -> list[str]:
        """
        Returns a list of achievement keys that must all be > 0 to consider the target solved.
        """
        if target == CrafterTarget.COLLECT_COAL:
            return ["collect_coal"]
        elif target == CrafterTarget.COLLECT_DIAMOND:
            return ["collect_diamond"]
        elif target == CrafterTarget.COLLECT_SAPLING:
            return ["collect_sapling"]
        elif target == CrafterTarget.COLLECT_STONE:
            return ["collect_stone"]
        elif target == CrafterTarget.COLLECT_DRINK:
            return ["collect_drink"]
        elif target == CrafterTarget.COLLECT_WOOD:
            return ["collect_wood"]
        elif target == CrafterTarget.COLLECT_ALL:
            return [
                "collect_coal", "collect_diamond", "collect_drink",
                "collect_iron", "collect_sapling", "collect_stone", "collect_wood"
            ]
        elif target == CrafterTarget.DEFEAT_SKELETON:
            return ["defeat_skeleton"]
        elif target == CrafterTarget.DEFEAT_ZOMBIE:
            return ["defeat_zombie"]
        elif target == CrafterTarget.DEFEAT_ALL:
            return ["defeat_skeleton", "defeat_zombie"]
        elif target == CrafterTarget.EAT_COW:
            return ["eat_cow"]
        elif target == CrafterTarget.EAT_PLANT:
            return ["eat_plant"]
        elif target == CrafterTarget.EAT_ALL:
            return ["eat_cow", "eat_plant"]
        elif target == CrafterTarget.MAKE_IRON_PICKAXE:
            return ["make_iron_pickaxe"]
        elif target == CrafterTarget.MAKE_IRON_SWORD:
            return ["make_iron_sword"]
        elif target == CrafterTarget.MAKE_STONE_PICKAXE:
            return ["make_stone_pickaxe"]
        elif target == CrafterTarget.MAKE_STONE_SWORD:
            return ["make_stone_sword"]
        elif target == CrafterTarget.MAKE_WOOD_PICKAXE:
            return ["make_wood_pickaxe"]
        elif target == CrafterTarget.MAKE_WOOD_SWORD:
            return ["make_wood_sword"]
        elif target == CrafterTarget.MAKE_ALL:
            return [
                "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
                "make_stone_sword", "make_wood_pickaxe", "make_wood_sword"
            ]
        elif target == CrafterTarget.PLACE_FURNACE:
            return ["place_furnace"]
        elif target == CrafterTarget.PLACE_PLANT:
            return ["place_plant"]
        elif target == CrafterTarget.PLACE_STONE:
            return ["place_stone"]
        elif target == CrafterTarget.PLACE_TABLE:
            return ["place_table"]
        elif target == CrafterTarget.PLACE_ALL:
            return ["place_furnace", "place_plant", "place_stone", "place_table"]
        elif target == CrafterTarget.WAKE_UP:
            return ["wake_up"]
        else:
            return []
    
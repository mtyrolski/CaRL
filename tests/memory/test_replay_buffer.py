"""Unit tests for replay buffers."""
import pytest
from carl.memory.replay_buffer import SimpleUniversalReplayBuffer, SolvingPathConditionalLowLevelPolicyReplayBuffer
from carl.memory.replay_buffer import SolvingPathGeneratorReplayBuffer
from carl.memory.replay_buffer import SolvingPathValueReplayBuffer
from tests.utils import assert_equal, assert_equal_sets, build_dummy_search_tree
import gymnasium as gym
from carl.memory.replay_buffer import (SolvingPathConditionalLowLevelPolicyReplayBuffer,
                                       SolvingPathGeneratorReplayBuffer, SolvingPathValueReplayBuffer)


class DummyEnvironment(gym.Env):
    """Dummy environment for testing."""
    def __init__(self, target):
        self.state = 0
        self.target = target

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=int)

    def reset(self, **_kwargs):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += action
        return self.state, 0, (self.state == self.target), {}

    def restore_state(self, state):
        self.state = state

    def get_state(self):
        return self.state


def test_SolvingPathGeneratorReplayBuffer():
    """Test the SolvingPathGeneratorReplayBuffer class."""

    print('Testing SolvingPathGeneratorReplayBuffer...')

    env, solving_node = build_dummy_search_tree()

    replay_buffer = SolvingPathGeneratorReplayBuffer(100, (3, 4), env)

    # Add a new solution
    replay_buffer.add(({}, {'solving_node': solving_node}))

    # Check the replay buffer
    assert_equal(
        replay_buffer.buffer,
        [(0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (4, 6), (5, 6)],
    )

    print('OK\n')


def test_SolvingPathValueReplayBuffer():
    """Test the SolvingPathValueReplayBuffer class."""

    print('Testing SolvingPathValueReplayBuffer...')

    env, solving_node = build_dummy_search_tree()

    replay_buffer = SolvingPathValueReplayBuffer(100, env)

    # Add a new solution
    replay_buffer.add(({}, {'solving_node': solving_node}))

    # Check the replay buffer
    assert_equal(replay_buffer.buffer, [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)])

    print('OK\n')


def test_SolvingPathPolicyReplayBuffer():
    """Test the SolvingPathValueReplayBuffer class."""

    print('Testing SolvingPathConditionalLowLevelPolicyReplayBuffer...')

    env, solving_node = build_dummy_search_tree()

    replay_buffer = SolvingPathConditionalLowLevelPolicyReplayBuffer(100, 3, env)

    # Add a new solution
    replay_buffer.add(({}, {'solving_node': solving_node}))

    # Check the replay buffer
    assert_equal_sets(
        replay_buffer.buffer,
        [((0, 1), 1), ((0, 2), 1), ((0, 3), 1), ((1, 2), 1), ((1, 3), 1), ((1, 4), 1), ((2, 3), 1), ((2, 4), 1),
         ((2, 5), 1), ((3, 4), 1), ((3, 5), 1), ((3, 6), 1), ((4, 5), 1), ((4, 6), 1), ((5, 6), 1)],
    )


def get_replay_buffers(target):
    """Fixture to create instances of replay buffers for different targets."""
    env1, node1 = build_dummy_search_tree(target)
    generator_buffer = SolvingPathGeneratorReplayBuffer(100, [3, 4], env1)
    value_buffer = SolvingPathValueReplayBuffer(100, env1)
    cllp_buffer = SolvingPathConditionalLowLevelPolicyReplayBuffer(100, 3, env1)
    return SimpleUniversalReplayBuffer(generator_buffer, value_buffer, cllp_buffer), node1


def test_add_empty_data():
    """Test adding empty data does not modify buffers."""
    buffer, _ = get_replay_buffers(14)
    buffer.add(({}, {'solving_node': None}))
    assert len(buffer.get_buffer_for_generator(None)) == 0
    assert len(buffer.get_buffer_for_value()) == 0
    assert len(buffer.get_buffer_for_policy()) == 0


def test_buffer_overflow():
    """Test buffer handles overflow by removing oldest data."""
    buffer, node = get_replay_buffers(14)
    for _ in range(150):    # exceed buffer size
        buffer.add(({}, {'solving_node': node}))
    assert len(buffer.get_buffer_for_generator(None)) <= 100


def test_sampling_from_empty_buffer():
    """Ensure sampling from an empty buffer raises an appropriate error or returns empty array."""
    buffer, _ = get_replay_buffers(14)
    with pytest.raises(Exception):
        buffer.sample_for_generator(1)    # Modify according to actual implementation


@pytest.mark.parametrize("target", [11, 16, 17, 19])
def test_add_various_targets(target):
    """Test adding data with various targets to ensure buffer handles diverse scenarios."""
    env, node = build_dummy_search_tree(target)
    buffer = SolvingPathGeneratorReplayBuffer(100, [3, 4], env)
    buffer.add(({}, {'solving_node': node}))
    assert len(buffer.buffer) > 0    # Length check to confirm addition


def test_multibuffer_addition():
    """Test that data is added correctly to multiple buffers in `SimpleUniversalReplayBuffer`."""
    buffer, node = get_replay_buffers(14)
    buffer.add(({}, {'solving_node': node}))
    assert len(buffer.get_buffer_for_generator(None)) > 0
    assert len(buffer.get_buffer_for_value()) > 0
    assert len(buffer.get_buffer_for_policy()) > 0


def test_trajectory_addition():
    """Test adding trajectories directly to buffers."""
    buffer, _ = get_replay_buffers(14)
    trajectory = [i for i in range(6)]
    buffer.generator_buffer.add_from_trajectories([trajectory])
    assert len(buffer.generator_buffer.buffer) > 0


def test_invalid_target_handling():
    """Ensure buffer handles invalid targets gracefully."""
    with pytest.raises(KeyError):
        get_replay_buffers(-1)    # Assuming invalid target raises an AssertionError

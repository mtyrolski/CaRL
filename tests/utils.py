from carl.solver.nodes import SearchTreeNode


def assert_equal(exp1, exp2):
    assert exp1 == exp2, f'{exp1} != {exp2}'


def assert_equal_sets(exp1, exp2):
    return set(exp1) == set(exp2), f'{set(exp1)} != {set(exp2)}'


class DummyGameTokenizer:
    """Dummy game tokenizer for testing."""
    def board_tokenizer(self, board):
        return board

    def board_detokenizer(self, sequence_of_tokens):
        return sequence_of_tokens

    def x_y_tokenizer(self, x, y, training_goal):
        _ = training_goal
        return x, y

    def action_detokenizer(self, sequence_of_tokens):
        return sequence_of_tokens


class DummyEnvironment:
    """Dummy environment for testing."""
    def __init__(self, target):
        self.state = 0
        self.target = target

        self.observation_space = None    # gym.spaces.Box(low=0, high=100, shape=(1,), dtype=int)
        self.tokenizer = DummyGameTokenizer()

    def reset(self, **_kwargs):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += action
        return self.state, 0, (self.state == self.target), {}

    def restore_state(self, state):
        self.state = state

    def restore_full_state_from_np_array_version(self, state):
        self.state = state

    def get_state(self):
        return self.state


def build_dummy_search_tree(target: int = 6):
    """
    Build a dummy search tree for testing.

    Brackets indicate the high-level subgoals.

                  9 - 10 - [11]        18 - [19]
                /                     /
              [7] - [8]       15 - [16] - [17]
             /               /
    (root) [0] - 1 - [2] - [3] - 4 - 5 - [6] (target)
                       \
                       12 - 13 - [14]
    """

    node_0 = SearchTreeNode(0, None, None, None)
    node_2 = SearchTreeNode(2, None, [1, 1], node_0)
    node_3 = SearchTreeNode(3, None, [1], node_2)
    node_6 = SearchTreeNode(6, None, [1, 1, 1], node_3)
    node_7 = SearchTreeNode(7, None, [7], node_0)
    node_8 = SearchTreeNode(8, None, [1], node_7)
    node_11 = SearchTreeNode(11, None, [2, 1, 1], node_7)
    node_14 = SearchTreeNode(14, None, [10, 1, 1], node_2)
    node_16 = SearchTreeNode(16, None, [12, 1], node_3)
    node_17 = SearchTreeNode(17, None, [1], node_16)
    node_19 = SearchTreeNode(19, None, [2, 1], node_16)

    nodes_mapping = {
        0: node_0,
        2: node_2,
        3: node_3,
        6: node_6,
        7: node_7,
        16: node_16,
        8: node_8,
        11: node_11,
        14: node_14,
        17: node_17,
        19: node_19,
    }

    return DummyEnvironment(target), nodes_mapping[target]

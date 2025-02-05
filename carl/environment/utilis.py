import numpy as np

from carl.environment.env import GameEnv


class HashableState:
    state = np.random.get_state()
    np.random.seed(0)
    hash_key = np.random.normal(size=10000)
    np.random.set_state(state)

    def __init__(self, one_hot, agent_pos, unmached_boxes, fast_eq=False):
        self.one_hot = one_hot
        self.agent_pos = agent_pos
        self.unmached_boxes = unmached_boxes
        self._hash = None
        self.fast_eq = fast_eq
        self._initial_state_hash = None

    def __iter__(self):
        yield from [self.one_hot, self.agent_pos, self.unmached_boxes]

    def __hash__(self):
        if self._hash is None:
            flat_np = self.one_hot.flatten()
            self._hash = int(np.dot(flat_np, HashableState.hash_key[:len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        if self.fast_eq:
            return hash(self) == hash(other)    # This is a conscious decision to speed up.
        else:
            return np.array_equal(self.one_hot, other.one_hot)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_raw(self):
        return self.one_hot, self.agent_pos, self.unmached_boxes

    def get_np_array_version(self):
        return self.one_hot


class DeadEndFinder:
    """Checks whether the given states are dead ends."""
    def __init__(self, env: GameEnv, n_actions):
        self.visited = {}
        self.env = env
        self.n_actions = n_actions

    def check_naive(self, state):
        # A simple dead-end test.

        conv = state[1:, 1:] + state[1:, :-1] + state[:-1, :-1] + state[:-1, 1:]

        if np.any(np.all(conv == [3, 0, 0, 0, 1, 0, 0], axis=-1)):
            # box in a corner
            return True

        if np.any(np.all(conv == [2, 0, 0, 0, 2, 0, 0], axis=-1)) or np.any(
                np.all(conv == [2, 0, 0, 1, 1, 0, 0], axis=-1)):
            # two boxes near a wall
            return True

        return False

    def check_dfs(self, state):
        """Checks whether the given state is a dead end."""
        state_flat = tuple(state.flatten())

        visited_value = self.visited.get(state_flat, None)

        if visited_value is not None:
            return visited_value

        if self.env.is_solved(state):
            self.visited[state_flat] = False
            return False

        self.visited[state_flat] = True

        if self.check_naive(state):
            return True

        for action in range(self.n_actions):
            next_state = self.env.next_state(state, action)

            if not self.check_dfs(next_state):
                self.visited[state_flat] = False
                return False

        return True

    @staticmethod
    def state_hash(state):
        x = sum([state[:, :, i] * i for i in range(7)])
        x = x.flatten()

        s_hash = 0

        for a in x:
            if a == 0:
                continue

            s_hash = s_hash * 6 + a - 1

        return s_hash

    def check_bfs(self, state):
        """Checks whether the given state is a dead end."""

        states = []
        queue = [state]
        res = True
        local_visited = set()

        while len(queue) > 0:
            state = queue.pop(0)
            state_flat = self.state_hash(state)
            # print('checking', state_flat)

            visited_value = self.visited.get(state_flat, None)

            if visited_value is not None:
                if visited_value == False:
                    res = False
                    break
                else:
                    continue

            if state_flat in local_visited:
                continue

            local_visited.add(state_flat)
            states.append(state)

            if self.env.is_solved(state):
                self.visited[state_flat] = False
                res = False
                break

            if self.check_naive(state):
                self.visited[state_flat] = True
                continue

            for action in range(self.n_actions):
                next_state = self.env.next_state(state, action)

                queue.append(next_state)

        if res == False:
            for state in states[::-1]:
                for action in range(self.n_actions):
                    next_state = self.env.next_state(state, action)

                    if self.visited.get(self.state_hash(next_state), None) == False:
                        self.visited[self.state_hash(state)] = False
                        break
        else:
            for state in states[::-1]:
                self.visited[self.state_hash(state)] = True

        return res
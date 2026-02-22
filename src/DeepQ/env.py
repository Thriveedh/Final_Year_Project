import numpy as np

class PartitionEnv:
    def __init__(self, items):
        self.items = items
        self.n, self.dim = items.shape
        self.reset()

    def reset(self):
        self.index = 0
        self.sumA = np.zeros(self.dim)
        self.sumB = np.zeros(self.dim)
        return self._get_state()

    def _get_state(self):
        # state = [index, diff_dim1, diff_dim2, ...]
        diff = self.sumA - self.sumB
        return np.concatenate(([self.index], diff))

    def step(self, action):
        item = self.items[self.index]

        if action == 0:
            self.sumA += item
        else:
            self.sumB += item

        self.index += 1
        done = self.index == self.n

        reward = -np.sum(np.abs(self.sumA - self.sumB))
        return self._get_state(), reward, done

"""Page 28."""

import numpy as np


class KArmedBanditEnv:
    def __init__(self, k=10):
        self.k = k
        self.action_space = np.arange(k)
        self.reset()

    def reset(self):
        self.true_values = np.fromiter((np.random.normal() for _ in range(10)), np.float64)
        self.optimal_action = np.argmax(self.true_values)

    def step(self, action):
        mu = self.true_values[action]
        sigma = 1
        return np.random.normal(mu, sigma)

    def print_true_values(self):
        for val in self.true_values:
            print(round(val, 2), end=' ')

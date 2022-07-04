"""Page 28."""

import random


class KArmedBanditEnv:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.action_space = [n for n in range(n_arms)]
        self.reset()

    def reset(self):
        self.true_values = [random.gauss(0, 1) for _ in range(self.n_arms)]
        self.optimal_action = self.true_values.index(max(self.true_values))

    def step(self, action):
        mu = self.true_values[action]
        sigma = 1
        return random.gauss(mu, sigma)

    def print_true_values(self):
        for val in self.true_values:
            print(round(val, 2), end=' ')

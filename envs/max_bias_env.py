"""
Page 135.

T B A T
"""
# TODO: Will finish later. How do I make different action spaces for different states?

import random


class MaxBiasEnv:
    def __init__(self):
        self.action_space = ('left', 'right')
        self.agent = None
        self.reset()

    def reset(self):
        self.agent.x = 2
        self.B_left_action

    def step(self, action):
        mu = self.true_values[action]
        sigma = 1
        return random.gauss(mu, sigma)

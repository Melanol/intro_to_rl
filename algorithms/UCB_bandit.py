""" Page 35. """

import math

import numpy as np


class UCBBandit:
    def __init__(self, env, c):
        self.env = env
        self.c = c
        self.Q = [0] * len(env.action_space)
        self.uses = [0] * len(env.action_space)

    def act(self, step):
        max_preference = -math.inf
        max_action = None
        for a in self.env.action_space:
            try:
                action_preference = self.Q[a] + self.c * (math.log(step) / self.uses[a])**0.5
            except ZeroDivisionError:  # If action was never selected, select it
                self.uses[a] += 1
                return a
            if action_preference > max_preference:
                max_preference = action_preference
                max_action = a
        self.uses[max_action] += 1
        return max_action

    def learn(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) / self.uses[action]

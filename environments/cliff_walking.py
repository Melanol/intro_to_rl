"""
Page 132. The origin is in the top left corner.

0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
S C C C C C C C C C C G

0 = empty space; S = start; C = cliff; G = goal.
"""

import random
from pprint import pprint


class CliffWalking:
    def __init__(self):
        self.action_space = ['up', 'down', 'left', 'right']
        self.cliff = [[x, y] for y in range(4) for x in range(12)[1:-1]]
        self.all_states = [[x, y] for y in range(4) for x in range(12)]
        # print(self.all_states)

    def start(self):
        self.state = [0, 3]
        return self.state

    def step(self, action):
        reward = -1
        termination = False
        if action == 'up':
            self.state[1] -= 1
        elif action == 'down':
            self.state[1] += 1
        elif action == 'left':
            self.state[0] -= 1
        elif action == 'right':
            self.state[0] += 1

        if self.state == [11, 3]:
            self.state = [0, 3]
            termination = True

        elif self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > 11:
            self.state[0] = 11
        elif self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > 3:
            self.state[1] = 3

        elif 0 < self.state[0] < 11 and self.state[1] == 3:
            self.state = [0, 3]
            reward = -100
            termination = True

        return reward, self.state, termination

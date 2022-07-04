"""Page 147."""

import random


class GridworldNoTP:
    def __init__(self):
        self.width = 3
        self.height = 3
        self.action_space = ['up', 'down', 'left', 'right']
        self.goal = [1, 0]
        self.all_states = [[x, y] for y in range(self.height) for x in range(self.width)]

    def reset(self):
        while True:
            self.state = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
            if self.state != self.goal:
                break

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

        if self.state == self.goal:
            self.reset()
            reward = 100
            termination = True

        elif self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > self.width-1:
            self.state[0] = self.width-1
        elif self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > self.height-1:
            self.state[1] = self.height-1

        return reward, self.state, termination

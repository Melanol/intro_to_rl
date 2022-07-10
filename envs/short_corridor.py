"""Page 323."""

import random

import numpy as np


class ShortCorridor:
    def __init__(self, width=3, reversed_states=np.array([1])):
        self.width = width  # Terminal state excluded
        self.reversed_states = reversed_states
        self.goal = self.width
        self.all_states = np.arange(self.width)
        self.action_space = np.array([[0, 1], [1, 0]])  # Left, right
        self.reset()

    def reset(self):
        self.state = 0

    def step(self, action):
        reward = -1
        done = False

        if self.state in self.reversed_states:
            if np.array_equal(action, [0, 1]):
                action = np.array([1, 0])
            else:
                action = np.array([0, 1])

        if self.state == 0 and np.array_equal(action, [0, 1]):
            return reward, self.state, done
        else:
            if np.array_equal(action, [0, 1]):
                self.state -= 1
            else:
                self.state += 1
            if self.state == self.goal:
                reward = 100
                done = True
                self.reset()

        return self.state, reward, done

    def print_game_state(self):
        if self.state == 0:
            print('SX', end=' ')
        else:
            print('S', end=' ')
        for state in range(1, self.width):
            if state == self.state:
                if state in self.reversed_states:
                    print('RX', end=' ')
                else:
                    print('X', end=' ')
            elif state in self.reversed_states:
                print('R', end=' ')
            else:
                print('□', end=' ')
        print('G')


class Human:
    def __init__(self, env):
        self.env = env

    def act(self):
        while True:
            action = input('Your move: ')
            if action not in ('left', 'right'):
                print('Action must be "left" or "right".')
            else:
                if action == 'left':
                    return np.array([0, 1])
                else:
                    return np.array([1, 0])


def exe():
    env = ShortCorridor()
    agent = Human(env)
    env.agent = agent
    done = False
    while not done:
        env.print_game_state()
        action = agent.act()
        state, reward, done = env.step(action)
    print('\nTermination')


if __name__ == '__main__':
    exe()

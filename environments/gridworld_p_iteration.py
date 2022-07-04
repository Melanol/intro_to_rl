"""
Policy iteration (chapter 4.3, page 80). Here, we call states with [y][x].

X  □  □  □
□  □  □  □
□  □  □  □
□  □  □  X
"""

import random
from pprint import pprint as print


WIDTH = 4
HEIGHT = 4
TERMINAL_STATE = [(0, 0), (3, 3)]
THETA = 0.01
DISCOUNT = 0.9


class Gridworld:
    def __init__(self):
        self.player = None
        self.action_space = 'up', 'down', 'left', 'right'

    def step(self, x, y, action):
        # # Terminal state:
        # if (x, y) in TERMINAL_STATE:
        #     return 0, x, y

        reward = -1

        # Borders:
        if (x == 0 and action == 'left' or
                x == WIDTH-1 and action == 'right:' or
                y == 0 and action == 'up' or
                y == HEIGHT-1 and action == 'down'):
            pass  # Nothing changes, only a step wasted and -1 received

        # Normal moves:
        elif action == 'left':
            x -= 1
        elif action == 'right':
            x += 1
        elif action == 'down':
            y += 1
        else:  # 'top;
            y -= 1

        return reward, x, y

class Agent:
    def __init__(self, game):
        self.game = game
        self.V = [[0 for _ in range(4)] for _ in range(4)]
        self.pi = [[random.choice(game.action_space) for _ in range(4)] for _ in range(4)]


env = Gridworld()
env.player = Agent(env)
while True:
    delta = 0
    # Iterative policy evaluation:
    for y in range(4):
        for x in range(4):
            v = env.player.V[y][x]
            reward, next_x, next_y = env.step(x, y, env.player.pi[y][x])
            print(next_x)
            env.player.V[y][x] = reward + DISCOUNT * env.player.V[next_y][next_x]
            delta = max(delta, abs(v - env.player.V[y][x]))

    if delta < THETA:
        break

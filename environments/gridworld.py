"""
This is an attempt to reproduce the gridworld of the chapter 3.5 (example 3.5), with modifications.
In indexing, the origin is at the top left, rows come 1st (ex: (0, 0) is the origin). Telestates are teleportation
states. 1st coord is the start, 2nd is the end, the last number is the reward for using the teleport. Trying to
escape the board results in no movement but -1 reward. Normal moves result in 0 reward. The policy is random.
"""

import random
from pprint import pprint


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Telestate:
    def __init__(self, init_coord, tele_coord, reward):
        self.init_coord = Coord(*init_coord)
        self.tele_coord = Coord(*tele_coord)
        self.reward = reward


class Gridworld:
    def __init__(self, episodes=1000, width=5, height=5):
        self.player = None
        self.episodes = episodes
        self.width = width
        self.height = height
        self.telestates = (
            # Telestate((2, 2), (0, 0), 10),
            Telestate((1, 0), (1, 4), 10),
            Telestate((3, 0), (3, 2), 5)
        )

        self.action_space = ['up', 'down', 'left', 'right']

    def position(self):
        return Coord(random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def print_value_function(self):
        Q = self.player.value_function
        count = 1
        for val in Q.values():
            print(str(round(val, 2)).rjust(6), end=' ')
            if count % self.width == 0:
                print()
            count += 1

    def move_player(self, action):
        if action == 'up':
            self.player.position.y -= 1
        elif action == 'down':
            self.player.position.y += 1
        elif action == 'left':
            self.player.position.x -= 1
        elif action == 'right':
            self.player.position.x += 1

        # Telestates:
        for tstate in self.telestates:
            x, y = self.player.position.x, self.player.position.y
            if x == tstate.init_coord.x and y == tstate.init_coord.y:
                self.player.state_selection_n[(x, y)] += 1
                self.player.value_function[(x, y)] += 1 / self.player.state_selection_n[(x, y)] \
                                                      * (tstate.reward - self.player.value_function[(x, y)])
                self.player.position.x = tstate.tele_coord.x
                self.player.position.y = tstate.tele_coord.y
                return tstate.reward

        return 0

    def play(self):
        episode = 1
        while episode < self.episodes:
            episode += 1
            x, y = self.player.position.x, self.player.position.y
            action = self.player.act()
            reward = 0
            if x == 0:
                if y == 0:
                    if action in ('up', 'left'):
                        reward = -1
                    else:
                        reward = self.move_player(action)
                elif y == self.height - 1:
                    if action in ('down', 'left'):
                        reward = -1
                    else:
                        reward = self.move_player(action)
                else:
                    if action == 'left':
                        reward = -1
                    else:
                        reward = self.move_player(action)
            elif x == self.width - 1:
                if y == 0:
                    if action in ('up', 'right'):
                        reward = -1
                    else:
                        reward = self.move_player(action)
                elif y == self.height - 1:
                    if action in ('down', 'right'):
                        reward = -1
                    else:
                        reward = self.move_player(action)
                else:
                    if action == 'right':
                        reward = -1
                    else:
                        reward = self.move_player(action)
            else:
                if y == 0:
                    if action == 'up':
                        reward = -1
                    else:
                        reward = self.move_player(action)
                elif y == self.height - 1:
                    if action == 'down':
                        reward = -1
                    else:
                        reward = self.move_player(action)
                else:
                    reward = self.move_player(action)

            self.player.state_selection_n[(x, y)] += 1
            self.player.value_function[(x, y)] += 1 / self.player.state_selection_n[(x, y)] \
                * (reward - self.player.value_function[(x, y)])
        self.print_value_function()


class RandomAgent:
    def __init__(self, game):
        self.game = game
        self.position = game.position()
        self.state_selection_n = {}
        self.value_function = {}
        for y in range(game.width):
            for x in range(game.height):
                self.state_selection_n[(x, y)] = 0
                self.value_function[(x, y)] = 0

    def act(self):
        return random.choice(self.game.action_space)


env = Gridworld()
env.player = RandomAgent(env)
env.play()

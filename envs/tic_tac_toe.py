"""
Although, this env is presented in the introductory part of the book, it is not recommended to start with due to
how complex it is. Read the part and move on to the k-armed bandits.

0 = empty, 1 = 1st player, 2 = 2nd player. Example:
0 0 1
0 1 2
1 0 2

Since this env is very different from the others, I had to make an agent specifically for it, it is in this file.

The controls for the human player is in the form "y x", where both are numbers.

Some non-square grids crash the env, but I doubt anyone will use them anyway.
"""

# Improvement vectors in priority:
# > Finish ValueFunctionAgent
# > Let a human play against a trained agent
# > Self-play
# > A way for non-specifically-designed agents to play
# > Afterstates


import copy
import math
import random

import numpy as np


class TicTacToe:
    def __init__(self, width=3, height=3,  win_length=3):
        self.width = width
        self.height = height
        self.win_length = min(win_length, max(width, height))
        self.action_space = np.array([(x, y) for x in range(width) for y in range(height)])
        self.reset()

    def reset(self):
        self.state = np.array([np.zeros(self.width, dtype=int) for _ in range(self.height)])

    def add_players(self, players):
        self.players = players
        for i, player in enumerate(players):
            player.mark = i+1

    def return_empty_cells(self):
        empty_cells = []
        for cell in self.action_space:
            if self.state[tuple(cell)] == 0:
                empty_cells.append(np.array(cell))
        return np.array(empty_cells)

    def step(self, player_mark, action):
        reward = 0
        done = False
        info = {}
        self.state[tuple(action)] = player_mark
        game_state = self.check_game_state()
        if game_state[0] == 'win':
            winner = game_state[1]
            info['winner'] = winner
            reward = 1
            done = True
            print(f'Player {winner} wins')
        elif game_state[0] == 'draw':
            info['winner'] = 0
            done = True
            print('Draw')
        obs = self.state
        return obs, reward, done, info

    def check_diagonals(self, state):
        # Collecting diagonal elements:
        diagonals = []
        # 1st half and the center (if the grid is square):
        for d_length in range(1, self.height + 1):
            diagonal = []
            starting_y = self.height - d_length
            i = 0
            while True:
                try:
                    diagonal.append(state[starting_y + i][i])
                    i += 1
                except IndexError:
                    break
            starting_y -= 1
            diagonals.append(diagonal)

        # 2nd half:
        starting_y = 0
        for starting_x in range(1, self.width):
            diagonal = []
            i = 0
            while True:
                try:
                    diagonal.append(state[starting_y + i][starting_x + i])
                    i += 1
                except IndexError:
                    break
            diagonals.append(diagonal)

        # Checking collected diagonals:
        for diagonal in diagonals:
            if len(diagonal) >= self.win_length:
                for i, el in enumerate(diagonal):
                    if el != 0:
                        count_marks = 1
                        for el1 in diagonal[i + 1:]:
                            if el1 == el:
                                count_marks += 1
                                if count_marks == self.win_length:
                                    return True, el

        return False, 0

    def check_game_state(self):
        """Returns game state and the winner (or 0)"""
        # Horizontal:
        for starting_y in self.state:
            for ix, x in enumerate(starting_y):
                if x != 0:
                    count_marks = 1
                    for x1 in starting_y[ix+1:]:
                        if x1 == x:
                            count_marks += 1
                            if count_marks == self.win_length:
                                return 'win', x
                        else:
                            break

        # Vertical:
        for ix in range(self.width):
            for iy in range(self.height):
                el = self.state[iy][ix]
                if el != 0:
                    count_marks = 1
                    for iy1 in range(1+iy, self.height):
                        try:
                            if el == self.state[iy1][ix]:
                                count_marks += 1
                                if count_marks == self.win_length:
                                    return 'win', el
                        except IndexError:
                            pass

        # Top-left-to-bottom-right diagonal, sweeping from bottom left to top right:
        won, winner = self.check_diagonals(self.state)
        if won:
            return 'win', winner

        # Bottom-left-to-top-right diagonal, sweeping from bottom left to top right after horizontal reflection:
        won, winner = self.check_diagonals(np.flip(self.state, axis=1))
        if won:
            return 'win', winner

        # Draw:
        if 0 not in self.state.flatten():
            return 'draw', 0

        # Playing:
        return 'playing', 0


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.mark = None

    def act(self):
        return random.choice(self.env.return_empty_cells())

    def learn(self, **kwargs):
        pass


class Human:
    def __init__(self, env):
        self.env = env
        self.mark = None

    def act(self):
        import re

        print(self.env.state)
        str_ = f'\nPlayer {self.mark} move: '
        empty_cells = self.env.return_empty_cells()
        action = input(str_)
        while True:
            if not re.match(r'\d+ \d+$', action):
                print('The format is "y x"')
                action = input(str_)
            else:
                action = np.array([int(el) for el in action.split()])
                found = False
                for a in empty_cells:
                    if np.array_equal(a, action):
                        found = True
                        break
                if not found:
                    print('\nChoose an empty cell.')
                    action = input(str_)
                else:
                    return action

    def learn(self, **kwargs):
        pass

class ValueFunctionAgent:
    """This agent is a bit ugly, but close to how it is described in the book."""

    def __init__(self, env=None, epsilon=0.1, alpha=0.1):
        self.env = env
        self.epsilon = epsilon
        self.alpha = 0.1
        self.mark = None
        self.V = {}  # fill it as you encounter new states

    def act(self):
        # I guess, even if learning here is jumping over our moves, acting doesn't have to
        if random.random() <= self.epsilon:
            action = random.choice(self.env.action_space)
        else:
            # Generate next states
            empty_cells = self.env.return_empty_cells()
            next_states = []
            for a in empty_cells:
                state = copy.copy(self.env.state)  # TODO: Check if really need copy
                state[tuple(a)] = self.mark
                next_states.append((state, a))

            # Find best state
            max_state_value = -math.inf
            max_state = None
            for i, s in enumerate(next_states[0]):
                state_value = self.V.get(str(s), 0.5)
                if state_value > max_state_value:
                    max_state_value = state_value
                    max_state = s

            # Select action
            action = next_states[i][1]  # FIXME: List index out of range

        return action

    def learn(self, **kwargs):
        obs = kwargs['obs']
        next_obs = kwargs['next_obs']
        reward = kwargs['reward']
        if reward:  # Game ended
            self.V[str(obs)] = self.V.get(str(obs), 0.5) + self.alpha * (reward - self.V.get(str(obs), 0.5))
        else:
            self.V[str(obs)] = self.V.get(str(obs), 0.5) + self.alpha * (self.V.get(str(next_obs), 0.5)
                                                                         - self.V.get(str(obs), 0.5))


def play():
    env = TicTacToe()
    players = (RandomAgent(env), Human(env))
    env.add_players(players)
    done = False
    player1_turn = True
    while not done:
        if player1_turn:
            player_mark = 1
            obs_before_p1_action = env.state
            action = players[0].act()
            obs, reward, done, info = env.step(player_mark, action)
            obs_after_p1_action = obs
            player1_turn = False
        else:
            player_mark = 2
            obs_before_p2_action = env.state
            action = players[1].act()
            obs, reward, done, info = env.step(player_mark, action)
            obs_after_p2_action = obs

            players[1].learn(obs=obs_before_p1_action, next_obs=obs_after_p2_action, reward=reward)

            player1_turn = True


if __name__ == '__main__':
    play()

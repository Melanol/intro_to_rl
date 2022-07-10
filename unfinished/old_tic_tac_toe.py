# This is my 1st attempt to make tic-tac-toe.

import os
import random
import sys
import copy
import math
import pickle


def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


class TicTacToe:
    def __init__(self, intermediate_print=True):
        if not intermediate_print:
            disable_print()
        self.players = []
        self.action_space = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = {index: 0 for index in self.action_space}
        self.all_states = []
        for q in range(3):
            for w in range(3):
                for e in range(3):
                    for a in range(3):
                        for s in range(3):
                            for d in range(3):
                                for z in range(3):
                                    for x in range(3):
                                        for c in range(3):
                                            self.all_states.append(
                                                {'q': q, 'w': w, 'e': e,
                                                 'a': a, 's': s, 'd': d,
                                                 'z': z, 'x': x, 'c': c})

    def reset(self):
        self.state = {index: 0 for index in self.action_space}
        disable_print()

    def is_valid_action(self, action):
        if action in self.action_space and self.state[action] == 0:
            return True
        else:
            return False

    def return_empty_cells(self):
        empty_cells = []
        for cell in self.action_space:
            if self.state[cell] == 0:
                empty_cells.append(cell)
        return empty_cells

    def next_states(self, mark):
        next_states = []
        for empty_cell in self.return_empty_cells():
            next_state = copy.copy(self.state)  # TODO: If doesn't work: try deepcopy
            next_state[empty_cell] = mark
            next_states.append(next_state)
        return next_states

    def play(self):
        finished = False
        while not finished:
            for player in self.players:
                self.intermediate_print()
                action = player.act(self.return_empty_cells())
                empty_cells = self.return_empty_cells()
                if action not in empty_cells:
                    raise Exception(f"Invalid action in this situation: {action}. Valid actions: {empty_cells}")
                self.state[action] = player.mark
                if self.win_state()[0] != 'playing':
                    finished = True
                    if self.win_state()[0] == 'win':
                        print(f'Winner: Player {player.mark}')
                        enable_print()
                        return player.mark
                    else:
                        print('Draw')
                        enable_print()
                        return 0

    def intermediate_print(self):
        str_ = ''
        for i, value in enumerate(self.state.values()):
            if (i + 1) % 3 == 0:
                str_ += f'{str(value)}\n'
            else:
                str_ += f'{str(value)} '
        str_ = str_[:-1]
        print(str_ + '\n')

    def win_state(self):
        """Returns game state and the winner (or 0)"""
        # Horizontal win:
        if 0 != self.state['q'] == self.state['w'] == self.state['e']:
            self.intermediate_print()
            return 'win', self.state['q']
        elif 0 != self.state['a'] == self.state['s'] == self.state['d']:
            self.intermediate_print()
            return 'win', self.state['a']
        elif 0 != self.state['z'] == self.state['x'] == self.state['c']:
            self.intermediate_print()
            return 'win', self.state['z']

        # Vertical win:
        elif 0 != self.state['q'] == self.state['a'] == self.state['z']:
            self.intermediate_print()
            return 'win', self.state['q']
        elif 0 != self.state['w'] == self.state['s'] == self.state['x']:
            self.intermediate_print()
            return 'win', self.state['w']
        elif 0 != self.state['e'] == self.state['d'] == self.state['c']:
            self.intermediate_print()
            return 'win', self.state['e']

        # Diagonal win:
        elif 0 != self.state['z'] == self.state['s'] == self.state['e']:
            self.intermediate_print()
            return 'win', self.state['z']
        elif 0 != self.state['q'] == self.state['s'] == self.state['c']:
            self.intermediate_print()
            return 'win', self.state['q']

        # Draw:
        elif 0 not in self.state.values():
            self.intermediate_print()
            return 'draw', 0

        else:
            return 'playing', 0

class RandomAgent:
    def __init__(self, game, mark):
        self.game = game
        self.mark = mark
        self.game = game

    @staticmethod
    def act(empty_cells):
        return random.choice(empty_cells)

class RLAgent:
    def __init__(self, game, mark, epsilon=0.1, learning_rate=0.1, save=True, load=False):
        self.game = game
        self.mark = mark
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.save = True

        if load:
            self.value_function = pickle.load(open("RLAgent.p", "rb"))
        else:
            self.value_function = {}  # Win states get 1, loss states get 0, other states get 0.5
            for state in game.all_states:
                # Horizontal:
                if 0 != state['q'] == state['w'] == state['e']:
                    if state['q'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                if 0 != state['a'] == state['s'] == state['d']:
                    if state['a'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                if 0 != state['z'] == state['x'] == state['c']:
                    if state['z'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue

                # Vertical:
                if 0 != state['z'] == state['a'] == state['q']:
                    if state['z'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                if 0 != state['x'] == state['s'] == state['w']:
                    if state['x'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                if 0 != state['c'] == state['d'] == state['e']:
                    if state['c'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue

                # Diagonal:
                if 0 != state['z'] == state['s'] == state['e']:
                    if state['z'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                if 0 != mark == state['q'] == state['s'] == state['c']:
                    if state['q'] == self.mark:
                        self.value_function[str(state)] = 1
                        continue
                    else:
                        self.value_function[str(state)] = 0
                        continue
                self.value_function[str(state)] = 0.5

    def act(self, empty_cells):
        if random.random() < self.epsilon:
            # Exploratory moves do not result in any learning
            return random.choice(empty_cells)
        else:  # Choose max next state
            max_next_state = None
            max = -math.inf
            for next_state in self.game.next_states(self.mark):
                val = self.value_function[str(next_state)]
                if val > max:
                    max_next_state = next_state
                    max = val

            # Update (learning):
            state = self.game.state
            self.value_function[str(state)] = self.value_function[str(state)] \
                + self.learning_rate * (self.value_function[str(max_next_state)]
                                        - self.value_function[str(state)])

            # Find difference between current and next state we want and return index:
            set1 = set(state.items())
            set2 = set(max_next_state.items())
            diff = set1 ^ set2
            return list(diff)[0][0]


class Human:
    def __init__(self, game, mark):
        self.game = game
        self.mark = mark

    def act(self, empty_cells):
        str_ = f'\nPlayer {self.mark} move: '
        action = input(str_)
        while True:
            if action not in ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']:
                print('\nEnter 1 on these: q, w, e, a, s, d, z, x, c')
                action = input(str_)
            elif action not in empty_cells:
                print('\nChoose an empty cell.')
                action = input(str_)
            else:
                print()
                return action


SAVE_EVERY = 500  # N of games
def play(env, agent1, agent2, n_games=1, save=False):
    env.players = [agent1, agent2]
    player1_wins = 0
    player2_wins = 0
    draws = 0
    for game_n in range(n_games):
        winner = env.play()
        if winner == 1:
            player1_wins += 1
        elif winner == 2:
            player2_wins += 1
        else:
            draws += 1
        env.reset()
        if save:
            if game_n % SAVE_EVERY == 0:
                if type(agent1) is RLAgent:
                    pickle.dump(agent1.value_function, open("RLAgent.p", "wb"))
    # Print statistics:
    if Human not in [agent1, agent2]:
        enable_print()
        print(player1_wins, player2_wins, draws)
        print(f'{round(player1_wins/n_games*100, 2)}% {round(player2_wins/n_games*100, 2)}% '
              f'{round(draws/n_games*100, 2)}%')

# env = TicTacToe(intermediate_print=False)
# agent1 = RLAgent(env, 1, save=True, load=True)
# agent2 = RandomAgent(env, 2)
# play(env, agent1, agent2, n_games=10000, save=True)

env = TicTacToe()
agent1 = RLAgent(env, 1, save=False, load=True)
agent2 = Human(env, 2)
play(env, agent1, agent2, n_games=1)

# TODO: Proper self-play
# TODO: non 3x3 grids
# TODO: Graphs

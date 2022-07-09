"""
Although, this env is presented in the introductory part of the book, it is not recommended to start with due to
how complex it is. Read the part and move on to the k-armed bandits.

0 = empty, 1 = 1st player, 2 = 2nd player. Example:
0 0 1
0 1 2
1 0 2

The controls for the human player is in the form "y x", where both are numbers.

Some non-square grids crash the env, but I doubt anyone will use them anyway.
"""


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
        self.state[tuple(action)] = player_mark
        game_state = self.check_game_state()
        if game_state[0] == 'win':
            reward = 1
            done = True
            print(f'Player {game_state[1]} wins')
        elif game_state[0] == 'draw':
            print('Draw')
            done = True
        obs = self.state
        return obs, reward, done

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

class ValueFunctionAgent:
    #  TODO: 0 value for loss and draw, 1 for a won state, 0.5 for everyting else.
    # TODO: we look at the all next possible moves and check up their values in the table. Epsilon action selection.
    # TODO: exploratory moves do not result in any learning. after each non-exploratory move, we update like this:
    # (TD) : V(St) = V(St) + a(V(St+1) - V(St), where V(St) is the prev state (not a cell), V(St+1) is the val of the
    # next state. Recommended to gradually reduce alpha

    """ Finally, the tic-tac-toe player was able to look ahead and know the states that would
result from each of its possible moves. To do this, it had to have a model of the game
that allowed it to foresee how its environment would change in response to moves that it
might never make """
    def __init__(self):
        pass


def play():
    env = TicTacToe()
    players = (RandomAgent(env), Human(env))
    env.add_players(players)
    done = False
    player_1_turn = True
    while not done:
        if player_1_turn:
            player_mark = 1
            action = players[0].act()
            player_1_turn = False
        else:
            player_mark = 2
            action = players[1].act()
            player_1_turn = True
        obs, reward, done = env.step(player_mark, action)


if __name__ == '__main__':
    play()

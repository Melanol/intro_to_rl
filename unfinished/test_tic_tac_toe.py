import numpy as np


from envs.tic_tac_toe import TicTacToe


def test_horizontal_win():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[1] = [1, 1, 1, 0, 0]
    print(env.state)
    assert env.check_game_state() == ('win', 1)

def test_horizontal_win1():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[4] = [0, 0, 1, 1, 1]
    assert env.check_game_state() == ('win', 1)

def test_horizontal_playing():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[0] = [1, 1, 0, 0, 0]
    assert env.check_game_state() == ('playing', 0)

def test_vertical_win():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[:, 1] = [1, 1, 1, 0, 0]
    assert env.check_game_state() == ('win', 1)

def test_vertical_win1():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[:, 4] = [0, 0, 1, 1, 1]
    assert env.check_game_state() == ('win', 1)

def test_vertical_playing():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[:, 4] = [1, 1, 0, 0, 0]
    assert env.check_game_state() == ('playing', 0)

def test_vertical_playing1():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state[:, 0] = [0, 0, 0, 1, 1]
    assert env.check_game_state() == ('playing', 0)

def test_diagonal_win():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    assert env.check_game_state() == ('win', 1)

def test_diagonal_win1():
    env = TicTacToe(width=5, height=5, win_length=3)
    env.state = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
    assert env.check_game_state() == ('win', 1)

def test_diagonal_win2():
    env = TicTacToe(width=5, height=2, win_length=2)
    env.state = np.array([[0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0]])
    assert env.check_game_state() == ('win', 1)

def test_diagonal_win3():
    env = TicTacToe(width=2, height=5, win_length=2)
    env.state = np.array([[0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 1],
                          [1, 0]])
    assert env.check_game_state() == ('win', 1)

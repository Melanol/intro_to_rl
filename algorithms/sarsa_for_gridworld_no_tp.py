import math
import random
from matplotlib import pyplot as plt

from environments.gridworld_no_tp import GridworldNoTP


EPISODES = 100
EPSILON = 0.1
STEP_SIZE = 0.1
DISCOUNT = 0.9

class Agent:
    def __init__(self, game):
        self.game = game
        self.Q = {}
        for state in game.all_states:
            for action in game.action_space:
                self.Q[(str(state), action)] = 0

    def get_Q(self, key):
        if key[0] == str(self.game.goal):
            return 0
        else:
            return self.Q[key]

    def act(self, state):
        if random.random() <= EPSILON:
            return random.choice(self.game.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.game.action_space:
                action_value = self.get_Q((str(state), action))
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def print_Q(self):
        arrow_doct = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        count = 1
        for state in self.game.all_states:
            if state != self.game.goal:
                max_action_value = -math.inf
                max_action = None
                for action in self.game.action_space:
                    action_value = self.get_Q((str(state), action))
                    if action_value > max_action_value:
                        max_action_value = action_value
                        max_action = action
                print(arrow_doct[max_action], end='   ')
            else:
                print('G', end='   ')
            if count % env.width == 0:
                print()
            count += 1
        print()


env = GridworldNoTP()
player = Agent(env)
player.print_Q()
env.player = player
for _ in range(EPISODES):
    env.reset()
    S = env.state
    action = player.act(S)
    while True:
        reward, next_state, termination = env.step(action)
        next_action = player.act(next_state)
        player.Q[(str(S), action)] += STEP_SIZE * (
                reward + DISCOUNT * player.get_Q((str(next_state), next_action)) - player.get_Q((str(S), action))
        )
        S, action = next_state, next_action
        if termination:
            break

player.print_Q()

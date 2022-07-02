# Page 131

import math
import random

from environments.cliff_walking import CliffWalking


EPISODES = 100
EPSILON = 0.5
STEP_SIZE = 0.1
DISCOUNT = 1

class Agent:
    def __init__(self, game):
        self.game = game
        self.Q = {}
        for state in game.all_states:
            for action in game.action_space:
                self.Q[(str(state), action)] = 0
        # print(self.Q)

    def act(self, state):
        if random.random() <= EPSILON:
            return random.choice(self.game.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.game.action_space:
                action_value = self.Q[(str(state), action)]
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def max_next_Q(self, next_state):
        max_action_value = -math.inf
        max_action = None
        for action in self.game.action_space:
            action_value = self.Q[(str(next_state), action)]
            if action_value > max_action_value:
                max_action_value = action_value
                max_action = action
        return max_action_value

    def print_Q(self):
        for key, value in self.Q.items():
            print(key, round(value, 2))


env = CliffWalking()
player = Agent(env)
env.player = player
for _ in range(EPISODES):
    S = env.start()
    while True:
        action = player.act(S)
        reward, next_state, termination = env.step(action)
        next_action = player.act(next_state)
        player.Q[(str(S), action)] += STEP_SIZE * (
                reward + DISCOUNT * player.max_next_Q(next_state) - player.Q[(str(S), action)]
        )
        S = next_state
        if termination:
            break

player.print_Q()

# Page 130

import math
import random
from matplotlib import pyplot as plt

from environments.cliff_walking import CliffWalking


EPISODES = 50
EPSILON = 0.5
STEP_SIZE = 0.1
DISCOUNT = 0.9

class Agent:
    def __init__(self, game):
        self.game = game
        self.Q = {}
        for state in game.all_states:
            for action in game.action_space:
                self.Q[(str(state), action)] = 0

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

    def print_Q(self):
        for key, value in list(self.Q.items())[2:-2]:
            print(key, round(value, 2))


env = CliffWalking()
player = Agent(env)
env.player = player
all_rewards = []
for _ in range(EPISODES):
    S = env.start()
    action = player.act(S)
    episode_rewards = []
    while True:
        reward, next_state, termination = env.step(action)
        episode_rewards.append(reward)
        next_action = player.act(next_state)
        player.Q[(str(S), action)] += STEP_SIZE * (
                reward + DISCOUNT * player.Q[(str(next_state), next_action)] - player.Q[(str(S), action)]
        )
        S, action = next_state, next_action
        if termination:
            break
    all_rewards.append(sum(episode_rewards))

player.print_Q()
plt.plot(all_rewards)
plt.show()

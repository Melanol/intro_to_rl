"""Page 130."""
# TODO: Make sure the algo works

import math
import random
from matplotlib import pyplot as plt

from environments.windy_gridworld import WindyGridworld


EPISODES = 200
EPSILON = 0.1
ALPHA = 0.5
DISCOUNT = 1
INITIAL_Q = 0


class Sarsa:
    def __init__(self, env):
        self.env = env
        # TODO: Generate Q as you encounter new states

        # All actions for the terminal state have the value of 0
        self._Q = {}
        for action in env.action_space:
            self._Q[((env.goal.x, env.goal.y), action)] = 0

    @property
    def Q(self):
        try:
            return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value


    def act(self, obs):
        if random.random() <= EPSILON:
            return random.choice(self.env.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.env.action_space:
                action_value = self.Q[(obs, action)]
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def learn(self, obs, action, reward):
        self.Q[(obs, action)] += ALPHA * (reward + DISCOUNT * self.Q[(obs, next_action)] - self.Q[(obs, action)])

    def print_Q(self):
        for key, value in list(self.Q.items())[2:-2]:
            print(key, round(value, 2))


env = WindyGridworld()
agent = Sarsa(env)
env.agent = agent
for _ in range(EPISODES):
    obs = env.reset()
    action = agent.act(obs)
    done = False
    while not done:
        obs, reward, done = env.step(action)
        next_action = agent.act(obs)
        agent.learn(obs, action, reward)
        obs, action = obs, next_action


# Plotting:

# agent.print_Q()
# ys_to_plot = []
# for plot_actions in plot_actions_vs_episodes:
#     ys_to_plot.append(sum(1 for el in plot_actions if el == 'right') / len(plot_actions))
# plt.plot(ys_to_plot)
#
# count = 1
# means_to_plot = []
# ys = []
# for y in ys_to_plot:
#     ys.append(y)
#     if count % 10 == 0:
#         means_to_plot.append(sum(ys) / len(ys))
#         ys = []
#     count += 1
# plt.plot([10 * x for x in range(EPISODES // 10)], means_to_plot)
# plt.title('Mean of the rate of right actions chosen')
# plt.show()

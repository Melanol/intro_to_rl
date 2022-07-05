"""Page 130."""
# TODO: Make sure the algo works

import math
import random
from matplotlib import pyplot as plt

from environments.windy_gridworld import WindyGridworld


EPISODES = 100
EPSILON = 0.1
ALPHA = 0.5
DISCOUNT = 1
DEFAULT_Q = 0


class Sarsa:
    def __init__(self, env):
        self.env = env
        self.Q = {}

        # # All actions for the terminal state have the value of 0
        # self._Q = {}
        # for action in env.action_space:
        #     self._Q[((env.goal.x, env.goal.y), action)] = 0

    # @property
    # def Q(self):
    #     try:
    #         return self._Q
    #
    # @Q.setter
    # def Q(self, value):
    #     self._Q = value


    def act(self, obs):
        if random.random() <= EPSILON:
            return random.choice(self.env.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.env.action_space:
                action_value = self.Q.get((obs, action), DEFAULT_Q)
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def learn(self, obs, action, reward, next_obs, next_action):
        self.Q[(obs, action)] = self.Q.get((obs, action), DEFAULT_Q) + ALPHA * \
            (reward + DISCOUNT * self.Q.get((next_obs, next_action), DEFAULT_Q) - self.Q.get((obs, action), DEFAULT_Q))

    def print_Q(self):
        for key, value in list(self.Q.items())[2:-2]:
            print(key, round(value, 2))


env = WindyGridworld()
agent = Sarsa(env)
env.agent = agent
steps_to_plot = []
for _ in range(EPISODES):
    obs = env.reset()
    action = agent.act(obs)
    done = False
    step = 0
    while not done:
        next_obs, reward, done = env.step(action)
        next_action = agent.act(next_obs)
        agent.learn(obs, action, reward, next_obs, next_action)
        obs, action = next_obs, next_action
        step += 1
    steps_to_plot.append(step)

plt.plot(steps_to_plot)
plt.show()

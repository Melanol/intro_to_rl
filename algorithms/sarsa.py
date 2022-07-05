"""Page 130."""

import math
import random
from matplotlib import pyplot as plt


class Sarsa:
    def __init__(self, env=None, epsilon=0.1, alpha=0.1, discount=0.9, default_Q=0):
        self.env = env
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.default_Q = default_Q


    def act(self, obs):
        if random.random() <= self.epsilon:
            return random.choice(self.env.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.env.action_space:
                action_value = self.Q.get((obs, action), self.default_Q)
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def learn(self, obs, action, reward, next_obs, next_action):
        self.Q[(obs, action)] = self.Q.get((obs, action), self.default_Q) + self.alpha * \
            (reward + self.discount * self.Q.get((next_obs, next_action), self.default_Q)
             - self.Q.get((obs, action), self.default_Q))


def exe():
    from environments.windy_gridworld import WindyGridworld

    ENV = WindyGridworld()
    EPISODES = 200
    EPSILON = 0.1
    ALPHA = 0.5
    DISCOUNT = 1
    DEFAULT_Q = 0

    agent = Sarsa(ENV, EPSILON, ALPHA, DISCOUNT, DEFAULT_Q)
    ENV.agent = agent

    steps_to_plot = []
    for _ in range(EPISODES):
        obs = ENV.reset()
        action = agent.act(obs)
        done = False
        step = 0
        while not done:
            next_obs, reward, done = ENV.step(action)
            next_action = agent.act(next_obs)
            agent.learn(obs, action, reward, next_obs, next_action)
            obs, action = next_obs, next_action
            step += 1
        steps_to_plot.append(step)

    plt.plot(steps_to_plot)
    plt.show()


if __name__ == '__main__':
    exe()

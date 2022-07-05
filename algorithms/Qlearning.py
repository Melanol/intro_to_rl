"""Page 131"""

import math
import random
from matplotlib import pyplot as plt


class QLearning:
    def __init__(self, env, epsilon, alpha, discount, default_Q):
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

    def learn(self, obs, action, reward, next_obs):
        # FIXME: Somehow, it diverges on any env
        max_action_value = -math.inf
        for a in self.env.action_space:
            action_value = self.Q.get((next_obs, a), self.default_Q)
            if action_value > max_action_value:
                max_action_value = action_value
        self.Q[(obs, action)] = self.Q.get((obs, action), self.default_Q) + self.alpha * \
            (reward + self.discount * max_action_value - self.Q.get((obs, action), self.default_Q))


def exe():
    from environments.windy_gridworld import WindyGridworld

    ENV = WindyGridworld()
    EPISODES = 100
    EPSILON = 0.1
    ALPHA = 0.1
    DISCOUNT = 1
    DEFAULT_Q = 0

    agent = QLearning(ENV, EPSILON, ALPHA, DISCOUNT, DEFAULT_Q)
    ENV.agent = agent

    steps_to_plot = []
    for _ in range(EPISODES):
        obs = ENV.reset()
        done = False
        step = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, done = ENV.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            step += 1
        steps_to_plot.append(step)

    plt.plot(steps_to_plot)
    plt.show()


if __name__ == '__main__':
    exe()

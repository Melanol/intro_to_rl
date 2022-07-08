""" Page 34. It is different from the simple bandit not only in initial values,
but also in having a constant learning rate. """

import random
import math

import numpy as np


class OptimisticBandit:
    def __init__(self, env=None, initial=5, epsilon=0.1, alpha=0.1):
        self.env = env
        self.epsilon = epsilon
        self.initial = initial
        self.alpha = alpha
        if env:
            self.Q = np.full(len(env.action_space), initial, dtype='float64')

    def assign_env(self, env):
        self.env = env
        self.Q = np.full(len(env.action_space), self.initial, dtype='float64')

    def act(self, step):
        if random.random() <= self.epsilon:
            action = random.choice(self.env.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for a in self.env.action_space:
                action_value = self.Q[a]
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = a
            action = max_action
        return action

    def learn(self, **kwargs):
        action = kwargs['action']
        reward = kwargs['reward']
        self.Q[action] += self.alpha * (reward - self.Q[action])


def exe():
    from matplotlib import pyplot as plt

    from envs.k_armed_bandit_env import KArmedBanditEnv

    STEPS = 1000
    EPISODES = 200
    EPSILON = 0
    INITIAL = 5
    ALPHA = 0.1
    ENV = KArmedBanditEnv()

    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES+1):
        ENV.reset()
        agent = OptimisticBandit(ENV, INITIAL, EPSILON, ALPHA)
        rewards = []
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = ENV.step(action)
            agent.learn(action=action, reward=reward)
            rewards.append(reward)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode

    plt.xticks([1] + list(range(200, STEPS+1, 200)))
    plt.plot(range(1, STEPS+1), avg_rewards)
    plt.title('Average rewards')
    plt.show()

if __name__ == '__main__':
    exe()

"""Page 28. Learning is already incremental."""

import random
import math

import numpy as np


class EpsilonBandit:
    def __init__(self, env=None, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.Q = np.zeros(len(env.action_space))
        self.uses = np.zeros(len(env.action_space))

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
        self.uses[action] += 1
        return action

    def learn(self, **kwargs):
        action = kwargs['action']
        reward = kwargs['reward']
        self.Q[action] += (reward - self.Q[action]) / self.uses[action]


def exe():
    from matplotlib import pyplot as plt

    from envs.k_armed_bandit_env import KArmedBanditEnv

    STEPS = 1000
    EPISODES = 200
    EPSILON = 0.1
    ENV = KArmedBanditEnv()

    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES+1):
        ENV.reset()
        agent = EpsilonBandit(ENV, EPSILON)
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

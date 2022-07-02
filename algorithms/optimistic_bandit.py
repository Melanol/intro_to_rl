""" Page 34. """

import random
from multiprocessing import Pool
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from environments.k_armed_bandit_env import KArmedBanditEnv


class Agent:
    def __init__(self, env, initial, epsilon):
        self.env = env
        self.epsilon = epsilon
        self.Q = [initial] * len(env.action_space)

    def act(self):
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


def exe(initial, epsilon):
    avg_rewards = np.zeros(STEPS)
    avg_perc_opt_actions = np.zeros(STEPS)
    for episode in range(1, EPISODES + 1):
        env = KArmedBanditEnv()
        agent = Agent(env, initial, epsilon)
        rewards = []
        optimal_actions_bool = []
        for _ in range(1, STEPS+1):
            action = agent.act()
            reward = env.step(action)
            agent.Q[action] += ALPHA * (reward - agent.Q[action])
            rewards.append(reward)
            optimal_actions_bool.append(action == env.optimal_action)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode
            avg_perc_opt_actions[i] += (optimal_actions_bool[i] - avg_perc_opt_actions[i]) / episode
    return avg_rewards, avg_perc_opt_actions

STEPS = 1000
EPISODES = 2000
ALPHA = 0.1
INITIALS = (0, 5)
EPSILONS = (0.1, 0)

with Pool() as pool:
    results = pool.starmap(exe, zip(INITIALS, EPSILONS))
    avg_rewards = []
    avg_perc_opt_actions = []
    for r, p in results:
        avg_rewards.append(r)
        avg_perc_opt_actions.append(p)
    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    for i, epsilon in enumerate(EPSILONS):
        plt.xticks([1] + list(range(200, STEPS+1, 200)))
        plt.plot(range(1, STEPS+1), avg_rewards[i], label=f'init: {INITIALS[i]} e: {epsilon}')
    plt.legend()
    plt.title('Average rewards')

    plt.subplot(2, 1, 2)
    for i, epsilon in enumerate(EPSILONS):
        plt.xticks([1] + list(range(200, STEPS+1, 200)))
        plt.plot(range(1, STEPS+1), avg_perc_opt_actions[i], label=f'init: {INITIALS[i]} e: {epsilon}')
    plt.legend()
    plt.title('% optimal actions')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.savefig('optimistic_bandit.png')
    plt.show()
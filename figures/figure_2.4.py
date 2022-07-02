""" Page 36. """

from multiprocessing import Pool
import math

import numpy as np
from matplotlib import pyplot as plt

from environments.k_armed_bandit_env import KArmedBanditEnv
from algorithms.simple_bandit import SimpleBandit
from algorithms.UCB_bandit import UCBBandit


def exe(initial, epsilon):
    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES + 1):
        ENV.reset()
        agent = UCBBandit(ENV, initial, epsilon)
        rewards = []
        optimal_actions_bool = []
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = ENV.step(action)
            agent.learn(action, reward)
            rewards.append(reward)
            optimal_actions_bool.append(action == ENV.optimal_action)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode
    return avg_rewards

STEPS = 1000
EPISODES = 2000
ENV = KArmedBanditEnv()
PLAYERS = [SimpleBandit(ENV, epsilon=0.1), UCBBandit(ENV, c=2)]

with Pool() as pool:
    results = pool.starmap(exe, PLAYERS)
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

    plt.savefig('../figures/figure_2.4.png')
    plt.show()

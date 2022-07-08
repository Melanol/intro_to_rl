""" Page 29. """

from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from envs.k_armed_bandit_env import KArmedBanditEnv
from algos.epsilon_bandit import EpsilonBandit


STEPS = 1000
EPISODES = 2000
EPSILONS = (0.1, 0.01, 0)

def exe(epsilon):
    avg_rewards = np.zeros(STEPS)
    avg_perc_opt_actions = np.zeros(STEPS)
    for episode in range(1, EPISODES + 1):
        env = KArmedBanditEnv()
        agent = EpsilonBandit(env, epsilon)
        rewards = []
        optimal_actions_bool = []
        for _ in range(1, STEPS+1):
            action = agent.act()
            reward = env.step(action)
            agent.learn(action, reward)
            rewards.append(reward)
            optimal_actions_bool.append(action == env.optimal_action)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode
            avg_perc_opt_actions[i] += (optimal_actions_bool[i] - avg_perc_opt_actions[i]) / episode
    return avg_rewards, avg_perc_opt_actions

with Pool() as pool:
    results = pool.map(exe, EPSILONS)
    avg_rewards = []
    avg_perc_opt_actions = []
    for r, p in results:
        avg_rewards.append(r)
        avg_perc_opt_actions.append(p)
    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    for i, epsilon in enumerate(EPSILONS):
        plt.xticks([1] + list(range(200, STEPS+1, 200)))
        plt.plot(range(1, STEPS+1), avg_rewards[i], label=f'e: {epsilon}')
    plt.legend()
    plt.title('Average rewards')

    plt.subplot(2, 1, 2)
    for i, epsilon in enumerate(EPSILONS):
        plt.xticks([1] + list(range(200, STEPS+1, 200)))
        plt.plot(range(1, STEPS+1), avg_perc_opt_actions[i], label=f'e: {epsilon}')
    plt.legend()
    plt.title('% optimal actions')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.savefig('figure_2.2_epsilon_bandits.png')
    plt.show()

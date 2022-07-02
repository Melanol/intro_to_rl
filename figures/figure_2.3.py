""" Page 34. For this figure, both algos have to be OptimisticBandit, because the simple bandit in this repo
does not use alpha. """

from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from environments.k_armed_bandit_env import KArmedBanditEnv
from algorithms.optimistic_bandit import OptimisticBandit


def exe(epsilon, initial):
    avg_perc_opt_actions = np.zeros(STEPS)
    for episode in range(1, EPISODES + 1):
        env = KArmedBanditEnv()
        player = OptimisticBandit(env, epsilon, initial)
        optimal_actions_bool = []
        for _ in range(1, STEPS+1):
            action = player.act()
            reward = env.step(action)
            player.learn(action, reward)
            optimal_actions_bool.append(action == env.optimal_action)
        for i in range(STEPS):
            avg_perc_opt_actions[i] += (optimal_actions_bool[i] - avg_perc_opt_actions[i]) / episode
    return avg_perc_opt_actions

STEPS = 1000
EPISODES = 2000
EPSILONS = (0.1, 0)
INITIALS = (0, 5)

with Pool() as pool:
    avg_perc_opt_actions = pool.starmap(exe, [(0.1, 0), (0, 5)])
    for i, eps in enumerate(EPSILONS):
        plt.plot(range(1, STEPS+1), avg_perc_opt_actions[i], label=f'init: {INITIALS[i]} e: {eps}')
    plt.legend()
    plt.title('% optimal actions')
    plt.xticks([1] + list(range(200, STEPS+1, 200)))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.savefig('figure_2.3.png')
    plt.show()

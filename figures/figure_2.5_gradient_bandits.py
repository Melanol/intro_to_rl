""" Page 42. """

from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from envs.k_armed_bandit_env import KArmedBanditEnv
from algos.gradient_bandit import GradientBandit


STEPS = 1000
EPISODES = 200
ALPHAS = (0.1, 0.4, 0.1, 0.4)
BASELINES = (True, True, False, False)
INPUT = ((GradientBandit, (alpha, baseline)) for alpha, baseline in zip(ALPHAS, BASELINES))


def exe(input):
    agent_class, params = input
    alpha, baseline = params
    avg_perc_opt_actions = np.zeros(STEPS)
    avg_reward = 0
    for episode in range(1, EPISODES+1):
        env = KArmedBanditEnv()
        agent = agent_class(env, alpha, baseline)
        avg_ep_reward = 0
        optimal_actions_bool = []
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = env.step(action)
            avg_ep_reward += (reward - avg_ep_reward) / step
            agent.learn(action=action, reward=reward, avg_ep_reward=avg_ep_reward)
            optimal_actions_bool.append(action == env.optimal_action)
        avg_reward += (avg_ep_reward - avg_reward) / episode
        for i in range(STEPS):
            avg_perc_opt_actions[i] += (optimal_actions_bool[i] - avg_perc_opt_actions[i]) / episode
    return avg_perc_opt_actions


with Pool() as pool:
    results = pool.map(exe, INPUT)
    for i in range(len(ALPHAS)):
        plt.plot(results[i], label=f'alpha: {ALPHAS[i]} baseline: {BASELINES[i]}')
    plt.legend()
    plt.title('Gradient bandits')
    plt.ylabel(f'% optimal action')
    plt.xticks([1] + list(range(200, STEPS+1, 200)))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # plt.savefig('figure_2.5_gradient_bandits.png')
    plt.show()

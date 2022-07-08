""" Page 36. """

from multiprocessing import Pool
from matplotlib import pyplot as plt

from envs.k_armed_bandit_env import KArmedBanditEnv
from algos.optimistic_bandit import OptimisticBandit
from algos.UCB_bandit import UCBBandit
import numpy as np


STEPS = 1000
EPISODES = 2000
AGENTS = [OptimisticBandit(), UCBBandit()]

def exe(agent):
    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES+1):
        env = KArmedBanditEnv()
        agent.assign_env(env)
        rewards = []
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = env.step(action)
            agent.learn(action, reward)
            rewards.append(reward)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode
    return avg_rewards

with Pool() as pool:
    avg_rewards = pool.map(exe, AGENTS)
    plt.plot(range(1, STEPS+1), avg_rewards[1], label=f'e-greedy e: {AGENTS[0].epsilon}')
    plt.plot(range(1, STEPS+1), avg_rewards[0], label=f'USB c: {AGENTS[1].c}')
    plt.legend()
    plt.title('Average rewards')
    plt.xticks([1] + list(range(200, STEPS+1, 200)))
    plt.savefig('figure_2.4_bandits_USB_vs_epsilon.png')
    plt.show()

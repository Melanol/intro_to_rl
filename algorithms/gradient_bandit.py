""" Page 37. Not using baselines results in overflow. There is a solution, but the algo without baselines
is useless anyway. """

from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt

from environments.k_armed_bandit_env import KArmedBanditEnv


class Agent:
    def __init__(self, env):
        self.env = env
        self.H = np.zeros(len(env.action_space))
        self.softmax()

    def softmax(self):
        self.pi = np.exp(self.H) / np.sum(np.exp(self.H))

    def act(self):
        return np.random.choice(self.env.action_space, p=self.pi)  # Choosing randomly according to a prob distro


def exe(alpha):
    avg_rewards = np.zeros(STEPS)
    env = KArmedBanditEnv()
    for episode in range(1, EPISODES+1):
        agent = Agent(env)
        rewards = []
        avg_reward = 0
        for step in range(1, STEPS+1):
            action = agent.act()
            reward = env.step(action)
            avg_reward += (reward - avg_reward) / step
            agent.H[action] += alpha * (reward - avg_reward) * agent.pi[action]
            for a in env.action_space:
                if a != action:
                    agent.H[a] -= alpha * (reward - avg_reward) * (1 - agent.pi[a])
            agent.softmax()
            rewards.append(reward)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode

        if episode % PRINT_EACH_N_EPISODES == 0:
            print(f'Alpha: {alpha}; episode: {episode} complete')

        env.reset()
    return avg_rewards


EPISODES = 100
PRINT_EACH_N_EPISODES = 100
STEPS = 1000
ALPHAS = (0.1, 0.4)
with Pool() as pool:
    arr = pool.map(exe, ALPHAS)
    for i, alpha in enumerate(ALPHAS):
        plt.plot(arr[i], label=f'alpha: {alpha}')
        plt.xlabel('Steps')
        plt.ylabel(f'Average rewards per {EPISODES} episodes')
plt.legend()
# plt.savefig('gradient_bandit.png')
plt.show()

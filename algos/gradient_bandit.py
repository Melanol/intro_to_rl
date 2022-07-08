"""
Page 37. Not using baselines or using too big alpha results in a crash,
so we have to use "self.H - np.max(self.H)" instead of "self.H" in GradientBandit.softmax().
The explanation is here: https://www.datahubbs.com/multi-armed-bandits-reinforcement-learning-2/
in (Softmax Nuance).
"""

import numpy as np


class GradientBandit:
    def __init__(self, env, alpha=0.1, baseline=True):
        self.env = env
        self.alpha = alpha
        self.baseline = baseline
        self.H = np.zeros(len(env.action_space))
        self.softmax()

    def softmax(self):
        # Here, we had to use "self.H - np.max(self.H)" instead of "self.H" because otherwise we get crashes when
        # not using baselines of having too big alpha.
        self.pi = np.exp(self.H - np.max(self.H)) / np.sum(np.exp(self.H - np.max(self.H)))

    def act(self, step):
        # Choosing randomly according to a prob distro
        return np.random.choice(self.env.action_space, p=self.pi)

    def learn(self, **kwargs):
        action = kwargs['action']
        reward = kwargs['reward']
        avg_ep_reward = kwargs['avg_ep_reward']
        if self.baseline:
            self.H[action] += self.alpha * (reward - avg_ep_reward) * self.pi[action]
            for a in self.env.action_space:
                if a != action:
                    self.H[a] -= self.alpha * (reward - avg_ep_reward) * (1 - self.pi[a])
        else:
            self.H[action] += self.alpha * (reward) * self.pi[action]
            for a in self.env.action_space:
                if a != action:
                    self.H[a] -= self.alpha * (reward) * (1 - self.pi[a])
        self.softmax()


def exe():
    from matplotlib import pyplot as plt

    from envs.k_armed_bandit_env import KArmedBanditEnv

    EPISODES = 20
    STEPS = 1000

    avg_rewards = np.zeros(STEPS)
    env = KArmedBanditEnv()
    for episode in range(1, EPISODES+1):
        agent = GradientBandit(env, 4)
        rewards = []
        avg_ep_reward = 0
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = env.step(action)
            avg_ep_reward += (reward - avg_ep_reward) / step
            agent.learn(action=action, reward=reward, avg_ep_reward=avg_ep_reward)
            rewards.append(reward)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode

        env.reset()
    plt.plot(avg_rewards)
    plt.xlabel('Steps')
    plt.ylabel(f'Average rewards per {EPISODES} episodes')
    # plt.savefig('gradient_bandit.png')
    plt.show()


if __name__ == '__main__':
    exe()

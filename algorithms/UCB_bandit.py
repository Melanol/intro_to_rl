""" Page 35. """

import math


class UCBBandit:
    def __init__(self, env=None, c=2):
        self.env = env
        self.c = c
        try:
            self.Q = [0] * len(env.action_space)
            self.uses = [0] * len(env.action_space)
        except AttributeError:
            pass

    def assign_env(self, env):
        self.env = env
        self.Q = [0] * len(env.action_space)
        self.uses = [0] * len(env.action_space)

    def act(self, step):
        max_preference = -math.inf
        max_action = None
        for a in self.env.action_space:
            try:
                action_preference = self.Q[a] + self.c * (math.log(step) / self.uses[a])**0.5
            except ZeroDivisionError:  # If action was never selected, select it
                self.uses[a] += 1
                return a
            if action_preference > max_preference:
                max_preference = action_preference
                max_action = a
        self.uses[max_action] += 1
        return max_action

    def learn(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) / self.uses[action]


def exe():
    from matplotlib import pyplot as plt
    import numpy as np

    from environments.k_armed_bandit_env import KArmedBanditEnv

    STEPS = 1000
    EPISODES = 2000

    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES+1):
        env = KArmedBanditEnv()
        agent = UCBBandit(env, c=2)
        rewards = []
        for step in range(1, STEPS+1):
            action = agent.act(step)
            reward = env.step(action)
            agent.learn(action, reward)
            rewards.append(reward)
        for i in range(STEPS):
            avg_rewards[i] += (rewards[i] - avg_rewards[i]) / episode

    plt.xticks([1] + list(range(200, STEPS+1, 200)))
    plt.plot(range(1, STEPS+1), avg_rewards)
    plt.title('Average rewards')
    plt.show()

if __name__ == '__main__':
    exe()

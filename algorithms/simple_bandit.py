"""Page 28. Learning is already incremental."""

import random
import math


class SimpleBandit:
    def __init__(self, env=None, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.Q = [0] * len(env.action_space)
        self.uses = [0] * len(env.action_space)

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
        self.uses[action] += 1
        return action

    def learn(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) / self.uses[action]


def exe():
    from matplotlib import pyplot as plt
    import numpy as np

    from environments.k_armed_bandit_env import KArmedBanditEnv

    STEPS = 1000
    EPISODES = 2000
    EPSILON = 0.1
    ENV = KArmedBanditEnv()

    avg_rewards = np.zeros(STEPS)
    for episode in range(1, EPISODES+1):
        ENV.reset()
        agent = SimpleBandit(ENV, EPSILON)
        rewards = []
        for _ in range(1, STEPS+1):
            action = agent.act()
            reward = ENV.step(action)
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

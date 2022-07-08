"""Page 147."""

import math
import random
from matplotlib import pyplot as plt


class Sarsa:
    def __init__(self, env=None, epsilon=0.1, alpha=0.1, discount=0.9, default_Q=0, n=2):
        self.env = env
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.default_Q = default_Q
        self.n = n

    def act(self, obs):
        if random.random() <= self.epsilon:
            return random.choice(self.env.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.env.action_space:
                action_value = self.Q.get((obs, action), self.default_Q)
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action


def exe():
    from envs.windy_gridworld import WindyGridworld

    ENV = WindyGridworld()
    EPISODES = 100
    EPSILON = 0.3
    ALPHA = 0.5
    GAMMA = 1
    DEFAULT_Q = 0
    n = 2

    agent = Sarsa(ENV, EPSILON, ALPHA, GAMMA, DEFAULT_Q, n)
    ENV.agent = agent

    steps_to_plot = []
    for _ in range(EPISODES):
        obs = ENV.reset()
        obses = [obs]
        actions = [agent.act(obs)]
        rewards = [0]  # I think the 0th element must be 0, and we just never use it
        T = math.inf
        done = False
        t = 0
        while True:
            if t < T:
                obs, reward, done = ENV.step(actions[t])
                obses.append(obs)
                rewards.append(reward)
                if done:
                    T = t + 1
                else:
                    actions.append(agent.act(obs))
            tau = t - n + 1
            if tau >= 0:
                G = 0
                lower_limit = t + 1
                upper_limit = min(tau+n, T)
                for i in range(lower_limit, upper_limit+1):
                    G += GAMMA ** (i-tau-1) * rewards[i]
                    if tau + n < T:
                        G += GAMMA ** n * agent.Q.get((obses[tau+n], actions[tau+n]), DEFAULT_Q)
                    agent.Q[(obses[tau], actions[tau])] = \
                        agent.Q.get((obses[tau], actions[tau]), DEFAULT_Q) + ALPHA * (G
                        - agent.Q.get((obses[tau], actions[tau]), DEFAULT_Q))
            t += 1
            if tau == T - 1:
                break
            print(t)
        steps_to_plot.append(t)

    plt.plot(steps_to_plot)
    plt.show()


if __name__ == '__main__':
    exe()

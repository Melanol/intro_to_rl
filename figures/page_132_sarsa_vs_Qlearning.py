"""Page 132."""

# TODO: Abandoned for now: head hurts

import numpy as np
import matplotlib.pyplot as plt

from envs.cliff_walking import CliffWalking
from algos.sarsa import Sarsa
from algos.Qlearning import QLearning


RUNS = 100
EPISODES = 500
EPSILON = 0.1
ALPHA = 0.5
DISCOUNT = 1
DEFAULT_Q = 0


def exe_sarsa():
    ENV = CliffWalking()
    sarsa_sums = np.zeros(EPISODES)
    for run in range(1, RUNS+1):
        agent = Sarsa(ENV)
        ENV.agent = agent

        for ep in range(EPISODES):
            obs = ENV.reset()
            action = agent.act(obs)
            done = False
            ep_reward_sum = 0
            while not done:
                next_obs, reward, done = ENV.step(action)
                next_action = agent.act(next_obs)
                agent.learn(obs, action, reward, next_obs, next_action)
                obs, action = next_obs, next_action
                ep_reward_sum += reward
            sarsa_sums[ep] += (ep_reward_sum - sarsa_sums[ep]) / run
    return sarsa_sums

def exe_Qlearning():
    ENV = CliffWalking()
    EPSILON = 0.1
    ALPHA = 0.1
    DISCOUNT = 1
    DEFAULT_Q = 0

    agent = QLearning(ENV, EPSILON, ALPHA, DISCOUNT, DEFAULT_Q)
    ENV.agent = agent

    Qlearning_ep_sums = []
    for _ in range(EPISODES):
        obs = ENV.reset()
        done = False
        reward_sum = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, done = ENV.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            reward_sum += reward
        Qlearning_ep_sums.append(reward_sum)
    return Qlearning_ep_sums


if __name__ == '__main__':
    sarsa_sums = exe_sarsa()
    print(sarsa_sums)
    # QLearning_sums = exe_Qlearning()
    plt.plot(sarsa_sums)
    # plt.plot(QLearning_sum)
    plt.ylim(-100, 0)
    plt.show()

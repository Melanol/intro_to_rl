""" Page 42. """

from multiprocessing import Pool
from matplotlib import pyplot as plt

from envs.k_armed_bandit_env import KArmedBanditEnv
from algos.epsilon_bandit import EpsilonBandit
from algos.optimistic_bandit import OptimisticBandit
from algos.UCB_bandit import UCBBandit
from algos.gradient_bandit import GradientBandit


STEPS = 1000
EPISODES = 200
TICKS = (1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4)
EPSILON_BANDIT_PARAMS = (1/128, 1/64, 1/32, 1/16, 1/8, 1/4)
EPSILON_BANDIT = (EpsilonBandit, EPSILON_BANDIT_PARAMS)
OPTIMISTIC_PARAMS = (1/4, 1/2, 1, 2, 4)
OPTIMISTIC = (OptimisticBandit, OPTIMISTIC_PARAMS)
UCB_PARAMS = (1/16, 1/8, 1/4, 1/2, 1, 2, 4)
UCB = (UCBBandit, UCB_PARAMS)
GRADIENT_PARAMS = (1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4)
GRADIENT = (GradientBandit, GRADIENT_PARAMS)
INPUT = [EPSILON_BANDIT, OPTIMISTIC, UCB, GRADIENT]


def exe(input):
    agent_class, params = input
    avg_rewards = []
    for param in params:
        avg_reward = 0
        for episode in range(1, EPISODES+1):
            env = KArmedBanditEnv()
            agent = agent_class(env, param)
            avg_ep_reward = 0
            for step in range(1, STEPS+1):
                action = agent.act(step)
                reward = env.step(action)
                avg_ep_reward += (reward - avg_ep_reward) / step
                agent.learn(action=action, reward=reward, avg_ep_reward=avg_ep_reward)
            avg_reward += (avg_ep_reward - avg_reward) / episode
        avg_rewards.append(avg_reward)
        print(f'{agent_class} param: {param} done')
    return avg_rewards


with Pool() as pool:
    avg_rewards = pool.map(exe, INPUT)
    plt.plot(EPSILON_BANDIT_PARAMS, avg_rewards[0], label='epsilon bandit: epsilon')
    plt.plot(OPTIMISTIC_PARAMS, avg_rewards[1], label='optimistic: initial')
    plt.plot(UCB_PARAMS, avg_rewards[2], label='UCB: c')
    plt.plot(GRADIENT_PARAMS, avg_rewards[3], label='gradient: alpha')
    plt.xscale("log")
    plt.legend()
    plt.title('Bandit algorithms parameter study')
    plt.ylabel(f'Average reward over {STEPS} steps')
    plt.xticks(TICKS, ('1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'))
    plt.savefig('figure_2.6_bandit_param_study.png')
    plt.show()
    # plt.show()

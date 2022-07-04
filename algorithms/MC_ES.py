""" Page 99. Monte Carlo Exploring Starts. """
import random
import math

from environments.blackjack import Blackjack


class MSESAgent:
    def __init__(self, env):
        self.env = env
        self.pi = {}  # We fill pi and Q as we encounter new states
        self.Q = {}

    def act(self, obs):
        # The obs is decoded dealer's showing card and the sum in our hand in the form (int, int)
        if obs[1] >= 20:
            return 'stick'
        else:
            return 'hit'

    def learn(self):
        pass


def exe():
    EPISODES = 100
    GAMMA = 1
    DEFAULT_Q = 0

    env = Blackjack()
    agent = MSESAgent(env)
    env.agent = agent
    returns = {}
    for _ in range(EPISODES):
        obs = env.reset()

        # # Exploring start (since it's Blackjack, start states are already random):
        # action = random.choice(['hit', 'stick'])
        # obs, reward, done, _ = env.step(action)
        # if done:
        #     returns.append(reward)
        #     continue

        observations = []
        actions = []
        rewards = []
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            # if not done:  # Cannot include terminal stuff
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        G = 0
        for t in reversed(range(len(observations))):
            if rewards[t]:
                G = GAMMA * G + rewards[t]
            St_At = observations[t], actions[t]
            found = False
            for S, A in zip(observations[:-1], actions[:-1]):
                if (S, A) == St_At:  # FIXME: Finds it each time
                    found = True
                    break
            if not found:
                try:
                    returns[St_At].append(G)
                except KeyError:
                    returns[St_At] = []
                try:
                    agent.Q[St_At] = sum(returns[St_At]) / len(returns[St_At])
                except ZeroDivisionError:
                    agent.Q[St_At] = DEFAULT_Q

                max_action_value = -math.inf
                max_action = None
                for a in env.action_space:
                    try:
                        action_value = agent.Q[(observations[t], a)]
                    except KeyError:
                        agent.Q[(observations[t], a)] = DEFAULT_Q
                        action_value = agent.Q[(observations[t], a)]
                    if action_value > max_action_value:
                        max_action_value = action_value
                        max_action = a
                agent.pi[observations[t]] = max_action
                # TODO: agent.learn() instead
    print(agent.Q)

if __name__ == '__main__':
    exe()

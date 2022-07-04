"""Page 130."""
# TODO: Make sure the algo works

import math
import random
from matplotlib import pyplot as plt

from environments.windy_gridworld import


EPISODES = 200
EPSILON = 0.1
ALPHA = 0.5
DISCOUNT = 1
INITIAL_Q = 0


class Agent:
    def __init__(self, game):
        self.game = game
        self.Q = {}
        for state in range(len(game.all_states)):
            for action in game.action_space:
                self.Q[(state, action)] = 0

    def act(self, state):
        if random.random() <= EPSILON:
            return random.choice(self.game.action_space)
        else:
            max_action_value = -math.inf
            max_action = None
            for action in self.game.action_space:
                action_value = self.Q[(state, action)]
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = action
            return max_action

    def print_Q(self):
        for key, value in list(self.Q.items())[2:-2]:
            print(key, round(value, 2))


env = RandomWalk()
player = Agent(env)
plot_actions_vs_episodes = []
for _ in range(EPISODES):
    plot_actions = []
    S = env.start()
    action = player.act(S)
    while True:
        plot_actions.append(action)
        reward, next_state, termination = env.step(action)
        next_action = player.act(next_state)
        player.Q[(S, action)] += ALPHA * (
                reward + DISCOUNT * player.Q[(next_state, next_action)] - player.Q[(S, action)]
        )
        S, action = next_state, next_action
        if termination:
            break
    plot_actions_vs_episodes.append(plot_actions)


# Plotting:

player.print_Q()
ys_to_plot = []
for plot_actions in plot_actions_vs_episodes:
    ys_to_plot.append(sum(1 for el in plot_actions if el == 'right') / len(plot_actions))
plt.plot(ys_to_plot)

count = 1
means_to_plot = []
ys = []
for y in ys_to_plot:
    ys.append(y)
    if count % 10 == 0:
        means_to_plot.append(sum(ys) / len(ys))
        ys = []
    count += 1
plt.plot([10 * x for x in range(EPISODES // 10)], means_to_plot)
plt.title('Mean of the rate of right actions chosen')
plt.show()

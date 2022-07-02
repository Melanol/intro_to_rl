import random

from environments.random_walk import RandomWalk


EPISODES = 1000
STEP_SIZE = 0.1
DISCOUNT = 1

class TDRandomAgent:
    def __init__(self, game):
        self.game = game
        self.V = [(0.5 if state != 'T' else 0) for state in game.all_states]

    def act(self):
        return random.choice(self.game.action_space)

    def print_V(self):
        print([round(el, 2) for el in self.V])


env = RandomWalk()
player = TDRandomAgent(env)
env.player = player
# Estimating v(pi) (page 120):
for _ in range(EPISODES):
    S = env.start()
    while True:
        action = player.act()
        reward, next_state, termination = env.step(action)
        player.V[S] += STEP_SIZE * (reward + DISCOUNT * player.V[next_state] - player.V[S])
        S = next_state
        if termination:
            break

player.print_V()

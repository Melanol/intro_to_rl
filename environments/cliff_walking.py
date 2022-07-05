"""
Page 132. We drop to start if we enter a cliff cell (without reset()).

0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
S C C C C C C C C C C G

0 = empty space; S = start; C = cliff; G = goal.
"""

import random


class CliffWalking:

    class Coord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, width=12, height=4, start=Coord(0, 3), goal=Coord(11, 3)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.cliff = [(x, height-1) for x in range(1, width-1)]
        self.action_space = ['up', 'down', 'left', 'right']
        self.agent = None

    def reset(self):
        self.agent.x = self.start.x
        self.agent.y = self.start.y
        obs = self.start
        reward = 0
        done = False
        return obs, reward, done

    def step(self, action):
        reward = -1
        done = False

        # Regular movement
        if action == 'up':
            self.agent.y -= 1
        elif action == 'down':
            self.agent.y += 1
        elif action == 'left':
            self.agent.x -= 1
        elif action == 'right':
            self.agent.x += 1

        # Cliff
        for x, y in self.cliff:
            if (self.agent.x, self.agent.y) == (x, y):
                reward = -100
                self.agent.x, self.agent.y = self.start.x, self.start.y

        # Hitting walls
        if self.agent.x < 0:
            self.agent.x = 0
        elif self.agent.x > self.width - 1:
            self.agent.x = self.width - 1
        elif self.agent.y < 0:
            self.agent.y = 0
        elif self.agent.y > self.height - 1:
            self.agent.y = self.height - 1

        # Check reaching goal
        if (self.agent.x, self.agent.y) == (self.goal.x, self.goal.y):
            done = True

        obs = self.agent.x, self.agent.y
        return obs, reward, done

    def render(self):
        """Render the grid in the console."""
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == (self.agent.x, self.agent.y):
                    print('X', end='  ')
                elif (x, y) in self.cliff:
                    print('C', end='  ')
                elif (x, y) == (self.goal.x, self.goal.y):
                    print('G', end='  ')
                else:
                    print('□', end='  ')
            print()
        print()


class RandomAgent:
    def __init__(self, game):
        self.game = game
        self.x = None
        self.y = None

    def act(self, obs):
        return random.choice(self.game.action_space)

    def learn(self, action, obs):
        pass


class Human:
    def __init__(self, env):
        self.env = env
        self.x = None
        self.y = None

    def act(self, obs):
        self.env.render()
        while True:
            action = input('Your move: ')
            if action not in ('left', 'right', 'up', 'down'):
                print('Action must be "left", "right", "up", or "down".')
            else:
                return action

    def learn(self, action, obs):
        pass


def exe():
    env = CliffWalking()
    # agent = RandomAgent(env)
    agent = Human(env)
    env.agent = agent

    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = agent.act(obs)
        obs, reward, done = env.step(action)
        agent.learn(action, reward)
        print(obs)
        step += 1
    print(f'Finished in {step} steps')


if __name__ == '__main__':
    exe()

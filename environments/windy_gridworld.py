"""Page 130. The image in the book is confusing in wind power, so I had to improvise.
No discount. -1 for each step."""


import random


class WindyGridworld:

    class Coord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class WindyColumn:
        def __init__(self, x, strength):
            self.x = x
            self.strength = strength

    def __init__(self, width=10, height=7, start=Coord(0, 3), goal=Coord(7, 3), windy_columns=(
            WindyColumn(x=3, strength=1),
            WindyColumn(x=4, strength=1),
            WindyColumn(x=5, strength=1),
            WindyColumn(x=6, strength=2),
            WindyColumn(x=7, strength=2),
            WindyColumn(x=8, strength=1),
    )):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.windy_columns = windy_columns
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
        reward = -1  # Reward is a constant
        done = False

        # Windy columns
        for windy_column in self.windy_columns:
            if windy_column.x == self.agent.x:
                self.agent.y -= windy_column.strength

        # Regular movement
        if action == 'up':
            self.agent.y -= 1
        elif action == 'down':
            self.agent.y += 1
        elif action == 'left':
            self.agent.x -= 1
        elif action == 'right':
            self.agent.x += 1

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
                elif (x, y) == (self.goal.x, self.goal.y):
                    print('G', end='  ')
                else:
                    print('□', end='  ')
            print()
        for x in range(self.width):
            found = False
            for windy_column in self.windy_columns:
                if windy_column.x == x:
                    print(windy_column.strength, end='  ')
                    found = True
                    break
            if not found:
                print(0, end='  ')
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
    env = WindyGridworld()
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

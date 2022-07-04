"""No discount. -1 for each step."""


import random


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Gridworld:

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
        """1st we move, then we apply windy columns, then we make sure we are within the borders, then we
        check if we have reached the goal.
        """

        # Reward is a constant
        reward = -1

        # Regular movement
        if action == 'up':
            self.agent.y -= 1
        elif action == 'down':
            self.agent.y += 1
        elif action == 'left':
            self.agent.x -= 1
        elif action == 'right':
            self.agent.x += 1

        # Windy columns
        for windy_column in self.windy_columns:
            if windy_column.x == self.agent.x:
                self.agent.y -= windy_column.strength

        # Hitting walls
        if self.agent.x < 0:
            self.agent.x = 0
        elif self.agent.x > self.width - 1:
            self.agent.x = self.width - 1
        elif self.agent.y < 0:
            self.agent.y = 0
        elif self.agent.y > self.height - 1:
            self.agent.y = self.height - 1

        # Reaching goal
        done = False
        if (self.agent.x, self.agent.y) == (self.goal.x, self.goal.y):
            done = True

        obs = self.agent.x, self.agent.y
        return obs, reward, done


class RandomAgent:
    def __init__(self, game):
        self.game = game
        self.x = None
        self.y = None
        self.state_selection_n = {}
        self.value_function = {}
        for y in range(game.width):
            for x in range(game.height):
                self.state_selection_n[(x, y)] = 0
                self.value_function[(x, y)] = 0

    def act(self, obs):
        return random.choice(self.game.action_space)

    def learn(self, action, obs):
        pass


def exe():
    env = Gridworld()
    agent = RandomAgent(env)
    env.agent = agent
    done = False
    obs = env.reset()
    step = 1
    while not done:
        action = agent.act(obs)
        obs, reward, done = env.step(action)
        agent.learn(action, reward)
        print(obs)
        step += 1
    print(step)
if __name__ == '__main__':
    exe()

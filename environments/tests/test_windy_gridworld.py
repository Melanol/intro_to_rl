from environments.windy_gridworld import WindyGridworld, Human


def test_reaching_goal():

    class TestAgent:
        def __init__(self, game):
            self.game = game
            self.x = None
            self.y = None

        @staticmethod
        def action_generator():
            for action in ('right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right',
                           'down', 'down', 'down', 'down', 'left', 'left'):
                yield action

    env = WindyGridworld()
    agent = TestAgent(env)
    env.agent = agent
    action_iterator = agent.action_generator()

    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = next(action_iterator)
        obs, reward, done = env.step(action)
        step += 1

    assert (obs, done, step) == ((7, 3), True, 15)

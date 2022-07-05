from environments.cliff_walking import CliffWalking


def test_cliff():

    class TestAgent:
        def __init__(self, game):
            self.game = game
            self.x = None
            self.y = None

        @staticmethod
        def act(obs):
            return 'right'

    env = CliffWalking()
    agent = TestAgent(env)
    env.agent = agent

    obs = env.reset()
    action = agent.act(obs)
    obs, reward, done = env.step(action)
    assert (obs, reward) == ((env.start.x, env.start.y), -100)

def test_reaching_goal():

    class TestAgent:
        def __init__(self, game):
            self.game = game
            self.x = None
            self.y = None

        @staticmethod
        def action_generator():
            for action in ('up', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right',
                           'right', 'right', 'right', 'down'):
                yield action

    env = CliffWalking()
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

    assert (obs, done, step) == ((11, 3), True, 13)

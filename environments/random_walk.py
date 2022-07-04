"""Page 125."""


class RandomWalk:
    def __init__(self):
        self.all_states = ['T', 'A', 'B', 'C', 'D', 'E', 'T']
        self.action_space = ['left', 'right']

    def start(self):
        self.state = 3  # We start in the middle
        return self.state

    def step(self, action):
        reward = 0
        termination = False
        if action == 'left':
            self.state -= 1
        elif action == 'right':
            self.state += 1
        if self.state == 0:
            termination = True
        elif self.state == len(self.all_states) - 1:
            reward = 1
            termination = True
        return reward, self.state, termination

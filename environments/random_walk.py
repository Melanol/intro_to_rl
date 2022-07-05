"""Page 125."""


class RandomWalk:
    def __init__(self):
        self.all_states = ['T', 'A', 'B', 'C', 'D', 'E', 'T']
        self.action_space = ['left', 'right']

    def reset(self):
        self.state = 3  # We start in the middle
        return self.state

    def step(self, action):
        reward = 0
        done = False
        if action == 'left':
            self.state -= 1
        elif action == 'right':
            self.state += 1
        if self.state == 0:
            done = True
        elif self.state == len(self.all_states) - 1:
            reward = 1
            done = True
        return self.state, reward, done

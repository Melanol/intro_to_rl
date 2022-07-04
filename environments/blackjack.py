"""Page 93. The deck is infinite (i.e., with replacement)."""

import random


class StickOn21Agent:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        # The obs is decoded dealer's showing card and the sum in our hand in the form (int, int)
        if obs[1] >= 20:  # The agent sticks only on 20 and 21
            return 'stick'
        else:
            return 'hit'

class Blackjack:
    def __init__(self):
        self.deck = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.action_space = ['hit', 'stick']

    def reset(self):
        self.dealer_hand = self.draw_cards(2)
        self.dealer_showing = random.choice(self.dealer_hand)
        self.first_turn = True
        return self.sum([self.dealer_showing]), 0

    def draw_cards(self, n):
        if n == 1:
            return random.choice(self.deck)
        else:  # n == 2
            return [random.choice(self.deck) for _ in range(2)]

    def sum(self, hand):
        if 'A' in hand:
            sum_ = 1
            for card in hand:
                if card != 'A':
                    sum_ += 1
            if sum_ + 10 <= 21:
                sum_ = sum_ + 10
                return sum_
            else:
                return sum_
        else:
            return sum(hand)

    def dealer_sum(self):
        return self.sum(self.dealer_hand)

    def agent_sum(self):
        return self.sum(self.agent.hand)

    def compare_hands(self):
        dealer_hand = self.dealer_sum()
        agent_hand = self.agent_sum()

        if dealer_hand > agent_hand:
            reward = -1
        elif dealer_hand < agent_hand:
            reward = 1
        else:
            reward = 0

        return reward

    def step(self, action):
        """ Win = 1, loss = -1, draw = 0 """
        if self.first_turn:
            self.agent.hand = self.draw_cards(2)
            if self.agent_sum() == 21:
                return (0, 0), 1, True, {}  # Natural
            self.first_turn = False
            return (self.dealer_showing, self.agent_sum()), None, False, {}
        else:
            done = False
            reward = None

            if action == 'hit':
                self.agent.hand.append(self.draw_cards(1))
                if self.agent_sum() > 21:
                    reward = -1
                    done = True
            else:  # action == 'stick'
                while self.dealer_sum() < 17:  # Dealer hits until reaches 17
                    self.dealer_hand.append(self.draw_cards(1))
                if self.dealer_sum() > 21:
                    reward = 1
                else:
                    reward = self.compare_hands()
                done = True

            obs = self.dealer_showing, self.agent_sum()
            info = {}

            return obs, reward, done, info


def exe():
    EPISODES = 100

    env = Blackjack()
    agent = StickOn21Agent(env)
    env.agent = agent
    returns = []
    for _ in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
        returns.append(reward)

    count_wins = 0
    count_losses = 0
    count_draws = 0
    for reward in returns:
        if reward == 1:
            count_wins += 1
        elif reward == -1:
            count_losses += 1
        else:
            count_draws += 1
    print(f'Win percentage: {count_wins / len(returns)}')
    print(f'Loss percentage: {count_losses / len(returns)}')
    print(f'Draw percentage: {count_draws / len(returns)}')

if __name__ == '__main__':
    exe()

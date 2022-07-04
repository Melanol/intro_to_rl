"""Multiple states."""

import random
from pprint import pprint as print


N_STATES = 3
N_ARMS = 10
STEPS = 1000 * N_STATES
TRUE_VALUES = [[random.randint(1, 10) for _ in range(N_ARMS)] for _ in range(N_STATES)]
print([[round(float(v), 1) for v in state] for state in TRUE_VALUES])
EPSILON = 0.1

def get_value(state, arm):
    mu = TRUE_VALUES[state][arm]
    sigma = 2.5
    return random.gauss(mu, sigma)

val_estimates = [[0] * N_ARMS for _ in range(N_STATES)]
uses = [[0] * N_ARMS for _ in range(N_STATES)]
# print(val_estimates)
# print(uses)

for _ in range(STEPS):
    state = random.randint(0, N_STATES-1)  # Random state
    if random.random() <= EPSILON:  # Choose random action
        arm = random.randint(0, N_ARMS-1)
    else:
        arm = val_estimates[state].index(max(val_estimates[state]))
    reward = get_value(state, arm)
    uses[state][arm] += 1
    val_estimates[state][arm] += (reward - val_estimates[state][arm]) / uses[state][arm]

print([[round(v, 1) for v in state] for state in val_estimates])

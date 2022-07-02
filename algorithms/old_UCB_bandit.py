"""Upper-confidence-bound action selection."""
import random
import math

from matplotlib import pyplot as plt


N_ARMS = 10
STEPS = 1000
TRUE_VALUES = [random.randint(1, 10) for _ in range(N_ARMS)]
print([round(float(v), 2) for v in TRUE_VALUES])
c = 0.1  # Controls the degree of exploration

def get_value(arm):
    mu = TRUE_VALUES[arm]
    sigma = 2.5
    return random.gauss(mu, sigma)

rewards = []
# 1st run
values = [[get_value(arm)] for arm in range(N_ARMS)]
uses = [1 for arm in range(N_ARMS)]
val_estimates = [v[0] for v in values]

# The rest runs
for step in range(1, STEPS+1):
    selected_arm = None
    max_val = -math.inf
    for arm, est in enumerate(val_estimates):
        val = est + c * math.sqrt(math.log(step) / uses[arm])
        if val > max_val:
            selected_arm = arm
            max_val = val
    reward = get_value(arm)
    uses[arm] += 1
    val_estimates[arm] += (reward - val_estimates[arm]) / uses[arm]
    rewards.append(reward)

print(f'{[round(v, 1) for v in val_estimates]}')
plt.plot([x for x in range(STEPS)], rewards)
plt.show()

import numpy as np
import random
from collections import defaultdict

items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)
n, dim = items.shape
scale = items.max(axis=0)
items=items/scale

Q = defaultdict(lambda: np.zeros(2))

# Hyper Parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 6000

# State Representation
def get_state(index, sumA, sumB):
    diff = sumA - sumB
    buckets = tuple(int(np.sign(d)) for d in diff)
    return (index,) + buckets

# ε-greedy function 
def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 1)
    return np.argmax(Q[state])


# Training Loop
for ep in range(episodes):

    sumA = np.zeros(dim)
    sumB = np.zeros(dim)
    index = 0

    state = get_state(index, sumA, sumB)
    action = choose_action(state)  

    while index < n:
        if action == 0:
            sumA += items[index]
        else:
            sumB += items[index]

        if index == n - 1:
            reward = -np.max(np.abs(sumA - sumB))
        else:
            reward = 0

        next_state = get_state(index + 1, sumA, sumB)
        next_action = choose_action(next_state)

        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state = next_state
        action = next_action
        index += 1

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 500 == 0:
        print(f"Episode {ep} completed")


# Final Greedy Policy
sumA = np.zeros(dim)
sumB = np.zeros(dim)
partition = []

for i in range(n):
    state = get_state(i, sumA, sumB)
    action = np.argmax(Q[state])
    partition.append(action)

    if action == 0:
        sumA += items[i]
    else:
        sumB += items[i]
sumA_original = sumA * scale
sumB_original = sumB * scale
final_diff_original = np.abs(sumA_original - sumB_original)
Max_imbalance_original = np.max(final_diff_original)
print("\nFinal Result")
print("Items:\n", items)
print("Partition (0=A, 1=B):", partition)
print("Sum A:", sumA_original)
print("Sum B:", sumB_original)
print("Final imbalance per dimension:", final_diff_original)
print("Max imbalance (original scale):", Max_imbalance_original)

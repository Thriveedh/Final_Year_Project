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

# Q Table
Q = defaultdict(lambda: np.zeros(2))

# Hyper Parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 6000

#State Representation
def get_state(index, sumA, sumB):
    diff = sumA - sumB
    buckets = tuple(int(np.sign(d)) for d in diff)
    return (index,) + buckets

# Training Loop
for ep in range(episodes):
    sumA = np.zeros(dim)
    sumB = np.zeros(dim)
    index = 0

    while index < n:
        state = get_state(index, sumA, sumB)

        # ε-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])

        # Apply action
        if action == 0:
            sumA += items[index]
        else:
            sumB += items[index]

        if index == n - 1:
            reward = -np.max(np.abs(sumA - sumB))
        else:
            reward = 0

        next_state = get_state(index + 1, sumA, sumB)

        # Q-learning update
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        index += 1

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 500 == 0:
        print(f"Episode {ep} completed")
    
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

print("\nFinal Result")
print("Items:\n", items)
print("Partition (0=A, 1=B):", partition)
# print("Sum A:", sumA*scale)
# print("Sum B:", sumB*scale)
print("Final imbalance:", np.abs(sumA - sumB))
print("Max imbalance:", np.max(np.abs(sumA - sumB)))
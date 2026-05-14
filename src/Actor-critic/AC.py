import numpy as np
import random
from collections import defaultdict

items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)
n, dim = items.shape
scale = items.max(axis=0)
items = items / scale

Actor = defaultdict(lambda: np.zeros(2))
Critic = defaultdict(float)

# Hyperparameters
actor_lr = 0.01      
critic_lr = 0.1      
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 5000

# State Representation
def get_state(index, sumA, sumB):
    diff = sumA - sumB
    buckets = tuple(int(np.sign(d)) for d in diff)
    return (index,) + buckets

# Softmax to get action probabilities from logits
def softmax(logits):
    logits = logits - np.max(logits)  
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()

# ε-greedy action selection using Actor's policy
def choose_action(state, eps):
    if random.random() < eps:
        return random.randint(0, 1)
    probs = softmax(Actor[state])
    return np.argmax(probs)

# Training Loop
for ep in range(episodes):
    sumA = np.zeros(dim)
    sumB = np.zeros(dim)
    index = 0
    prev_diff=0

    while index < n:
        state = get_state(index, sumA, sumB)
        action = choose_action(state, epsilon)

        # Take action
        if action == 0:
            sumA += items[index]
        else:
            sumB += items[index]

        new_diff = np.max(np.abs(sumA - sumB))
        reward = prev_diff - new_diff
        prev_diff = new_diff

        next_state = get_state(index + 1, sumA, sumB)

        if index == n - 1:
            td_target = reward
        else:
            td_target = reward + gamma * Critic[next_state]

        td_error = td_target - Critic[state]   
        Critic[state] += critic_lr * td_error

        probs = softmax(Actor[state])
        grad = -probs.copy()      
        grad[action] += 1.0
        Actor[state] += actor_lr * td_error * grad

        index += 1

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 500 == 0:
        print(f"Episode {ep} completed")

sumA = np.zeros(dim)
sumB = np.zeros(dim)
partition = []

for i in range(n):
    state = get_state(i, sumA, sumB)
    probs = softmax(Actor[state])
    action = np.argmax(probs)
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
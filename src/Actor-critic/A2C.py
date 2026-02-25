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

actor_lr = 0.01
critic_lr = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 6000

def get_state(index, sumA, sumB):
    diff = sumA - sumB
    buckets = tuple(int(np.sign(d)) for d in diff)
    return (index,) + buckets

def softmax(logits):
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()

def choose_action(state, eps):
    if random.random() < eps:
        return random.randint(0, 1)

    probs = softmax(Actor[state])
    return np.random.choice([0, 1], p=probs)



for ep in range(episodes):

    sumA = np.zeros(dim)
    sumB = np.zeros(dim)

    # trajectory buffers (IMPORTANT for A2C)
    states = []
    actions = []
    rewards = []
    prev_diff=0

    # ----- Generate Episode -----
    for index in range(n):

        state = get_state(index, sumA, sumB)
        action = choose_action(state, epsilon)

        states.append(state)
        actions.append(action)

        # take action
        if action == 0:
            sumA += items[index]
        else:
            sumB += items[index]

        new_diff = np.max(np.abs(sumA - sumB))
        reward = prev_diff - new_diff
        prev_diff = new_diff
        rewards.append(reward)

    # ----- Compute Returns (Monte Carlo) -----
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # ----- A2C Updates -----
    for t in range(n):

        state = states[t]
        action = actions[t]
        G = returns[t]

        # Advantage
        advantage = G - Critic[state]

        # Critic update
        Critic[state] += critic_lr * advantage

        # Actor update
        probs = softmax(Actor[state])
        grad = -probs.copy()
        grad[action] += 1.0

        Actor[state] += actor_lr * advantage * grad

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
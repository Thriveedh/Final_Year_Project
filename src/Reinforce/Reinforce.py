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

actor_lr = 0.01
gamma = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 1000

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

    states = []
    actions = []
    rewards = []

    for index in range(n):

        state = get_state(index, sumA, sumB)
        action = choose_action(state, epsilon)

        states.append(state)
        actions.append(action)

        # Take action
        if action == 0:
            sumA += items[index]
        else:
            sumB += items[index]

        reward = -np.sum(np.abs(sumA - sumB))
        rewards.append(reward)

    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = np.array(returns)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for t in range(n):

        state = states[t]
        action = actions[t]
        G = returns[t]

        probs = softmax(Actor[state])

        grad = -probs.copy()
        grad[action] += 1.0

        # POLICY GRADIENT UPDATE
        Actor[state] += actor_lr * G * grad

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


print("\nFinal Result")
print("Partition (0=A, 1=B):", partition)
print("Final imbalance:", np.abs(sumA - sumB))
import numpy as np
import random
import threading
from collections import defaultdict
import copy

items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)

n, dim = items.shape
scale = items.max(axis=0)
items = items / scale

GlobalActor = defaultdict(lambda: np.zeros(2))
GlobalCritic = defaultdict(float)

lock = threading.Lock()

actor_lr = 0.01
critic_lr = 0.1
gamma = 0.9

episodes_per_worker = 1500
num_workers = 4

epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

def get_state(index, sumA, sumB):
    diff = sumA - sumB
    buckets = tuple(int(np.sign(d)) for d in diff)
    return (index,) + buckets

def softmax(logits):
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()

def worker(worker_id):

    epsilon = epsilon_start

    for ep in range(episodes_per_worker):

        sumA = np.zeros(dim)
        sumB = np.zeros(dim)

        states = []
        actions = []
        rewards = []
        prev_diff=0

        for index in range(n):

            state = get_state(index, sumA, sumB)

            with lock:
                policy_logits = copy.deepcopy(GlobalActor[state])

            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                probs = softmax(policy_logits)
                action = np.random.choice([0, 1], p=probs)

            states.append(state)
            actions.append(action)

            if action == 0:
                sumA += items[index]
            else:
                sumB += items[index]

            new_diff = np.max(np.abs(sumA - sumB))
            reward = prev_diff - new_diff
            prev_diff = new_diff
            rewards.append(reward)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for t in range(n):

            state = states[t]
            action = actions[t]
            G = returns[t]

            with lock:

                # Advantage
                advantage = G - GlobalCritic[state]

                # Critic update
                GlobalCritic[state] += critic_lr * advantage

                # Actor update
                probs = softmax(GlobalActor[state])
                grad = -probs.copy()
                grad[action] += 1.0

                GlobalActor[state] += actor_lr * advantage * grad

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if ep % 300 == 0:
            print(f"Worker {worker_id} Episode {ep}")

threads = []

for wid in range(num_workers):
    t = threading.Thread(target=worker, args=(wid,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

sumA = np.zeros(dim)
sumB = np.zeros(dim)
partition = []

for i in range(n):

    state = get_state(i, sumA, sumB)
    probs = softmax(GlobalActor[state])

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
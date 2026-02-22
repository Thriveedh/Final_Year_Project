import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random

from env import PartitionEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer

items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)

items = items[:10]
scale=items.max(axis=0)
items = items / scale
print(scale)

env = PartitionEnv(items)

state_size = 1 + items.shape[1]
action_size = 2

model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

# Experience Replay Buffer
buffer = ReplayBuffer(capacity=20000)
batch_size = 64

# Hyper Parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.1

episodes = 6000
target_update_freq = 50

# Early stopping
best_reward = float("-inf")
patience = 500
no_improve = 0

for episode in range(episodes):

    state = torch.tensor(env.reset(), dtype=torch.float32)
    total_reward = 0

    while True:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()

        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:

            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = target_model(next_states).max(1)[0]

            targets = rewards + gamma * next_q * (1 - dones)

            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    if total_reward > best_reward + 1:
        best_reward = total_reward
        no_improve = 0
        torch.save(model.state_dict(), "best_dqn_model.pth")
    else:
        no_improve += 1

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}")

    if no_improve >= patience:
        print(f"\nEarly stopping at episode {episode}")
        break

model.load_state_dict(torch.load("best_dqn_model.pth"))

env.reset()
partition = []

for _ in range(len(items)):
    state = torch.tensor(env._get_state(), dtype=torch.float32)
    action = torch.argmax(model(state)).item()
    partition.append(action)
    env.step(action)

print("\nFinal Result")
print("Items:\n", items)
print("Partition:", partition)
# print("Sum A:", env.sumA)
# print("Sum B:", env.sumB)
print("Final imbalance:", np.abs(env.sumA - env.sumB))

# original_sumA = env.sumA * scale
# original_sumB = env.sumB * scale

# print("Real imbalance:", np.abs(original_sumA - original_sumB))


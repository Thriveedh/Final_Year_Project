import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)

n, dim = items.shape
items = items / items.max(axis=0)


gamma = 0.95
lr = 0.0007
episodes = 600          
entropy_beta = 0.01
n_step = 32             

STATE_DIM = dim + 2
ACTION_DIM = 2


def build_state(index, sumA, sumB):
    diff = sumA - sumB
    index_norm = index / n
    remaining = (n - index) / n
    return np.concatenate(([index_norm, remaining], diff))

class A2CNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(STATE_DIM, 64)
        self.fc2 = nn.Linear(64, 64)

        self.policy = nn.Linear(64, ACTION_DIM)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value

net = A2CNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

for ep in range(episodes):

    sumA = np.zeros(dim)
    sumB = np.zeros(dim)

    states = []
    actions = []
    rewards = []

    prev_diff = 0

    for i in range(n):

        state_np = build_state(i, sumA, sumB)
        state = torch.FloatTensor(state_np).to(device)

        logits, _ = net(state)
        probs = F.softmax(logits, dim=-1)

        action = torch.multinomial(probs, 1).item()

        states.append(state)
        actions.append(action)

        # take action
        if action == 0:
            sumA += items[i]
        else:
            sumB += items[i]

        # incremental reward (stable learning)
        new_diff = np.sum(np.abs(sumA - sumB))
        reward = prev_diff - new_diff
        prev_diff = new_diff

        rewards.append(reward)

        if len(states) == n_step or i == n-1:

            # -------- returns --------
            returns = []
            G = 0

            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns).to(device)

            actor_loss = 0
            critic_loss = 0
            entropies = []
            values = []

            for t in range(len(states)):

                logits, value = net(states[t])
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-8)

                entropy = -(probs * log_probs).sum()
                entropies.append(entropy)

                log_prob = log_probs[actions[t]]

                advantage = returns[t] - value.squeeze()

                actor_loss += -log_prob * advantage.detach()
                critic_loss += advantage.pow(2)

                values.append(value.squeeze())

            values = torch.stack(values)

            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            entropy_loss = torch.stack(entropies).mean()

            loss = actor_loss + 0.5 * critic_loss - entropy_beta * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clear buffers
            states, actions, rewards = [], [], []

    if ep % 100 == 0:
        print(f"Episode {ep} completed")

sumA = np.zeros(dim)
sumB = np.zeros(dim)
partition = []

for i in range(n):

    state = torch.FloatTensor(build_state(i, sumA, sumB)).to(device)
    logits, _ = net(state)

    action = torch.argmax(F.softmax(logits, dim=-1)).item()
    partition.append(action)

    if action == 0:
        sumA += items[i]
    else:
        sumB += items[i]

print("\nFinal Result")
print("Partition (0=A,1=B):")
print(partition)

print("\nFinal imbalance:")
print(np.abs(sumA - sumB))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F

items = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)

n, dim = items.shape
items = items / items.max(axis=0)

STATE_DIM = dim + 1      
ACTION_DIM = 2

gamma = 0.9
lr = 0.0005
workers = 4
episodes_per_worker = 500

def build_state(index, sumA, sumB):
    diff = sumA - sumB
    return np.concatenate(([index / n], diff))

class A3CNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)

        self.policy = nn.Linear(128, ACTION_DIM)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy(x)
        value = self.value(x)

        return policy_logits, value

def worker(worker_id, global_net, optimizer):

    local_net = A3CNet()
    local_net.load_state_dict(global_net.state_dict())

    for ep in range(episodes_per_worker):

        sumA = np.zeros(dim)
        sumB = np.zeros(dim)

        states = []
        actions = []
        rewards = []

        # --------- Generate Episode ----------
        for i in range(n):

            state_np = build_state(i, sumA, sumB)
            state = torch.FloatTensor(state_np)

            logits, _ = local_net(state)
            probs = F.softmax(logits, dim=-1)

            action = torch.multinomial(probs, 1).item()

            states.append(state)
            actions.append(action)

            if action == 0:
                sumA += items[i]
            else:
                sumB += items[i]

            reward = -np.sum(np.abs(sumA - sumB))
            rewards.append(reward)

        # --------- Compute Returns ----------
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # --------- Compute Loss ----------
        actor_loss = 0
        critic_loss = 0

        for t in range(n):

            logits, value = local_net(states[t])
            probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(probs[actions[t]])

            advantage = returns[t] - value.squeeze()

            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)

        loss = actor_loss + 0.5 * critic_loss

        # --------- Update Global Network ----------
        optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(
                local_net.parameters(),
                global_net.parameters()):
            global_param._grad = local_param.grad

        optimizer.step()

        # sync local
        local_net.load_state_dict(global_net.state_dict())

        if ep % 100 == 0:
            print(f"Worker {worker_id} Episode {ep}")

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    global_net = A3CNet()
    global_net.share_memory()

    optimizer = optim.Adam(global_net.parameters(), lr=lr)

    processes = []

    for wid in range(workers):
        p = mp.Process(target=worker,
                       args=(wid, global_net, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ==================================================
    # FINAL GREEDY POLICY
    # ==================================================
    sumA = np.zeros(dim)
    sumB = np.zeros(dim)
    partition = []

    for i in range(n):

        state = torch.FloatTensor(build_state(i, sumA, sumB))
        logits, _ = global_net(state)

        action = torch.argmax(F.softmax(logits, dim=-1)).item()
        partition.append(action)

        if action == 0:
            sumA += items[i]
        else:
            sumB += items[i]

    print("\nFinal imbalance:", np.abs(sumA - sumB))
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
scale=items.max(axis=0)
items = items / scale

STATE_DIM = dim + 1
ACTION_DIM = 2

gamma = 0.9
lr = 0.0005
workers = 4
episodes_per_worker = 200
entropy_beta = 0.01

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
        return self.policy(x), self.value(x)

def worker(worker_id, global_net, optimizer):

    local_net = A3CNet()
    local_net.load_state_dict(global_net.state_dict())

    for ep in range(episodes_per_worker):

        sumA = np.zeros(dim)
        sumB = np.zeros(dim)

        states = []
        actions = []
        rewards = []

        prev_diff = 0

        # ---------- Generate Episode ----------
        for i in range(n):

            state_np = build_state(i, sumA, sumB)
            state = torch.FloatTensor(state_np)

            logits, _ = local_net(state)
            probs = F.softmax(logits, dim=-1)

            action = torch.multinomial(probs, 1).item()

            states.append(state)
            actions.append(action)

            # apply action
            if action == 0:
                sumA += items[i]
            else:
                sumB += items[i]

            # dense shaped reward
            new_diff = np.max(np.abs(sumA - sumB))
            reward = prev_diff - new_diff
            prev_diff = new_diff

            rewards.append(reward)

        # ---------- Returns ----------
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # ---------- Loss ----------
        actor_loss = 0
        critic_loss = 0

        entropies = []
        values = []

        for t in range(n):

            logits, value = local_net(states[t])
            probs = F.softmax(logits, dim=-1)

            log_probs = torch.log(probs + 1e-8)
            log_prob = log_probs[actions[t]]

            entropy = -(probs * log_probs).sum()
            entropies.append(entropy)

            values.append(value.squeeze())

        values = torch.stack(values)

        # advantage normalization
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        for t in range(n):
            log_probs = torch.log(
                F.softmax(local_net(states[t])[0], dim=-1) + 1e-8
            )
            log_prob = log_probs[actions[t]]

            actor_loss += -log_prob * advantages[t]
            critic_loss += (returns[t] - values[t]).pow(2)

        entropy_loss = torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss - entropy_beta * entropy_loss

        # ---------- Update Global ----------
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (VERY IMPORTANT)
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)

        for local_param, global_param in zip(
                local_net.parameters(),
                global_net.parameters()):
            global_param._grad = local_param.grad

        optimizer.step()

        # sync local net
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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import math

# ── Data ─────────────────────────────────────────────────────────────────────
items_raw = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)

n_full, dim = items_raw.shape
scale       = items_raw.max(axis=0)
scale[scale == 0] = 1.0
items_norm  = items_raw / scale

order      = np.argsort(-np.max(np.abs(items_norm), axis=1))
items_full = items_norm[order]   

FIXED_IDX  = n_full - 1          # last item (after reorder) fixed into set A
N_DECIDE   = n_full - 1          # items 0 … N_DECIDE-1 are decided by the agent

STATE_DIM  = dim * 3 + 1
ACTION_DIM = 2

GAMMA        = 0.99           # increased from 0.9 — less myopic
GAE_LAMBDA   = 0.95           # GAE λ
LR           = 3e-4           # slightly higher than baseline
WORKERS      = 4
EP_PER_WORKER = 200           # fewer eps because each is more expensive

ENTROPY_BETA_MAX = 0.05
LEVY_ALPHA       = 0.5

LS_PROB      = 0.9
VND_TOP_K    = 3              # refine best-K episodes per RL-update cycle

STAGNATION_EPISODES = 50


# ── Helpers ───────────────────────────────────────────────────────────────────
def linf(sumA: np.ndarray, sumB: np.ndarray) -> float:
    return float(np.max(np.abs(sumA - sumB)))


def build_state(step: int, sumA: np.ndarray, sumB: np.ndarray) -> np.ndarray:
    """
    Richer state than baseline: concatenate fraction, difference, both sums.
    shape = dim*3 + 1
    """
    frac = step / N_DECIDE
    return np.concatenate(([frac], sumA - sumB, sumA, sumB)).astype(np.float32)


def levy_entropy_beta(episode: int) -> float:
    return ENTROPY_BETA_MAX / ((episode + 1) ** LEVY_ALPHA)


def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last: bool) -> float:
    """
    From Rainbow code: terminal reward = -linf; step reward = 0.1 * improvement.
    Keeps reward magnitudes stable across episodes regardless of dataset scale.
    """
    if is_last:
        return -linf(new_sumA, new_sumB)
    return 0.1 * (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB))


# ── Improved VND (full 2-swap, from Rainbow code) ────────────────────────────
def vnd(partition: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Variable Neighborhood Descent:
      N1  – best single-item move (O(n))
      N2  – full best 2-swap     (O(|A|·|B|))   ← replaces restricted L∞ swap
    Strictly dominates the baseline's N1.5 heuristic on solution quality.
    """
    part = partition.copy()
    sumA = items_full[part == 0].sum(axis=0)
    sumB = items_full[part == 1].sum(axis=0)
    best_obj = linf(sumA, sumB)

    improved = True
    while improved:
        improved = False

        # ── N1: best single-item move ─────────────────────────────────────
        best_delta, best_i = 0.0, None
        best_sA = best_sB = None

        for i in range(n_full):
            if i == FIXED_IDX:
                continue
            if part[i] == 0:
                sA, sB = sumA - items_full[i], sumB + items_full[i]
            else:
                sA, sB = sumA + items_full[i], sumB - items_full[i]
            delta = best_obj - linf(sA, sB)
            if delta > best_delta:
                best_delta, best_i = delta, i
                best_sA, best_sB   = sA, sB

        if best_i is not None:
            part[best_i] = 1 - part[best_i]
            sumA, sumB   = best_sA, best_sB
            best_obj    -= best_delta
            improved     = True
            continue

        # ── N2: full 2-swap ───────────────────────────────────────────────
        idx_A = [i for i in np.where(part == 0)[0] if i != FIXED_IDX]
        idx_B = [i for i in np.where(part == 1)[0] if i != FIXED_IDX]
        best_delta, best_swap = 0.0, None

        for v in idx_A:
            for w in idx_B:
                sA = sumA - items_full[v] + items_full[w]
                sB = sumB + items_full[v] - items_full[w]
                delta = best_obj - linf(sA, sB)
                if delta > best_delta:
                    best_delta = delta
                    best_swap  = (v, w, sA, sB)

        if best_swap is not None:
            v, w, sumA, sumB = best_swap
            part[v] = 1 - part[v]
            part[w] = 1 - part[w]
            best_obj -= best_delta
            improved  = True

    return part, best_obj


# ── Greedy warm-start (from Rainbow code) ────────────────────────────────────
def greedy_partition() -> tuple[np.ndarray, float]:
    """
    Assign each item to whichever set keeps the L∞ imbalance smallest.
    Run VND on top to get a solid starting point before any RL.
    """
    sumA = items_full[FIXED_IDX].copy()
    sumB = np.zeros(dim)
    part = np.ones(n_full, dtype=int)
    part[FIXED_IDX] = 0

    for i in range(N_DECIDE):
        sA0 = sumA + items_full[i]
        sA1 = sumA
        sB0 = sumB
        sB1 = sumB + items_full[i]
        if linf(sA0, sB0) <= linf(sA1, sB1):
            part[i] = 0; sumA = sA0
        else:
            part[i] = 1; sumB = sB1

    return part, linf(sumA, sumB)


# ── Improved Actor-Critic Network ─────────────────────────────────────────────
class A3CNet(nn.Module):
    """
    Deeper network with LayerNorm and orthogonal init.
    Shared trunk → separate policy head + value head.
    """
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, ACTION_DIM)
        self.value  = nn.Linear(128, 1)

        # Orthogonal init — standard for PPO/A3C
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy(h), self.value(h)


# ── GAE computation ───────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, last_value,
                gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Generalised Advantage Estimation (Schulman et al. 2016).
    Lower variance than plain MC returns, lower bias than 1-step TD.
    """
    advantages = []
    gae = 0.0
    values_ext = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ── Worker ────────────────────────────────────────────────────────────────────
def worker(worker_id: int,
           global_net: A3CNet,
           optimizer: optim.Optimizer,
           global_best: mp.Value,
           lock: mp.Lock):

    local_net = A3CNet()
    local_net.load_state_dict(global_net.state_dict())

    no_improve_count = 0

    for ep in range(EP_PER_WORKER):
        ent_beta = levy_entropy_beta(ep)

        # Re-sync with global net on stagnation
        if no_improve_count >= STAGNATION_EPISODES:
            local_net.load_state_dict(global_net.state_dict())
            no_improve_count = 0

        # ── Episode rollout ───────────────────────────────────────────────
        sumA = items_full[FIXED_IDX].copy()
        sumB = np.zeros(dim)

        states, actions, rewards, values_list, dones = [], [], [], [], []

        for i in range(N_DECIDE):
            state_np = build_state(i, sumA, sumB)
            state_t  = torch.FloatTensor(state_np)

            logits, val = local_net(state_t)
            probs  = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

            states.append(state_t)
            actions.append(action)
            values_list.append(val.squeeze().item())

            prev_sumA, prev_sumB = sumA.copy(), sumB.copy()
            if action == 0:
                sumA += items_full[i]
            else:
                sumB += items_full[i]

            is_last = (i == N_DECIDE - 1)
            reward  = shaped_reward(prev_sumA, prev_sumB, sumA, sumB, is_last)
            rewards.append(reward)
            dones.append(float(is_last))

        ep_obj = linf(sumA, sumB)

        # ── VND polish (with probability LS_PROB) ────────────────────────
        if np.random.rand() < LS_PROB:
            partition = np.ones(n_full, dtype=int)
            partition[FIXED_IDX] = 0
            for i, a in enumerate(actions):
                partition[i] = a

            refined_part, refined_obj = vnd(partition)

            with lock:
                if refined_obj < global_best.value:
                    global_best.value = refined_obj
                    no_improve_count  = 0
                else:
                    no_improve_count += 1

            # Behaviour cloning toward VND solution
            ref_actions = [int(refined_part[i]) for i in range(N_DECIDE)]
            bc_loss = sum(
                -F.log_softmax(local_net(states[t])[0], dim=-1)[ref_actions[t]]
                for t in range(N_DECIDE)
            ) / N_DECIDE

            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)
            for lp, gp in zip(local_net.parameters(), global_net.parameters()):
                gp._grad = lp.grad
            optimizer.step()
            local_net.load_state_dict(global_net.state_dict())
        else:
            with lock:
                if ep_obj < global_best.value:
                    global_best.value = ep_obj
                    no_improve_count  = 0
                else:
                    no_improve_count += 1

        # ── GAE advantage + A3C gradient ──────────────────────────────────
        # Bootstrap value at terminal is 0 (episode ended)
        advantages_list, returns_list = compute_gae(
            rewards, values_list, dones, last_value=0.0
        )

        advantages_t = torch.FloatTensor(advantages_list)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t    = torch.FloatTensor(returns_list)

        actor_loss  = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        entropy_sum = torch.tensor(0.0)

        for t in range(N_DECIDE):
            logits, val = local_net(states[t])
            probs       = F.softmax(logits, dim=-1)
            log_probs   = torch.log(probs + 1e-8)

            actor_loss  = actor_loss  + (-log_probs[actions[t]] * advantages_t[t])
            critic_loss = critic_loss + (returns_t[t] - val.squeeze()).pow(2)
            entropy_sum = entropy_sum + (-(probs * log_probs).sum())

        entropy_loss = entropy_sum / N_DECIDE
        loss = actor_loss + 0.5 * critic_loss - ent_beta * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
        optimizer.step()
        local_net.load_state_dict(global_net.state_dict())

        if ep % 50 == 0:
            print(f"Worker {worker_id}  ep {ep:4d}  "
                  f"ep_obj={ep_obj:.4f}  "
                  f"global_best={global_best.value:.4f}  "
                  f"ent_beta={ent_beta:.4f}")


# ── Greedy decode ─────────────────────────────────────────────────────────────
def greedy_decode(net: A3CNet) -> tuple[list, float]:
    net.eval()
    sumA = items_full[FIXED_IDX].copy()
    sumB = np.zeros(dim)

    partition = np.ones(n_full, dtype=int)
    partition[FIXED_IDX] = 0

    with torch.no_grad():
        for i in range(N_DECIDE):
            state   = torch.FloatTensor(build_state(i, sumA, sumB))
            logits, _ = net(state)
            action  = torch.argmax(F.softmax(logits, dim=-1)).item()
            partition[i] = action
            if action == 0:
                sumA += items_full[i]
            else:
                sumB += items_full[i]

    partition, obj = vnd(partition)
    return partition.tolist(), obj


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    g_part, g_obj = greedy_partition()
    g_part, g_obj = vnd(g_part)
    print(f"Greedy + VND baseline L∞ (norm): {g_obj:.4f}")

    # ── Shared global net and state ───────────────────────────────────────
    global_net = A3CNet()
    global_net.share_memory()
    optimizer  = optim.Adam(global_net.parameters(), lr=LR)

    global_best = mp.Value('d', g_obj)   # initialise from warm-start
    lock        = mp.Lock()

    # ── Launch workers ────────────────────────────────────────────────────
    processes = []
    for wid in range(WORKERS):
        p = mp.Process(
            target=worker,
            args=(wid, global_net, optimizer, global_best, lock)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ── Final decode + VND ────────────────────────────────────────────────
    print("\nRunning final greedy decode + VND polish...")
    partition, obj_norm = greedy_decode(global_net)

    # Remap to original item order
    original_partition = np.zeros(n_full, dtype=int)
    for reordered_idx, original_idx in enumerate(order):
        original_partition[original_idx] = partition[reordered_idx]

    sumA_orig = items_raw[original_partition == 0].sum(axis=0)
    sumB_orig = items_raw[original_partition == 1].sum(axis=0)
    diff      = np.abs(sumA_orig - sumB_orig)
    max_imb   = float(np.max(diff))

    print("\n── Final Result ──────────────────────────────────────")
    print(f"Partition (0=A, 1=B): {original_partition.tolist()}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {diff}")
    print(f"Max imbalance L∞ (original scale): {max_imb:.4f}")
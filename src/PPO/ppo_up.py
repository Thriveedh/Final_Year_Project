import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple
import math

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_path: str = r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt"

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 512, 256, 128])

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    entropy_start: float = 0.05
    entropy_end: float = 0.005
    n_iterations: int = 200
    n_episodes: int = 16
    ppo_epochs: int = 4
    mini_batch: int = 512
    max_grad_norm: float = 0.5

    # LR schedule: linear warmup then cosine decay
    warmup_iters: int = 10

    # VND
    vnd_top_k: int = 3
    vnd_n2_sample: int = 200   # cap on N2 pairs sampled per VND pass (avoids O(n^2))

    # Population restart: if best doesn't improve for this many iters, reset bottom half
    stagnation_limit: int = 30


CFG = Config()

# ── Data ─────────────────────────────────────────────────────────────────────
items_raw = np.loadtxt(CFG.data_path, skiprows=1)
n_full, dim = items_raw.shape
scale = items_raw.max(axis=0)
scale[scale == 0] = 1.0
items_norm = items_raw / scale

# Sort descending by L-inf norm so the hardest items are decided first
order = np.argsort(-np.max(np.abs(items_norm), axis=1))
inv_order = np.empty_like(order)
inv_order[order] = np.arange(n_full)   # reordered_idx -> original_idx lookup

items = items_norm[order]
FIXED_IDX = n_full - 1
N_DECIDE = n_full - 1

# FIX: state now includes item features; +dim for current item
STATE_DIM = dim * 4 + 1   # [frac, sumA-sumB (dim), sumA (dim), sumB (dim), item (dim)]
ACTION_DIM = 2


# ── Helpers ───────────────────────────────────────────────────────────────────
def linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def build_state(step: int, sumA: np.ndarray, sumB: np.ndarray) -> np.ndarray:
    frac = step / N_DECIDE
    item_feat = items[step]   # current item the agent must assign
    return np.concatenate(([frac], sumA - sumB, sumA, sumB, item_feat), dtype=np.float32)


# ── Improved VND (capped N2) ──────────────────────────────────────────────────
def vnd(partition: np.ndarray, n2_sample: int = CFG.vnd_n2_sample) -> Tuple[np.ndarray, float]:
    """
    Variable neighbourhood descent with:
      N1: best single-move (full scan, fast O(n))
      N2: best swap sampled from random pairs (capped to avoid O(n^2))
    """
    part = partition.copy()
    sumA = items[part == 0].sum(axis=0)
    sumB = items[part == 1].sum(axis=0)
    best_obj = linf(sumA, sumB)

    improved = True
    while improved:
        improved = False

        # N1: best single-item move
        best_delta, best_i = 0.0, None
        for i in range(n_full):
            if i == FIXED_IDX:
                continue
            if part[i] == 0:
                sA, sB = sumA - items[i], sumB + items[i]
            else:
                sA, sB = sumA + items[i], sumB - items[i]
            delta = best_obj - linf(sA, sB)
            if delta > best_delta:
                best_delta, best_i = delta, i
                best_sA, best_sB = sA, sB

        if best_i is not None:
            part[best_i] = 1 - part[best_i]
            sumA, sumB = best_sA, best_sB
            best_obj -= best_delta
            improved = True
            continue

        # N2: sampled swap (much faster than full O(n^2))
        idx_A = np.array([i for i in np.where(part == 0)[0] if i != FIXED_IDX])
        idx_B = np.array([i for i in np.where(part == 1)[0] if i != FIXED_IDX])
        if len(idx_A) == 0 or len(idx_B) == 0:
            break

        rng = np.random.default_rng()
        sample_A = idx_A[rng.choice(len(idx_A), min(n2_sample, len(idx_A)), replace=False)]
        sample_B = idx_B[rng.choice(len(idx_B), min(n2_sample, len(idx_B)), replace=False)]

        best_delta, best_swap = 0.0, None
        for v in sample_A:
            for w in sample_B:
                sA = sumA - items[v] + items[w]
                sB = sumB + items[v] - items[w]
                delta = best_obj - linf(sA, sB)
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (v, w, sA, sB)

        if best_swap is not None:
            v, w, sumA, sumB = best_swap
            part[v] = 1 - part[v]
            part[w] = 1 - part[w]
            best_obj -= best_delta
            improved = True

    return part, best_obj


def greedy_partition() -> Tuple[np.ndarray, float]:
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)
    part = np.ones(n_full, dtype=int)
    part[FIXED_IDX] = 0

    for i in range(N_DECIDE):
        sA0 = sumA + items[i]
        sA1 = sumA
        sB1 = sumB + items[i]
        if linf(sA0, sumB) <= linf(sA1, sB1):
            part[i] = 0
            sumA = sA0
        else:
            part[i] = 1
            sumB = sB1

    return part, linf(sumA, sumB)


# ── Running reward normalizer ─────────────────────────────────────────────────
class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var = x.var()
        n = len(x)
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


reward_rms = RunningMeanStd()


# ── Network ───────────────────────────────────────────────────────────────────
class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [STATE_DIM] + CFG.hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.LayerNorm(sizes[i + 1]),
                nn.ReLU(),
            ]
        # Drop LayerNorm on the last hidden layer (empirically helps value accuracy)
        self.shared = nn.Sequential(*layers)
        self.policy = nn.Linear(CFG.hidden_sizes[-1], ACTION_DIM)
        self.value = nn.Linear(CFG.hidden_sizes[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy(h), self.value(h)

    def get_action(self, state_np: np.ndarray):
        with torch.no_grad():
            state = torch.FloatTensor(state_np).unsqueeze(0)
            logits, value = self(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


# ── Reward ────────────────────────────────────────────────────────────────────
def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last: bool) -> float:
    if is_last:
        return -linf(new_sumA, new_sumB)
    return (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB)) * 0.1


# ── Episode ───────────────────────────────────────────────────────────────────
def run_episode(net: PPONet) -> dict:
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)
    states, actions, log_probs, values, rewards = [], [], [], [], []

    for i in range(N_DECIDE):
        state_np = build_state(i, sumA, sumB)
        action, lp, val = net.get_action(state_np)
        prev_sumA, prev_sumB = sumA.copy(), sumB.copy()

        states.append(state_np)
        actions.append(action)
        log_probs.append(lp)
        values.append(val)

        if action == 0:
            sumA += items[i]
        else:
            sumB += items[i]

        is_last = (i == N_DECIDE - 1)
        rewards.append(shaped_reward(prev_sumA, prev_sumB, sumA, sumB, is_last))

    partition = np.ones(n_full, dtype=int)
    partition[FIXED_IDX] = 0
    for i, a in enumerate(actions):
        partition[i] = a

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "partition": partition,
        "final_obj": linf(sumA, sumB),
    }


# ── GAE ───────────────────────────────────────────────────────────────────────
def compute_gae(rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + CFG.gamma * next_val - values[t]
        last_gae = delta + CFG.gamma * CFG.gae_lambda * last_gae
        adv[t] = last_gae
    return adv + values


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(net: PPONet, optimizer: optim.Optimizer, batch: dict, entropy_coef: float):
    states = torch.FloatTensor(batch["states"])
    actions = torch.LongTensor(batch["actions"])
    old_logp = torch.FloatTensor(batch["log_probs"])
    returns = torch.FloatTensor(batch["returns"])
    advs = torch.FloatTensor(batch["advantages"])
    old_values = torch.FloatTensor(batch["values"])   # for value clipping

    for _ in range(CFG.ppo_epochs):
        idx = torch.randperm(len(states))
        for start in range(0, len(states), CFG.mini_batch):
            mb = idx[start : start + CFG.mini_batch]
            log_prob, values, entropy = net.evaluate(states[mb], actions[mb])

            adv = advs[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = torch.exp(log_prob - old_logp[mb])
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CFG.clip_eps, 1 + CFG.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss (standard PPO improvement)
            v_clip = old_values[mb] + torch.clamp(values - old_values[mb], -CFG.clip_eps, CFG.clip_eps)
            critic_loss = torch.max(
                F.mse_loss(values, returns[mb]),
                F.mse_loss(v_clip, returns[mb]),
            )

            loss = actor_loss + CFG.vf_coef * critic_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CFG.max_grad_norm)
            optimizer.step()


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(iteration: int) -> float:
    if iteration < CFG.warmup_iters:
        return CFG.lr * (iteration + 1) / CFG.warmup_iters
    progress = (iteration - CFG.warmup_iters) / max(1, CFG.n_iterations - CFG.warmup_iters)
    return CFG.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Partition mapping helpers ─────────────────────────────────────────────────
def reordered_to_original(part_reordered: np.ndarray) -> np.ndarray:
    """Convert partition on reordered items -> partition on original item indices."""
    part_orig = np.empty(n_full, dtype=int)
    for reordered_idx in range(n_full):
        original_idx = order[reordered_idx]
        part_orig[original_idx] = part_reordered[reordered_idx]
    return part_orig


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    net = PPONet()
    optimizer = optim.Adam(net.parameters(), lr=CFG.lr, eps=1e-5)

    print("Computing greedy warm-start...")
    best_partition, best_obj_norm = greedy_partition()
    best_partition, best_obj_norm = vnd(best_partition)
    print(f"Greedy+VND warm-start L∞ (norm): {best_obj_norm:.4f}")

    stagnation_count = 0
    prev_best = best_obj_norm

    for iteration in range(CFG.n_iterations):
        # LR schedule
        new_lr = get_lr(iteration)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        frac = 1.0 - iteration / CFG.n_iterations
        entropy_coef = CFG.entropy_end + frac * (CFG.entropy_start - CFG.entropy_end)

        all_states, all_actions, all_logps = [], [], []
        all_returns, all_advantages, all_values = [], [], []
        episode_objs, candidates = [], []

        for _ in range(CFG.n_episodes):
            ep = run_episode(net)
            rets = compute_gae(ep["rewards"], ep["values"])
            advs = rets - ep["values"]

            all_states.extend(ep["states"])
            all_actions.extend(ep["actions"])
            all_logps.extend(ep["log_probs"])
            all_returns.extend(rets)
            all_advantages.extend(advs)
            all_values.extend(ep["values"])
            episode_objs.append(ep["final_obj"])
            candidates.append((ep["final_obj"], ep["partition"].copy()))

            if ep["final_obj"] < best_obj_norm:
                best_obj_norm = ep["final_obj"]
                best_partition = ep["partition"].copy()

        raw_returns = np.array(all_returns, dtype=np.float32)
        reward_rms.update(raw_returns)
        norm_returns = reward_rms.normalize(raw_returns)

        batch = {
            "states": np.array(all_states, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.int64),
            "log_probs": np.array(all_logps, dtype=np.float32),
            "returns": norm_returns,
            "advantages": np.array(all_advantages, dtype=np.float32),
            "values": np.array(all_values, dtype=np.float32),
        }

        ppo_update(net, optimizer, batch, entropy_coef)

        # VND on top-K candidates
        candidates.sort(key=lambda x: x[0])
        for obj, part in candidates[: CFG.vnd_top_k]:
            refined, refined_obj = vnd(part)
            if refined_obj < best_obj_norm:
                best_obj_norm = refined_obj
                best_partition = refined.copy()

        # Stagnation check + population restart
        if best_obj_norm < prev_best - 1e-6:
            stagnation_count = 0
            prev_best = best_obj_norm
        else:
            stagnation_count += 1

        if stagnation_count >= CFG.stagnation_limit:
            print(f"  [restart] Stagnated for {CFG.stagnation_limit} iters — re-initialising policy head")
            nn.init.orthogonal_(net.policy.weight, gain=0.01)
            nn.init.zeros_(net.policy.bias)
            stagnation_count = 0

        mean_ep_obj = np.mean(episode_objs)
        print(
            f"Iter {iteration+1:4d}/{CFG.n_iterations}  "
            f"mean_ep={mean_ep_obj:.4f}  "
            f"best_norm={best_obj_norm:.4f}  "
            f"ent={entropy_coef:.4f}  "
            f"lr={new_lr:.2e}"
        )

    return net, best_partition, best_obj_norm


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training PPO for MDTWNPP (improved)...")
    net, best_partition, best_obj_norm = train()

    print("\nRunning final VND polish (full N2 sweep)...")
    # Final pass: use full N2 (no sample cap) for maximum quality
    best_partition, best_obj_norm = vnd(best_partition, n2_sample=10_000)

    original_partition = reordered_to_original(best_partition)

    sumA_orig = items_raw[original_partition == 0].sum(axis=0)
    sumB_orig = items_raw[original_partition == 1].sum(axis=0)
    diff = np.abs(sumA_orig - sumB_orig)
    max_imb = float(np.max(diff))

    print(f"\nPartition (0=A, 1=B): {original_partition.tolist()}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {diff}")
    print(f"Max imbalance L∞ (original scale): {max_imb:.4f}")
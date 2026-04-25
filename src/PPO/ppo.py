import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── Data ─────────────────────────────────────────────────────────────────────
items_raw = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)
n_full, dim = items_raw.shape
scale       = items_raw.max(axis=0)
scale[scale == 0] = 1.0
items_norm  = items_raw / scale

order       = np.argsort(-np.max(np.abs(items_norm), axis=1))
items       = items_norm[order]
FIXED_IDX   = n_full - 1
N_DECIDE    = n_full - 1

# Hyperparameters
STATE_DIM       = dim * 3 + 1
ACTION_DIM      = 2

LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95          
CLIP_EPS        = 0.2
VF_COEF         = 0.5
ENTROPY_START   = 0.05
ENTROPY_END     = 0.005
N_ITERATIONS    = 200          
N_EPISODES      = 16            
PPO_EPOCHS      = 4
MINI_BATCH      = 512           
LS_PROB         = 1.0           
VND_TOP_K       = 3             


def linf(sumA, sumB):
    return float(np.max(np.abs(sumA - sumB)))

def build_state(step, sumA, sumB):
    frac = step / N_DECIDE
    return np.concatenate(([frac], sumA - sumB, sumA, sumB)).astype(np.float32)


def vnd(partition):
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
                best_sA, best_sB   = sA, sB

        if best_i is not None:
            part[best_i] = 1 - part[best_i]
            sumA, sumB   = best_sA, best_sB
            best_obj    -= best_delta
            improved     = True
            continue

        # ✅ N2: full 2-swap (not restricted to nearest neighbor)
        idx_A = [i for i in np.where(part == 0)[0] if i != FIXED_IDX]
        idx_B = [i for i in np.where(part == 1)[0] if i != FIXED_IDX]
        best_delta, best_swap = 0.0, None

        for v in idx_A:
            for w in idx_B:
                sA = sumA - items[v] + items[w]
                sB = sumB + items[v] - items[w]
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

def greedy_partition():
    """Greedy: assign each item to whichever set has smaller L∞ after adding."""
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)
    part = np.ones(n_full, dtype=int)
    part[FIXED_IDX] = 0

    for i in range(N_DECIDE):
        sA0 = sumA + items[i]; sB0 = sumB
        sA1 = sumA;             sB1 = sumB + items[i]
        if linf(sA0, sB0) <= linf(sA1, sB1):
            part[i] = 0; sumA = sA0
        else:
            part[i] = 1; sumB = sB1

    return part, linf(sumA, sumB)

class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, ACTION_DIM)
        self.value  = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)

    def get_action(self, state_np):
        with torch.no_grad():
            state  = torch.FloatTensor(state_np).unsqueeze(0)
            logits, value = self(state)
            probs  = F.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, states, actions):
        logits, values = self(states)
        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last):
    """
    Intermediate reward = improvement in L∞ at each step.
    Terminal reward = -final L∞ (scaled).
    """
    if is_last:
        return -linf(new_sumA, new_sumB)
    # Shaping: reward reduction in imbalance
    return (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB)) * 0.1


def run_episode(net):
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)

    states, actions, log_probs, values, rewards = [], [], [], [], []

    for i in range(N_DECIDE):
        state_np        = build_state(i, sumA, sumB)
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
        "states":    np.array(states,    dtype=np.float32),
        "actions":   np.array(actions,   dtype=np.int64),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "values":    np.array(values,    dtype=np.float32),
        "rewards":   np.array(rewards,   dtype=np.float32),
        "partition": partition,
        "final_obj": linf(sumA, sumB),
    }

# ── GAE returns 

def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized Advantage Estimation."""
    n       = len(rewards)
    adv     = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val  = values[t + 1] if t + 1 < n else 0.0
        delta     = rewards[t] + gamma * next_val - values[t]
        last_gae  = delta + gamma * lam * last_gae
        adv[t]    = last_gae
    returns = adv + values
    return returns

def ppo_update(net, optimizer, batch, entropy_coef):
    states   = torch.FloatTensor(batch["states"])
    actions  = torch.LongTensor(batch["actions"])
    old_logp = torch.FloatTensor(batch["log_probs"])
    returns  = torch.FloatTensor(batch["returns"])
    advs     = torch.FloatTensor(batch["advantages"])

    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(len(states))
        for start in range(0, len(states), MINI_BATCH):
            mb = idx[start: start + MINI_BATCH]

            log_prob, values, entropy = net.evaluate(states[mb], actions[mb])

            # Use pre-computed GAE advantages (whitened per mini-batch)
            adv = advs[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio      = torch.exp(log_prob - old_logp[mb])
            surr1      = ratio * adv
            surr2      = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            v_clipped   = returns[mb]
            critic_loss = F.mse_loss(values, v_clipped)

            loss = actor_loss + VF_COEF * critic_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

def train():
    net       = PPONet()
    optimizer = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    #  Warm-start with greedy solution
    print("Computing greedy warm-start...")
    best_partition, best_obj_norm = greedy_partition()
    best_partition, best_obj_norm = vnd(best_partition)
    print(f"Greedy+VND warm-start L∞ (norm): {best_obj_norm:.4f}")

    # Keep top-K partitions for VND
    top_partitions = [(best_obj_norm, best_partition.copy())]

    for iteration in range(N_ITERATIONS):
        frac         = 1.0 - iteration / N_ITERATIONS
        entropy_coef = ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END)

        all_states, all_actions, all_logps = [], [], []
        all_returns, all_advantages = [], []
        episode_objs = []
        candidates = []

        for _ in range(N_EPISODES):
            ep   = run_episode(net)
            rets = compute_gae(ep["rewards"], ep["values"])   # ✅ GAE
            advs = rets - ep["values"]

            all_states.extend(ep["states"])
            all_actions.extend(ep["actions"])
            all_logps.extend(ep["log_probs"])
            all_returns.extend(rets)
            all_advantages.extend(advs)
            episode_objs.append(ep["final_obj"])
            candidates.append((ep["final_obj"], ep["partition"].copy()))

            if ep["final_obj"] < best_obj_norm:
                best_obj_norm  = ep["final_obj"]
                best_partition = ep["partition"].copy()

        batch = {
            "states":     np.array(all_states,     dtype=np.float32),
            "actions":    np.array(all_actions,    dtype=np.int64),
            "log_probs":  np.array(all_logps,      dtype=np.float32),
            "returns":    np.array(all_returns,    dtype=np.float32),
            "advantages": np.array(all_advantages, dtype=np.float32),
        }

        ppo_update(net, optimizer, batch, entropy_coef)

        # VND on top-K candidates this iteration
        candidates.sort(key=lambda x: x[0])
        for obj, part in candidates[:VND_TOP_K]:
            refined, refined_obj = vnd(part)
            if refined_obj < best_obj_norm:
                best_obj_norm  = refined_obj
                best_partition = refined.copy()

        mean_ep_obj = np.mean(episode_objs)
        print(f"Iter {iteration+1:4d}/{N_ITERATIONS}  "
              f"mean_ep={mean_ep_obj:.4f}  "
              f"best_norm={best_obj_norm:.4f}  "
              f"ent={entropy_coef:.4f}")

    return net, best_partition, best_obj_norm

if __name__ == "__main__":
    print("Training PPO for MDTWNPP...")
    net, best_partition, best_obj_norm = train()

    print("\nRunning final VND polish...")
    best_partition, best_obj_norm = vnd(best_partition)

    original_partition = np.zeros(n_full, dtype=int)
    for reordered_idx, original_idx in enumerate(order):
        original_partition[original_idx] = best_partition[reordered_idx]

    sumA_orig = items_raw[original_partition == 0].sum(axis=0)
    sumB_orig = items_raw[original_partition == 1].sum(axis=0)
    diff      = np.abs(sumA_orig - sumB_orig)
    max_imb   = float(np.max(diff))

    print(f"Partition (0=A, 1=B): {original_partition.tolist()}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {diff}")
    print(f"Max imbalance L∞ (original scale): {max_imb:.4f}")
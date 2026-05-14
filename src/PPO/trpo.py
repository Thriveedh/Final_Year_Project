import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── Data ─────────────────────────────────────────────────────────────────────
items_raw = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20c.txt",
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

# ── Hyperparameters ───────────────────────────────────────────────────────────
STATE_DIM       = dim * 3 + 1
ACTION_DIM      = 2

GAMMA           = 0.99
GAE_LAMBDA      = 0.95
MAX_KL          = 0.01          # TRPO max KL divergence constraint
DAMPING         = 0.1           # conjugate gradient damping
CG_ITERS        = 10            # conjugate gradient iterations
BACKTRACK_ALPHA = 0.5           # line search backtrack factor
BACKTRACK_ITERS = 10            # max line search steps
VF_LR           = 3e-4          # value network learning rate
VF_ITERS        = 5             # value update steps per iteration
ENTROPY_START   = 0.05
ENTROPY_END     = 0.005
N_ITERATIONS    = 200
N_EPISODES      = 16
VND_TOP_K       = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

def linf(sumA, sumB):
    return float(np.max(np.abs(sumA - sumB)))

def build_state(step, sumA, sumB):
    frac = step / N_DECIDE
    return np.concatenate(([frac], sumA - sumB, sumA, sumB)).astype(np.float32)

# ── VND Local Search ──────────────────────────────────────────────────────────

def vnd(partition):
    part     = partition.copy()
    sumA     = items[part == 0].sum(axis=0)
    sumB     = items[part == 1].sum(axis=0)
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

        # N2: full 2-swap
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

# ── Greedy warm-start ─────────────────────────────────────────────────────────

def greedy_partition():
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

# ── Policy Network (actor only) ───────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Linear(128, ACTION_DIM),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)

    def get_probs(self, x):
        return F.softmax(self.forward(x), dim=-1)

    def get_action(self, state_np):
        with torch.no_grad():
            state  = torch.FloatTensor(state_np).unsqueeze(0)
            probs  = self.get_probs(state)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item()

# ── Value Network (critic, updated separately with Adam) ─────────────────────

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── Reward shaping ────────────────────────────────────────────────────────────

def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last):
    if is_last:
        return -linf(new_sumA, new_sumB)
    return (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB)) * 0.1

# ── Episode rollout ───────────────────────────────────────────────────────────

def run_episode(policy, value_net):
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)

    states, actions, log_probs, values, rewards = [], [], [], [], []

    for i in range(N_DECIDE):
        state_np       = build_state(i, sumA, sumB)
        action, lp     = policy.get_action(state_np)

        with torch.no_grad():
            val = value_net(torch.FloatTensor(state_np).unsqueeze(0)).item()

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

# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    n        = len(rewards)
    adv      = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val  = values[t + 1] if t + 1 < n else 0.0
        delta     = rewards[t] + gamma * next_val - values[t]
        last_gae  = delta + gamma * lam * last_gae
        adv[t]    = last_gae
    return adv + values, adv   # returns, advantages

# ── TRPO core utilities ───────────────────────────────────────────────────────

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx: idx + n].view(p.shape))
        idx += n

def flat_grad(loss, model, retain=False):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain)
    return torch.cat([g.contiguous().view(-1) for g in grads])

def kl_divergence(policy, old_probs, states):
    """Mean KL between old policy and current policy."""
    new_probs = policy.get_probs(states)
    kl = (old_probs * (old_probs.log() - new_probs.log())).sum(dim=-1)
    return kl.mean()

def hessian_vector_product(policy, old_probs, states, v, damping=DAMPING):
    """Computes (H + damping*I) @ v where H is the KL Hessian."""
    kl   = kl_divergence(policy, old_probs, states)
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_g = torch.cat([g.contiguous().view(-1) for g in grads])

    gv   = (flat_g * v.detach()).sum()
    hvp  = torch.autograd.grad(gv, policy.parameters())
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp])
    return flat_hvp.detach() + damping * v

def conjugate_gradient(policy, old_probs, states, g, n_iters=CG_ITERS):
    """Solve H x = g via conjugate gradient."""
    x  = torch.zeros_like(g)
    r  = g.clone()
    p  = g.clone()
    rr = torch.dot(r, r)

    for _ in range(n_iters):
        hvp   = hessian_vector_product(policy, old_probs, states, p)
        alpha = rr / (torch.dot(p, hvp) + 1e-8)
        x    += alpha * p
        r    -= alpha * hvp
        rr_new = torch.dot(r, r)
        p     = r + (rr_new / (rr + 1e-8)) * p
        rr    = rr_new
        if rr < 1e-10:
            break

    return x

def surrogate_loss(policy, old_log_probs, states, actions, advantages):
    probs    = policy.get_probs(states)
    dist     = torch.distributions.Categorical(probs)
    log_prob = dist.log_prob(actions)
    ratio    = torch.exp(log_prob - old_log_probs)
    return (ratio * advantages).mean()

# ── TRPO Update ───────────────────────────────────────────────────────────────

def trpo_update(policy, value_net, vf_optimizer, batch, entropy_coef):
    states     = torch.FloatTensor(batch["states"])
    actions    = torch.LongTensor(batch["actions"])
    old_logp   = torch.FloatTensor(batch["log_probs"])
    returns    = torch.FloatTensor(batch["returns"])
    advantages = torch.FloatTensor(batch["advantages"])

    # Whiten advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Value network update (standard Adam) ──────────────────────────────
    for _ in range(VF_ITERS):
        vf_optimizer.zero_grad()
        v_pred = value_net(states)
        vf_loss = F.mse_loss(v_pred, returns)
        vf_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        vf_optimizer.step()

    # ── Policy update via TRPO ────────────────────────────────────────────
    with torch.no_grad():
        old_probs = policy.get_probs(states)

    # Compute policy gradient
    loss_old = surrogate_loss(policy, old_logp, states, actions, advantages)

    # Add entropy bonus to encourage exploration
    probs   = policy.get_probs(states)
    entropy = -(probs * probs.log().clamp(min=-20)).sum(dim=-1).mean()
    loss_with_entropy = loss_old + entropy_coef * entropy

    g = flat_grad(loss_with_entropy, policy, retain=True)

    # Conjugate gradient to get natural gradient direction
    step_dir = conjugate_gradient(policy, old_probs.detach(), states, g.detach())

    # Compute max step size satisfying KL constraint
    sHs      = torch.dot(step_dir,
                    hessian_vector_product(policy, old_probs.detach(),
                                           states, step_dir))
    step_size = torch.sqrt(2 * MAX_KL / (sHs + 1e-8))
    full_step = step_size * step_dir

    # Line search (backtracking)
    old_params  = flat_params(policy)
    old_loss    = surrogate_loss(policy, old_logp, states, actions, advantages).item()
    success     = False

    for i in range(BACKTRACK_ITERS):
        new_params = old_params + (BACKTRACK_ALPHA ** i) * full_step
        set_params(policy, new_params)

        new_loss = surrogate_loss(policy, old_logp, states, actions, advantages).item()
        kl       = kl_divergence(policy, old_probs.detach(), states).item()

        if new_loss > old_loss and kl <= MAX_KL:
            success = True
            break
        set_params(policy, old_params)   # revert

    if not success:
        set_params(policy, old_params)   # revert fully if line search failed

# ── Main training loop ────────────────────────────────────────────────────────

def train():
    policy      = PolicyNet()
    value_net   = ValueNet()
    vf_optimizer = optim.Adam(value_net.parameters(), lr=VF_LR, eps=1e-5)

    # Greedy warm-start
    print("Computing greedy warm-start...")
    best_partition, best_obj_norm = greedy_partition()
    best_partition, best_obj_norm = vnd(best_partition)
    print(f"Greedy+VND warm-start L∞ (norm): {best_obj_norm:.4f}")

    for iteration in range(N_ITERATIONS):
        frac         = 1.0 - iteration / N_ITERATIONS
        entropy_coef = ENTROPY_END + frac * (ENTROPY_START - ENTROPY_END)

        all_states, all_actions, all_logps = [], [], []
        all_returns, all_advantages = [], []
        episode_objs = []
        candidates   = []

        for _ in range(N_EPISODES):
            ep             = run_episode(policy, value_net)
            rets, advs     = compute_gae(ep["rewards"], ep["values"])

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

        trpo_update(policy, value_net, vf_optimizer, batch, entropy_coef)

        # VND on top-K candidates
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

    return policy, best_partition, best_obj_norm

# ── Final decode and report ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training TRPO for MDTWNPP...")
    policy, best_partition, best_obj_norm = train()

    print("\nRunning final VND polish...")
    best_partition, best_obj_norm = vnd(best_partition)

    original_partition = np.zeros(n_full, dtype=int)
    for reordered_idx, original_idx in enumerate(order):
        original_partition[original_idx] = best_partition[reordered_idx]

    sumA_orig = items_raw[original_partition == 0].sum(axis=0)
    sumB_orig = items_raw[original_partition == 1].sum(axis=0)
    diff      = np.abs(sumA_orig - sumB_orig)
    max_imb   = float(np.max(diff))

    print("\n── Final Result ──────────────────────────────────────────")
    print(f"Partition (0=A, 1=B): {original_partition.tolist()}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {diff}")
    print(f"Max imbalance L∞ (original scale): {max_imb:.4f}")
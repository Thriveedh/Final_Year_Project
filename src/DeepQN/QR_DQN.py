import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math

# ── Data loading (identical to your Rainbow code) ─────────────────────────────
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

STATE_DIM   = dim * 3 + 1
ACTION_DIM  = 2

# ── QRDQN Hyperparameters ─────────────────────────────────────────────────────
LR           = 1e-4
GAMMA        = 0.99
N_ITERATIONS = 200
N_EPISODES   = 16
VND_TOP_K    = 3

N_QUANTILES  = 51          # replaces N_ATOMS — number of quantile heads
KAPPA        = 1.0         # Huber loss threshold for quantile regression

REPLAY_SIZE    = 50_000
REPLAY_INITIAL = 1_000
BATCH_SIZE     = 256
TARGET_UPDATE  = 200
N_STEP         = 5

ALPHA_PER  = 0.6
BETA_START = 0.4
BETA_END   = 1.0

SIGMA_INIT = 0.5

# ── Helpers (identical) ───────────────────────────────────────────────────────
def linf(sumA, sumB):
    return float(np.max(np.abs(sumA - sumB)))

def build_state(step, sumA, sumB):
    frac = step / N_DECIDE
    return np.concatenate(([frac], sumA - sumB, sumA, sumB)).astype(np.float32)

# ── VND local search (identical) ─────────────────────────────────────────────
def vnd(partition):
    part = partition.copy()
    sumA = items[part == 0].sum(axis=0)
    sumB = items[part == 1].sum(axis=0)
    best_obj = linf(sumA, sumB)

    improved = True
    while improved:
        improved = False

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

# ── NoisyLinear (identical) ───────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=SIGMA_INIT):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._f(torch.randn(self.in_features))
        eps_j = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

# ── KEY DIFFERENCE: QRDQN Network ─────────────────────────────────────────────
# Rainbow: outputs softmax probabilities over fixed atoms  → [B, A, N_ATOMS]
# QRDQN:  outputs raw quantile VALUES directly            → [B, A, N_QUANTILES]
# No support needed, no projection step, simpler forward pass

class QRDQNNet(nn.Module):
    """
    Quantile Regression DQN with:
    - Dueling architecture (Value + Advantage streams)
    - NoisyLinear layers for exploration
    - N_QUANTILES output heads per action (replaces C51 atoms)
    """
    def __init__(self):
        super().__init__()
        self.n_quantiles = N_QUANTILES

        # Fixed quantile midpoints τ = (2i-1)/2N  for i=1..N
        # These are the target quantile levels used in the loss
        taus = (2 * torch.arange(N_QUANTILES) + 1) / (2 * N_QUANTILES)
        self.register_buffer('taus', taus)   # shape [N_QUANTILES]

        # Shared feature extractor (same as your Rainbow code)
        self.feature = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Dueling: Value stream → [B, N_QUANTILES]
        self.value_hidden = NoisyLinear(256, 128)
        self.value_out    = NoisyLinear(128, N_QUANTILES)

        # Dueling: Advantage stream → [B, ACTION_DIM * N_QUANTILES]
        self.advantage_hidden = NoisyLinear(256, 128)
        self.advantage_out    = NoisyLinear(128, ACTION_DIM * N_QUANTILES)

        for m in self.feature:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns quantile values (not probabilities).
        Shape: [B, ACTION_DIM, N_QUANTILES]
        Each entry Z[b, a, i] = estimated τ_i-th quantile of return for action a
        """
        B = x.size(0)
        h = self.feature(x)

        # Value stream
        v = F.relu(self.value_hidden(h))
        v = self.value_out(v).view(B, 1, N_QUANTILES)             # [B, 1, N_Q]

        # Advantage stream
        a = F.relu(self.advantage_hidden(h))
        a = self.advantage_out(a).view(B, ACTION_DIM, N_QUANTILES) # [B, A, N_Q]

        # Dueling combination — same formula, applied per quantile
        q = v + a - a.mean(dim=1, keepdim=True)                   # [B, A, N_Q]
        return q  # raw quantile values, NO softmax

    def get_q_values(self, x):
        """
        Expected Q = mean over quantiles.
        Shape: [B, ACTION_DIM]
        """
        quantiles = self.forward(x)          # [B, A, N_Q]
        return quantiles.mean(dim=2)         # [B, A]

    def act(self, state_np):
        with torch.no_grad():
            x = torch.FloatTensor(state_np).unsqueeze(0)
            q = self.get_q_values(x)
        return int(q.argmax(dim=1).item())

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

# ── PER + NStepBuffer (identical to your Rainbow code) ───────────────────────
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data      = [None] * capacity
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):      return float(self.tree[0])
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx  = self._retrieve(0, s)
        didx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[didx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=ALPHA_PER):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.max_prio = 1.0
        self.capacity = capacity

    def push(self, *args):
        self.tree.add(self.max_prio ** self.alpha, Transition(*args))

    def sample(self, batch_size, beta):
        batch, idxs, weights = [], [], []
        segment  = self.tree.total() / batch_size
        min_prob = (self.tree.tree[-self.tree.capacity:].min() + 1e-8) / (self.tree.total() + 1e-8)
        max_weight = (min_prob * self.tree.n_entries) ** (-beta)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            prob   = prio / (self.tree.total() + 1e-8)
            weight = ((prob * self.tree.n_entries) ** (-beta)) / max_weight
            idxs.append(idx); weights.append(weight); batch.append(data)

        return batch, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs, priorities):
        for idx, prio in zip(idxs, priorities):
            prio = float(np.clip(prio, 1e-6, None))
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(idx, prio ** self.alpha)

    def __len__(self): return self.tree.n_entries

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n; self.gamma = gamma; self.buf = deque()

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) < self.n and not done:
            return None
        R = sum((self.gamma ** i) * r for i, (_, _, r, _, _) in enumerate(self.buf))
        s0, a0 = self.buf[0][0], self.buf[0][1]
        sN, dN = next_state, done
        self.buf.popleft()
        return Transition(s0, a0, R, sN, dN)

    def flush(self):
        transitions = []
        while self.buf:
            R   = sum((self.gamma ** i) * r for i, (_, _, r, _, _) in enumerate(self.buf))
            s0, a0 = self.buf[0][0], self.buf[0][1]
            sN, dN = self.buf[-1][3], self.buf[-1][4]
            transitions.append(Transition(s0, a0, R, sN, dN))
            self.buf.popleft()
        return transitions

# ── Reward shaping (identical) ────────────────────────────────────────────────
def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last):
    if is_last:
        return -linf(new_sumA, new_sumB)
    return (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB)) * 0.1

# ── Episode runner (identical structure) ──────────────────────────────────────
def run_episode_collect(online_net, replay_buffer, n_step_buf, total_steps):
    sumA = items[FIXED_IDX].copy()
    sumB = np.zeros(dim)
    actions_taken = []

    online_net.train()
    online_net.sample_noise()

    for i in range(N_DECIDE):
        state_np = build_state(i, sumA, sumB)
        action   = online_net.act(state_np)

        prev_sumA, prev_sumB = sumA.copy(), sumB.copy()
        if action == 0:
            sumA += items[i]
        else:
            sumB += items[i]
        actions_taken.append(action)

        is_last       = (i == N_DECIDE - 1)
        reward        = shaped_reward(prev_sumA, prev_sumB, sumA, sumB, is_last)
        done          = is_last
        next_state_np = build_state(i + 1, sumA, sumB) if not done else np.zeros(STATE_DIM, dtype=np.float32)

        trans = n_step_buf.push(state_np, action, reward, next_state_np, float(done))
        if trans is not None:
            replay_buffer.push(*trans)

    for trans in n_step_buf.flush():
        replay_buffer.push(*trans)

    partition = np.ones(n_full, dtype=int)
    partition[FIXED_IDX] = 0
    for i, a in enumerate(actions_taken):
        partition[i] = a

    return partition, linf(sumA, sumB)

# ── KEY DIFFERENCE: QRDQN Loss ────────────────────────────────────────────────
# Rainbow uses:  cross-entropy between projected C51 distribution and current
# QRDQN uses:    quantile Huber loss — no projection needed, simpler and faster

def quantile_huber_loss(online_quantiles, target_quantiles, taus, weights):
    """
    Computes the quantile regression (Huber) loss.

    Args:
        online_quantiles : [B, N_QUANTILES]  — current net's quantile estimates
        target_quantiles : [B, N_QUANTILES]  — target net's quantile estimates (detached)
        taus             : [N_QUANTILES]     — quantile levels τ_i
        weights          : [B]               — PER importance sampling weights

    How it works:
        For each pair (τ_i, z_j):
          u      = target_j - online_i          ← TD error
          rho(u) = |τ_i - 1(u < 0)| * HuberLoss(u, κ)
        Loss = mean over j, sum over i of rho
    """
    B  = online_quantiles.size(0)
    N  = N_QUANTILES

    # Expand for pairwise computation
    # online: [B, N, 1]  target: [B, 1, N]
    online = online_quantiles.unsqueeze(2)   # [B, N_Q, 1]
    target = target_quantiles.unsqueeze(1)   # [B, 1,   N_Q]

    # Pairwise TD errors: u[b, i, j] = target[b,j] - online[b,i]
    u = target - online                      # [B, N_Q, N_Q]

    # Huber loss element-wise
    huber = torch.where(
        u.abs() <= KAPPA,
        0.5 * u.pow(2),
        KAPPA * (u.abs() - 0.5 * KAPPA)
    )                                        # [B, N_Q, N_Q]

    # Asymmetric quantile weighting
    taus_expand = taus.view(1, N, 1)        # [1, N_Q, 1]
    rho = (taus_expand - (u < 0).float()).abs() * huber  # [B, N_Q, N_Q]

    # Mean over target quantiles j, sum over online quantiles i
    loss_per_sample = rho.mean(dim=2).sum(dim=1)         # [B]

    # PER weighting
    return (weights * loss_per_sample).mean(), loss_per_sample

def qrdqn_update(online_net, target_net, optimizer, replay_buffer, beta):
    if len(replay_buffer) < REPLAY_INITIAL:
        return 0.0

    batch, idxs, weights = replay_buffer.sample(BATCH_SIZE, beta)
    weights_t = torch.FloatTensor(weights)

    states      = torch.FloatTensor(np.array([t.state      for t in batch]))
    actions     = torch.LongTensor( np.array([t.action     for t in batch]))
    rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
    next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
    dones       = torch.FloatTensor(np.array([t.done       for t in batch]))

    gamma_n = GAMMA ** N_STEP

    # ── Double DQN action selection (same logic, simpler because no projection) 
    online_net.eval()
    online_net.sample_noise()
    with torch.no_grad():
        next_q_online  = online_net.get_q_values(next_states)  # [B, A]
        next_actions   = next_q_online.argmax(dim=1)           # [B]

    target_net.eval()
    with torch.no_grad():
        next_quantiles_all = target_net(next_states)           # [B, A, N_Q]
        next_quantiles     = next_quantiles_all[
            torch.arange(BATCH_SIZE), next_actions
        ]                                                      # [B, N_Q]

        # ── Target quantiles: r + γ^n * Z(s', a*)  (no projection needed!) ──
        target_quantiles = (
            rewards.unsqueeze(1)
            + gamma_n * (1 - dones.unsqueeze(1)) * next_quantiles
        ).detach()                                             # [B, N_Q]

    # ── Current quantile estimates for taken actions ──────────────────────────
    online_net.train()
    online_net.sample_noise()
    current_quantiles_all = online_net(states)                 # [B, A, N_Q]
    current_quantiles     = current_quantiles_all[
        torch.arange(BATCH_SIZE), actions
    ]                                                          # [B, N_Q]

    # ── Quantile Huber loss ───────────────────────────────────────────────────
    loss, loss_per_sample = quantile_huber_loss(
        current_quantiles, target_quantiles, online_net.taus, weights_t
    )

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()

    # Update PER priorities
    priorities = loss_per_sample.detach().abs().cpu().numpy() + 1e-6
    replay_buffer.update_priorities(idxs, priorities)

    return float(loss.item())

# ── Main training loop ────────────────────────────────────────────────────────
def train():
    online_net = QRDQNNet()
    target_net = QRDQNNet()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer     = optim.Adam(online_net.parameters(), lr=LR, eps=1.5e-4)
    replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    n_step_buf    = NStepBuffer(N_STEP, GAMMA)

    print("Computing greedy warm-start...")
    best_partition, best_obj_norm = greedy_partition()
    best_partition, best_obj_norm = vnd(best_partition)
    print(f"Greedy+VND warm-start L∞ (norm): {best_obj_norm:.4f}")

    total_steps = 0

    for iteration in range(N_ITERATIONS):
        beta = BETA_START + (BETA_END - BETA_START) * (iteration / N_ITERATIONS)

        episode_objs = []
        candidates   = []
        total_loss   = 0.0
        update_count = 0

        for _ in range(N_EPISODES):
            partition, final_obj = run_episode_collect(
                online_net, replay_buffer, n_step_buf, total_steps
            )
            total_steps += N_DECIDE
            episode_objs.append(final_obj)
            candidates.append((final_obj, partition.copy()))

            if final_obj < best_obj_norm:
                best_obj_norm  = final_obj
                best_partition = partition.copy()

            loss = qrdqn_update(online_net, target_net, optimizer, replay_buffer, beta)
            total_loss   += loss
            update_count += 1

        if (iteration + 1) % (TARGET_UPDATE // N_DECIDE + 1) == 0:
            target_net.load_state_dict(online_net.state_dict())

        candidates.sort(key=lambda x: x[0])
        for obj, part in candidates[:VND_TOP_K]:
            refined, refined_obj = vnd(part)
            if refined_obj < best_obj_norm:
                best_obj_norm  = refined_obj
                best_partition = refined.copy()

        mean_ep_obj = np.mean(episode_objs)
        mean_loss   = total_loss / max(update_count, 1)

        print(f"Iter {iteration+1:4d}/{N_ITERATIONS}  "
              f"mean_ep={mean_ep_obj:.4f}  "
              f"best_norm={best_obj_norm:.4f}  "
              f"loss={mean_loss:.5f}  "
              f"beta={beta:.3f}  "
              f"buf={len(replay_buffer)}")

    return online_net, best_partition, best_obj_norm

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training QRDQN for MDTWNPP...")
    print(f"  Components: QR (N_QUANTILES={N_QUANTILES}, κ={KAPPA}) + "
          f"Dueling + NoisyNets + PER + Double DQN + {N_STEP}-step returns")
    print(f"  Items: {n_full}, Dimensions: {dim}\n")

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

    print(f"\nPartition (0=A, 1=B): {original_partition.tolist()}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {diff}")
    print(f"Max imbalance L∞ (original scale): {max_imb:.4f}")
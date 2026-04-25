import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math

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

STATE_DIM       = dim * 3 + 1
ACTION_DIM      = 2

LR              = 1e-4
GAMMA           = 0.99
N_ITERATIONS    = 200
N_EPISODES      = 16
VND_TOP_K       = 3

N_ATOMS         = 51           
V_MIN           = -1.0         
V_MAX           = 0.0          
SUPPORT         = torch.linspace(V_MIN, V_MAX, N_ATOMS)   

REPLAY_SIZE     = 50_000       
REPLAY_INITIAL  = 1_000        
BATCH_SIZE      = 256
TARGET_UPDATE   = 200          
N_STEP          = 5            

ALPHA_PER       = 0.6          
BETA_START      = 0.4          
BETA_END        = 1.0

SIGMA_INIT      = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────
def linf(sumA, sumB):
    return float(np.max(np.abs(sumA - sumB)))

def build_state(step, sumA, sumB):
    frac = step / N_DECIDE
    return np.concatenate(([frac], sumA - sumB, sumA, sumB)).astype(np.float32)

# ── VND local search (unchanged from your PPO code) ──────────────────────────
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

class NoisyLinear(nn.Module):
    """Factorised Noisy Linear layer (Fortunato et al., 2017)."""
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

# ── Rainbow Component 2: Dueling + Distributional Network ────────────────────
class RainbowNet(nn.Module):
    """
    Combines:
    - Distributional RL (C51): outputs distribution over returns, not scalar
    - Dueling architecture:    separate Value and Advantage streams
    - Noisy nets:              NoisyLinear layers for exploration
    """
    def __init__(self):
        super().__init__()
        self.n_atoms = N_ATOMS
        self.support = SUPPORT

        # Shared feature extractor (standard linear — noise only in heads)
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

        # Dueling streams with NoisyLinear
        # Value stream: outputs [batch, n_atoms]
        self.value_hidden    = NoisyLinear(256, 128)
        self.value_out       = NoisyLinear(128, N_ATOMS)

        # Advantage stream: outputs [batch, action_dim, n_atoms]
        self.advantage_hidden = NoisyLinear(256, 128)
        self.advantage_out    = NoisyLinear(128, ACTION_DIM * N_ATOMS)

        # Orthogonal init for shared layers
        for m in self.feature:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns log-probabilities over atoms for each action. Shape: [B, A, N_ATOMS]"""
        B = x.size(0)
        h = self.feature(x)

        # Value stream
        v = F.relu(self.value_hidden(h))
        v = self.value_out(v).view(B, 1, N_ATOMS)          

        # Advantage stream
        a = F.relu(self.advantage_hidden(h))
        a = self.advantage_out(a).view(B, ACTION_DIM, N_ATOMS)  

        # Dueling combination: Q = V + A - mean(A)
        q = v + a - a.mean(dim=1, keepdim=True)             
        return F.log_softmax(q, dim=2)                     

    def get_q_values(self, x):
        """Expected Q-value = sum( p(atom) * atom_value ). Shape: [B, A]"""
        log_probs = self.forward(x)                          # [B, A, N_ATOMS]
        probs     = log_probs.exp()
        support   = self.support.to(x.device)
        return (probs * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [B, A]

    def act(self, state_np):
        """Greedy action from expected Q-values (noise sampled in forward)."""
        with torch.no_grad():
            x = torch.FloatTensor(state_np).unsqueeze(0)
            q = self.get_q_values(x)
        return int(q.argmax(dim=1).item())

    def sample_noise(self):
        """Re-sample factorised noise for all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

# ── Rainbow Component 3: Prioritized Experience Replay ───────────────────────
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """Binary sum-tree for O(log n) prioritized sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data     = [None] * capacity
        self.write    = 0
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

    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx  = self._retrieve(0, s)
        didx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[didx]

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (Schaul et al., 2016)."""
    def __init__(self, capacity, alpha=ALPHA_PER):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.max_prio = 1.0
        self.capacity = capacity

    def push(self, *args):
        self.tree.add(self.max_prio ** self.alpha, Transition(*args))

    def sample(self, batch_size, beta):
        batch, idxs, weights = [], [], []
        segment = self.tree.total() / batch_size
        min_prob = (self.tree.tree[-self.tree.capacity:].min() + 1e-8) / (self.tree.total() + 1e-8)
        max_weight = (min_prob * self.tree.n_entries) ** (-beta)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            prob   = prio / (self.tree.total() + 1e-8)
            weight = ((prob * self.tree.n_entries) ** (-beta)) / max_weight
            idxs.append(idx)
            weights.append(weight)
            batch.append(data)

        return batch, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs, priorities):
        for idx, prio in zip(idxs, priorities):
            prio = float(np.clip(prio, 1e-6, None))
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(idx, prio ** self.alpha)

    def __len__(self):
        return self.tree.n_entries

# ── Rainbow Component 4: Multi-Step Return Buffer ────────────────────────────
class NStepBuffer:
    """Accumulates N-step transitions before pushing to replay."""
    def __init__(self, n, gamma):
        self.n     = n
        self.gamma = gamma
        self.buf   = deque()

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) < self.n and not done:
            return None
        # Compute N-step return
        R = 0.0
        for i, (s, a, r, ns, d) in enumerate(self.buf):
            R += (self.gamma ** i) * r
        s0, a0 = self.buf[0][0], self.buf[0][1]
        sN, dN = next_state, done
        if not done:
            pass  # sN is already the correct next state
        self.buf.popleft()
        return Transition(s0, a0, R, sN, dN)

    def flush(self):
        """Drain remaining transitions at episode end."""
        transitions = []
        while self.buf:
            R = 0.0
            for i, (s, a, r, ns, d) in enumerate(self.buf):
                R += (self.gamma ** i) * r
            s0, a0 = self.buf[0][0], self.buf[0][1]
            sN, dN = self.buf[-1][3], self.buf[-1][4]
            transitions.append(Transition(s0, a0, R, sN, dN))
            self.buf.popleft()
        return transitions

# ── Distributional RL: Categorical projection ─────────────────────────────────
def project_distribution(next_log_probs, rewards, dones, gamma_n, support, n_atoms, v_min, v_max):
    """
    Project the target distribution onto the fixed support.
    (Bellemare et al., 2017 — C51 projection step)
    """
    B          = rewards.size(0)
    delta_z    = (v_max - v_min) / (n_atoms - 1)
    support    = support.to(rewards.device)

    # Target atoms: r + γ^n * z  (clipped to [V_MIN, V_MAX])
    target_z   = rewards.unsqueeze(1) + gamma_n * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    target_z   = target_z.clamp(v_min, v_max)             

    # Project onto fixed support
    b    = (target_z - v_min) / delta_z                   
    l, u = b.floor().long(), b.ceil().long()
    l    = l.clamp(0, n_atoms - 1)
    u    = u.clamp(0, n_atoms - 1)

    # Target probabilities (from online network, Double DQN style)
    next_probs = next_log_probs.exp()                       

    m = torch.zeros(B, n_atoms, device=rewards.device)
    for j in range(n_atoms):
        p = next_probs[:, j]                               
        m.scatter_add_(1, l[:, j:j+1], (p * (u[:, j].float() - b[:, j])).unsqueeze(1))
        m.scatter_add_(1, u[:, j:j+1], (p * (b[:, j] - l[:, j].float())).unsqueeze(1))

    return m                                              

# ── Reward shaping (same logic as your PPO code) ─────────────────────────────
def shaped_reward(prev_sumA, prev_sumB, new_sumA, new_sumB, is_last):
    if is_last:
        return -linf(new_sumA, new_sumB)
    return (linf(prev_sumA, prev_sumB) - linf(new_sumA, new_sumB)) * 0.1

# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode_collect(online_net, replay_buffer, n_step_buf, total_steps):
    """
    Run one full episode, push N-step transitions into replay buffer.
    Returns (partition, final_obj, steps_taken).
    """
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

        is_last = (i == N_DECIDE - 1)
        reward  = shaped_reward(prev_sumA, prev_sumB, sumA, sumB, is_last)
        done    = is_last

        next_state_np = build_state(i + 1, sumA, sumB) if not done else np.zeros(STATE_DIM, dtype=np.float32)

        trans = n_step_buf.push(state_np, action, reward, next_state_np, float(done))
        if trans is not None:
            replay_buffer.push(*trans)

    # Flush remaining n-step buffer
    for trans in n_step_buf.flush():
        replay_buffer.push(*trans)

    partition = np.ones(n_full, dtype=int)
    partition[FIXED_IDX] = 0
    for i, a in enumerate(actions_taken):
        partition[i] = a

    return partition, linf(sumA, sumB)

# ── Rainbow update step ───────────────────────────────────────────────────────
def rainbow_update(online_net, target_net, optimizer, replay_buffer, beta, total_steps):
    if len(replay_buffer) < REPLAY_INITIAL:
        return 0.0

    # Sample with PER
    batch, idxs, weights = replay_buffer.sample(BATCH_SIZE, beta)
    weights_t = torch.FloatTensor(weights)

    states      = torch.FloatTensor(np.array([t.state      for t in batch]))
    actions     = torch.LongTensor( np.array([t.action     for t in batch]))
    rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
    next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
    dones       = torch.FloatTensor(np.array([t.done       for t in batch]))

    gamma_n = GAMMA ** N_STEP

    # ── Double DQN action selection ───────────────────────────────────────────
    online_net.eval()
    online_net.sample_noise()
    with torch.no_grad():
        next_q_online = online_net.get_q_values(next_states)      
        next_actions  = next_q_online.argmax(dim=1)             

    # Use TARGET net to evaluate that action's distribution
    target_net.eval()
    with torch.no_grad():
        next_log_probs_all = target_net(next_states)              
        next_log_probs     = next_log_probs_all[
            torch.arange(BATCH_SIZE), next_actions
        ]                                                         

    # Project target distribution
    m = project_distribution(next_log_probs, rewards, dones,
                             gamma_n, SUPPORT, N_ATOMS, V_MIN, V_MAX)  

    # ── Current distribution ──────────────────────────────────────────────────
    online_net.train()
    online_net.sample_noise()
    log_probs_all  = online_net(states)                          
    log_probs      = log_probs_all[torch.arange(BATCH_SIZE), actions] 

    # Cross-entropy loss (distributional TD error)
    loss_per_sample = -(m * log_probs).sum(dim=1)              

    # Importance-sampling weights (PER correction)
    loss = (weights_t * loss_per_sample).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()

    # Update priorities with TD errors
    priorities = loss_per_sample.detach().abs().cpu().numpy() + 1e-6
    replay_buffer.update_priorities(idxs, priorities)

    return float(loss.item())

# ── Main training loop ────────────────────────────────────────────────────────
def train():
    online_net  = RainbowNet()
    target_net  = RainbowNet()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR, eps=1.5e-4)

    replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    n_step_buf    = NStepBuffer(N_STEP, GAMMA)

    # ── Greedy warm-start (same as PPO) ──────────────────────────────────────
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

            # One gradient update per episode
            loss = rainbow_update(online_net, target_net, optimizer,
                                  replay_buffer, beta, total_steps)
            total_loss   += loss
            update_count += 1

        # Hard target network update
        if (iteration + 1) % (TARGET_UPDATE // N_DECIDE + 1) == 0:
            target_net.load_state_dict(online_net.state_dict())

        # VND polish on top-K candidates
        candidates.sort(key=lambda x: x[0])
        for obj, part in candidates[:VND_TOP_K]:
            refined, refined_obj = vnd(part)
            if refined_obj < best_obj_norm:
                best_obj_norm  = refined_obj
                best_partition = refined.copy()

        mean_ep_obj = np.mean(episode_objs)
        mean_loss   = total_loss / max(update_count, 1)
        buf_size    = len(replay_buffer)

        print(f"Iter {iteration+1:4d}/{N_ITERATIONS}  "
              f"mean_ep={mean_ep_obj:.4f}  "
              f"best_norm={best_obj_norm:.4f}  "
              f"loss={mean_loss:.5f}  "
              f"beta={beta:.3f}  "
              f"buf={buf_size}")

    return online_net, best_partition, best_obj_norm

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training Rainbow DQN for MDTWNPP...")
    print(f"  Components: C51 (N_ATOMS={N_ATOMS}) + Dueling + NoisyNets + "
          f"PER + Double DQN + {N_STEP}-step returns")
    print(f"  Items: {n_full}, Dimensions: {dim}\n")

    net, best_partition, best_obj_norm = train()

    print("\nRunning final VND polish...")
    best_partition, best_obj_norm = vnd(best_partition)

    # Remap to original item order
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
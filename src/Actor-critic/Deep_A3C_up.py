import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
 
# ── Data ────────────────────────────────────────────────────────────────────
items_raw = np.loadtxt(
    r"C:\Users\Thriveedh\Downloads\mdtwnpp_500_20a.txt",
    skiprows=1
)
 
n_full, dim = items_raw.shape
scale = items_raw.max(axis=0)
scale[scale == 0] = 1.0           
items_full = items_raw / scale     
 

N_DECIDE  = n_full - 1            
FIXED_IDX = n_full - 1           
 
STATE_DIM    = dim + 1            
ACTION_DIM   = 2                  
GAMMA        = 0.9
LR           = 0.0005
WORKERS      = 4
EP_PER_WORKER = 1000
 

ENTROPY_BETA_MAX = 0.05
LEVY_ALPHA       = 0.5           

LS_PROB             = 0.9         
STAGNATION_EPISODES = 50          
 
 
def linf(a: np.ndarray, b: np.ndarray) -> float:
    """L∞ distance between two vectors — the paper's objective."""
    return float(np.max(np.abs(a - b)))
 
 
def mdtwnpp_obj(sumA: np.ndarray, sumB: np.ndarray) -> float:
    """MDTWNPP objective: L∞ norm of (sumA − sumB)."""
    return linf(sumA, sumB)
 
 
def build_state(step_idx: int, sumA: np.ndarray, sumB: np.ndarray) -> np.ndarray:
    """
    State vector: [normalised step index | per-dim difference].
    Using N_DECIDE (not n_full) so the fraction stays in [0,1].
    """
    diff = sumA - sumB
    return np.concatenate(([step_idx / N_DECIDE], diff))
 
 
def levy_entropy_beta(episode: int) -> float:
    """Power-law entropy coefficient — large early, small later."""
    return ENTROPY_BETA_MAX / ((episode + 1) ** LEVY_ALPHA)
 
 
 
def vnd_local_search(partition: np.ndarray,
                     items: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Variable Neighborhood Descent using N1 (single move) and N1.5
    (restricted 2-swap based on L∞ similarity).
 
    partition: shape (n_full,) with values 0 (A) or 1 (B).
    items:     normalised item matrix, shape (n_full, dim).
 
    Returns improved partition and its objective value.
    """
    part = partition.copy()
 
    sumA = items[part == 0].sum(axis=0)
    sumB = items[part == 1].sum(axis=0)
    best_obj = mdtwnpp_obj(sumA, sumB)
 
    improved = True
    while improved:
        improved = False
 
        # ── N1: best single-item move ────────────────────────────────────
        best_delta = 0.0
        best_move  = None
 
        for i in range(n_full):
            if i == FIXED_IDX:       # fixed item must stay in A
                continue
 
            if part[i] == 0:         # try moving i from A → B
                sA = sumA - items[i]
                sB = sumB + items[i]
            else:                    # try moving i from B → A
                sA = sumA + items[i]
                sB = sumB - items[i]
 
            new_obj = mdtwnpp_obj(sA, sB)
            delta   = best_obj - new_obj   # positive = improvement
            if delta > best_delta:
                best_delta = delta
                best_move  = (i, sA, sB, new_obj)
 
        if best_move is not None:
            i, sumA, sumB, best_obj = best_move
            part[i] = 1 - part[i]
            improved = True
            continue          # restart outer loop from N1
 
        # ── N1.5: restricted 2-swap ──────────────────────────────────────
        # Identify the larger subset.
        idx_A = np.where(part == 0)[0]
        idx_B = np.where(part == 1)[0]
        if len(idx_A) == 0 or len(idx_B) == 0:
            break
 
        if len(idx_A) >= len(idx_B):
            large_idx, small_idx = idx_A, idx_B
            large_set, small_set = 0, 1
        else:
            large_idx, small_idx = idx_B, idx_A
            large_set, small_set = 1, 0
 
        # For each item v in the large subset, find its L∞-closest
        # item w in the small subset, then evaluate the 2-swap.
        best_delta = 0.0
        best_swap  = None
 
        for v in large_idx:
            if v == FIXED_IDX:
                continue
            # L∞ distances from v to all items in the small subset
            dists = np.max(np.abs(items[small_idx] - items[v]), axis=1)
            w_rel = int(np.argmin(dists))
            w     = small_idx[w_rel]
 
            # Simulate the 2-swap: v moves from large→small, w from small→large
            if large_set == 0:      # v in A → B, w in B → A
                sA = sumA - items[v] + items[w]
                sB = sumB + items[v] - items[w]
            else:                   # v in B → A, w in A → B
                sA = sumA + items[v] - items[w]
                sB = sumB - items[v] + items[w]
 
            new_obj = mdtwnpp_obj(sA, sB)
            delta   = best_obj - new_obj
            if delta > best_delta:
                best_delta = delta
                best_swap  = (v, w, sA, sB, new_obj)
 
        if best_swap is not None:
            v, w, sumA, sumB, best_obj = best_swap
            part[v] = 1 - part[v]
            part[w] = 1 - part[w]
            improved = True
 
    return part, best_obj
 
 
# ── Neural network ──────────────────────────────────────────────────────────
 
class A3CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(STATE_DIM, 128)
        self.fc2    = nn.Linear(128, 128)
        self.policy = nn.Linear(128, ACTION_DIM)
        self.value  = nn.Linear(128, 1)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy(x), self.value(x)
 
 
# ── Worker ──────────────────────────────────────────────────────────────────
 
def worker(worker_id: int,
           global_net: A3CNet,
           optimizer: optim.Optimizer,
           global_best: mp.Value,
           stagnation_counter: mp.Value):
 
    local_net = A3CNet()
    local_net.load_state_dict(global_net.state_dict())
 
    last_best = float('inf')
    no_improve_count = 0
 
    for ep in range(EP_PER_WORKER):
 
        ent_beta = levy_entropy_beta(ep)
        if no_improve_count >= STAGNATION_EPISODES:
            local_net.load_state_dict(global_net.state_dict())
            no_improve_count = 0
 
        sumA = items_full[FIXED_IDX].copy()   
        sumB = np.zeros(dim)
 
        states  = []
        actions = []
        rewards = []
        prev_obj = mdtwnpp_obj(sumA, sumB)
 
        for i in range(N_DECIDE):             
            state_np = build_state(i, sumA, sumB)
            state    = torch.FloatTensor(state_np)
 
            logits, _ = local_net(state)
            probs     = F.softmax(logits, dim=-1)
            action    = torch.multinomial(probs, 1).item()
 
            states.append(state)
            actions.append(action)
 
            if action == 0:
                sumA += items_full[i]
            else:
                sumB += items_full[i]
 
            new_obj = mdtwnpp_obj(sumA, sumB)
            rewards.append(prev_obj - new_obj)
            prev_obj = new_obj

        if np.random.rand() < LS_PROB:
            partition = np.ones(n_full, dtype=int)  
            partition[FIXED_IDX] = 0                
            for i, a in enumerate(actions):
                partition[i] = a
 
            refined_part, refined_obj = vnd_local_search(partition, items_full)
 
            # Update global best
            with global_best.get_lock():
                if refined_obj < global_best.value:
                    global_best.value = refined_obj
                    no_improve_count  = 0
                else:
                    no_improve_count += 1
 
            ref_actions = [int(refined_part[i]) for i in range(N_DECIDE)]
            bonus_loss  = 0.0
            for t in range(N_DECIDE):
                logits, _ = local_net(states[t])
                log_probs  = F.log_softmax(logits, dim=-1)
                bonus_loss += -log_probs[ref_actions[t]]   # cross-entropy toward refined
 
            bonus_loss = bonus_loss / N_DECIDE
            optimizer.zero_grad()
            bonus_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)
            for lp, gp in zip(local_net.parameters(), global_net.parameters()):
                gp._grad = lp.grad
            optimizer.step()
            local_net.load_state_dict(global_net.state_dict())
        else:
            no_improve_count += 1

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        values    = []
        entropies = []
 
        for t in range(N_DECIDE):
            logits, value = local_net(states[t])
            probs         = F.softmax(logits, dim=-1)
            log_probs     = torch.log(probs + 1e-8)
            values.append(value.squeeze())
            entropies.append(-(probs * log_probs).sum())
 
        values    = torch.stack(values)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        actor_loss  = 0.0
        critic_loss = 0.0
        for t in range(N_DECIDE):
            logits, _ = local_net(states[t])
            lp         = F.log_softmax(logits, dim=-1)
            actor_loss  += -lp[actions[t]] * advantages[t]
            critic_loss += (returns[t] - values[t]).pow(2)
 
        entropy_loss = torch.stack(entropies).mean()
        loss = actor_loss + 0.5 * critic_loss - ent_beta * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)
 
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
 
        optimizer.step()
        local_net.load_state_dict(global_net.state_dict())
 
        if ep % 50 == 0:
            gb = global_best.value
            print(f"Worker {worker_id}  ep {ep:4d}  "
                  f"ent_beta={ent_beta:.4f}  global_best={gb:.4f}")
 

 
def greedy_decode(net: A3CNet) -> tuple[list, float]:
    """
    Use the trained policy to construct a partition greedily, then
    apply one round of VND to refine it.
    """
    sumA = items_full[FIXED_IDX].copy()
    sumB = np.zeros(dim)
 
    partition = np.ones(n_full, dtype=int)
    partition[FIXED_IDX] = 0
 
    for i in range(N_DECIDE):
        state  = torch.FloatTensor(build_state(i, sumA, sumB))
        logits, _ = net(state)
        action = torch.argmax(F.softmax(logits, dim=-1)).item()
        partition[i] = action
        if action == 0:
            sumA += items_full[i]
        else:
            sumB += items_full[i]
 
    # Final VND polish (LS_PROB = 1.0 at decode time)
    partition, obj = vnd_local_search(partition, items_full)
    return partition.tolist(), obj
 
 
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
 
    global_net = A3CNet()
    global_net.share_memory()
 
    optimizer = optim.Adam(global_net.parameters(), lr=LR)

    global_best       = mp.Value('d', float('inf'))
    stagnation_counter = mp.Value('i', 0)
 
    processes = []
    for wid in range(WORKERS):
        p = mp.Process(
            target=worker,
            args=(wid, global_net, optimizer, global_best, stagnation_counter)
        )
        p.start()
        processes.append(p)
 
    for p in processes:
        p.join()
 
    # ── Final decode ─────────────────────────────────────────────────────
    partition, obj_norm = greedy_decode(global_net)
 
    sumA_orig = np.zeros(dim)
    sumB_orig = np.zeros(dim)
    for i, a in enumerate(partition):
        if a == 0:
            sumA_orig += items_raw[i]
        else:
            sumB_orig += items_raw[i]
 
    final_diff     = np.abs(sumA_orig - sumB_orig)
    max_imbalance  = float(np.max(final_diff))
 
    print("\n── Final Result ──────────────────────────────────────")
    print(f"Partition (0=A, 1=B): {partition}")
    print(f"Sum A: {sumA_orig}")
    print(f"Sum B: {sumB_orig}")
    print(f"Imbalance per dimension: {final_diff}")
    print(f"Max imbalance (L∞, original scale): {max_imbalance:.4f}")
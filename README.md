# Reinforcement Learning for Multi-Dimensional Two-Way Number Partitioning Problem

## Overview

This project focuses on solving the **Multi-Dimensional Two-Way Number Partitioning Problem (MDTWNPP)** using **Reinforcement Learning (RL)** techniques.

MDTWNPP is an NP-Hard combinatorial optimization problem where a set of multi-dimensional vectors must be partitioned into two subsets such that the imbalance between them is minimized.

The project explores both traditional RL algorithms and advanced Deep Reinforcement Learning methods combined with heuristic optimization techniques like **Variable Neighbourhood Descent (VND)**.

---

## Problem Statement

Given a set of vectors:

\[
v_i = (v_{i1}, v_{i2}, ..., v_{id})
\]

The objective is to partition them into two subsets \(A\) and \(B\) such that:

\[
\|S_A - S_B\|_\infty
\]

is minimized.

Where:

- \(S_A\) = sum of vectors in subset A
- \(S_B\) = sum of vectors in subset B
- \(L_\infty\) norm represents the maximum imbalance across dimensions

---

## Motivation

The MDTWNPP problem has applications in:

- Load balancing
- Task scheduling
- Resource allocation
- Distributed systems
- Optimization problems

Traditional heuristic approaches are highly problem-specific. This project explores whether Reinforcement Learning can learn effective partitioning strategies automatically.

---

## Algorithms Implemented

### Basic RL Models

- SARSA
- Q-Learning
- Actor-Critic
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous Advantage Actor-Critic)
- Deep A2C
- Deep A3C

### Advanced RL Models

- PPO (Proximal Policy Optimization)
- TRPO (Trust Region Policy Optimization)
- Rainbow DQN
- QR-DQN (Quantile Regression DQN)

### Optimization & Heuristics

- Greedy Initialization
- Variable Neighbourhood Descent (VND)
  - 1-Flip Move
  - Swap Operations

---

## Key Improvements

The following improvements were introduced after the mid-review phase:

- Improved state representation
- Reward scaling for stability
- Greedy initialization
- VND local search
- Better exploration strategies
- Advanced Deep RL architectures

---

## State Representation

The state used by the RL agent is:

```python
State = [progress, (A - B), A, B]
```

Where:

- `progress` → fraction of processed items
- `(A - B)` → difference between subset sums
- `A`, `B` → cumulative subset vectors

---

## Reward Function

The reward is based on imbalance reduction:

\[
r_t = \|S_A^{t-1} - S_B^{t-1}\|_\infty - \|S_A^t - S_B^t\|_\infty
\]

Final reward:

\[
r_{final} = -\|S_A - S_B\|_\infty
\]

---

## Dataset

Benchmark datasets used:

- `mdtwnpp_500_20a.txt`
- `mdtwnpp_500_20b.txt`
- `mdtwnpp_500_20c.txt`
- `mdtwnpp_500_20d.txt`
- `mdtwnpp_500_20e.txt`

Dataset specifications:

- Number of vectors: 500
- Dimensions per vector: 20

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

Install dependencies

---

## Running the Models

### PPO

```bash
python PPO.py
```

### TRPO

```bash
python TRPO.py
```

### Rainbow DQN

```bash
python Rainbow_DQN.py
```

### QR-DQN

```bash
python QR_DQN.py
```

---

## Results

### Before Mid-Review

Basic RL models showed:

- High imbalance values
- Slow convergence
- Training instability

### After Mid-Review

Advanced RL models significantly improved performance.

| Model | Best Performance Trend |
|---|---|
| PPO | Stable and consistent |
| TRPO | Strong convergence |
| Rainbow DQN | Better exploration |
| QR-DQN | Improved uncertainty modeling |

### Key Observations

- PPO and TRPO performed best overall
- VND significantly improved solutions
- RL performance became closer to benchmark methods for larger dimensions
- Higher-dimensional datasets showed better learning capability

---

## Benchmark Comparison

The project compares RL models with benchmark optimization approaches:

- iMADEB
- MADEB
- GRASP + ePR

Although benchmark methods still achieve slightly better results, RL approaches demonstrated competitive performance and scalability.

---

## Future Improvements

Possible future enhancements:

- Better hyperparameter tuning
- More advanced neural architectures
- Transformer-based RL models
- Hybrid metaheuristic + RL frameworks
- Larger-scale dataset evaluation
- Distributed training

---

## Contributors

### Thundurthy Sai Manikhanta Thriveedh (CS22B1010)

Implemented:

- DQN
- Actor-Critic
- A3C
- Deep A3C
- PPO
- TRPO

Worked on:

- Training stability
- Gradient clipping
- Entropy regularization
- Parallel updates

### Kothamasu Naga Venkata Ganesh (CS22B1089)

Implemented:

- SARSA
- Q-Learning
- A2C
- Deep A2C
- Rainbow DQN
- QR-DQN

Worked on:

- Actor-Critic framework
- Performance evaluation

---

## References

1. Santucci et al., *An improved memetic algebraic differential evolution for solving the multidimensional two-way number partitioning problem*, Expert Systems with Applications, 2021.

2. Sutton & Barto, *Reinforcement Learning: An Introduction*, MIT Press.

3. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.

4. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.

5. Schulman et al., *Trust Region Policy Optimization*, 2015.

---



## License

This project is developed for academic and research purposes.

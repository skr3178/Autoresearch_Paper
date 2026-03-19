# Hyperparameters

Extracted from **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving**

All values from Section 4.1 (Implementation Details) unless otherwise noted.

---

## Architecture

| Symbol | Name | Value | Module | Section |
|--------|------|-------|--------|---------|
| $N_{\text{lon}}$ | Longitudinal modes | 12 | Mode Selector | 4.1 |
| $N_{\text{lat}}$ | Max lateral modes | 5 | Mode Selector | 4.1 |
| $N_{\text{mode}} = N_{\text{lat}} \times N_{\text{lon}}$ | Total candidate trajectories | 60 | Mode Selector / Rule Selector | 4.1 |
| $T$ | Trajectory horizon (steps) | 8 | All modules | Fig 4, 4.1 |
| $H$ | History length (steps) | (not stated explicitly; context window for ego + agents) | IVM | 3.3.3 |
| $D$ | Feature dimension | (not stated; inferred ~256 from transformer standard) | Transformer | 3.3 |
| $K$ | K-nearest neighbors in IVM | $N/2$ (half of map/agent elements) | IVM | 3.3.3 |
| $K_r$ | Route points per lateral mode | $N_r / 4$ | IVM | 3.3.3 |
| $N$ | Number of surrounding agents | (not stated; nuPlan standard ~20–80) | Transition Model | 3.2 |

---

## Training — General

| Symbol | Name | Value | Module | Section |
|--------|------|-------|--------|---------|
| — | Optimizer | AdamW | All | 4.1 |
| — | Initial learning rate | 1e-4 | All | 4.1 |
| — | LR schedule | ReduceLROnPlateau | All | 4.1 |
| — | LR patience | 0 (reduce immediately on plateau) | All | 4.1 |
| — | LR reduction factor | 0.3 | All | 4.1 |
| — | Epochs | 50 | All | 4.1 |
| — | Batch size (per GPU) | 64 | All | 4.1 |
| — | GPUs | 2 × NVIDIA RTX 3090 | — | 4.1 |
| — | Training scenarios | 176,218 | nuPlan train split | 4.1 |
| — | Validation scenarios | 1,118 (100 per type × 14 types) | nuPlan val14 | 4.1 |

---

## Training — Stage 1: Transition Model

| Symbol | Name | Value | Module | Section |
|--------|------|-------|--------|---------|
| — | Loss | L1 (MAE) on agent positions | Transition Model | Eq 5, 3.2 |
| — | Frozen after pre-training | Yes | Transition Model | 3.4 |

---

## Training — Stage 2: IL Pre-training

| Symbol | Name | Value | Module | Section |
|--------|------|-------|--------|---------|
| — | Mode selector loss | Cross-entropy | Mode Selector | Eq 6 |
| — | Side task loss | L1 on ego waypoints | Mode Selector | Eq 7 |
| — | Generator loss | L1 on ego waypoints | Trajectory Generator | Eq 11 |
| — | Total IL loss | $L_{\text{CE}} + L_{\text{SideTask}} + L_{\text{generator}}$ (equal weights) | All | Sec 3.4 |

---

## Training — Stage 3: RL Fine-tuning (PPO)

| Symbol | Name | Value | Module | Section |
|--------|------|-------|--------|---------|
| $\gamma$ | Discount factor | 0.1 | GAE / PPO | Eq 8, 4.1 |
| $\lambda$ | GAE parameter | 0.9 | GAE | GAE formula, 4.1 |
| $\epsilon$ | PPO clip ratio | (not stated; standard 0.2) | PPO | Eq 8 |
| $\lambda_{\text{policy}}$ | Policy loss weight | 100 | RL total loss | Eq 8, 4.1 |
| $\lambda_{\text{value}}$ | Value loss weight | 3 | RL total loss | Eq 9, 4.1 |
| $\lambda_{\text{entropy}}$ | Entropy bonus weight | 0.001 | RL total loss | Eq 10, 4.1 |
| — | Generator loss weight in RL | 1 (unscaled, same as IL) | RL total loss | Eq 11 |
| — | Backbone sharing in RL | No (policy and value heads separate) | RL | Table 4 |

---

## Rewards

| Reward Component | Description | Section |
|-----------------|-------------|---------|
| Displacement Error (DE) | L1 distance from ground-truth trajectory | 3.4 |
| Collision | Binary penalty for collisions with agents/obstacles | 3.4 |
| Drivable Area | Penalty for leaving drivable area | 3.4 |

---

## Inference

| Name | Value | Module | Section |
|------|-------|--------|---------|
| Candidates evaluated | 60 ($N_{\text{mode}}$) | Rule Selector | 3.3.4 |
| Sampling | Mean of Gaussian (deterministic, no sampling) | Trajectory Generator | 3.3.4 |
| Rule selector weights | Collision, drivable area, comfort, progress (weighted sum with mode selector scores) | Rule Selector | 3.3.4 |

---

## Ablation Best Configurations (Table 4)

| Config | Mode Dropout | Selector Side Task | Ego-history Dropout | Backbone Sharing | CLS-NR |
|--------|-------------|-------------------|--------------------|--------------------|--------|
| IL best | ✓ | ✓ | ✓ | ✓ | 93.41 |
| RL best | ✓ | ✓ | ✗ | ✗ | 94.07 |

---

## Training Horizon (Figure 4)

Default: $T = 8$ (both train and test). Training with $T=8$ achieves CLS-NR=94.07; shorter horizons degrade performance, especially $T=1$ (75.79).

---

## Notes on Ambiguous Hyperparameters

- **PPO clip $\epsilon$**: Not stated in paper. Standard value 0.2 — marked **Medium impact ambiguity** in `paper_contract.md`.
- **History length $H$**: Not explicitly stated. nuPlan standard is 2s × 10Hz = 20 frames, likely $H=20$ — marked **Medium impact**.
- **Feature dimension $D$**: Not stated. Infer from transformer practice (~256 or 512) — marked **Low impact** (shapes will reveal it).
- **Number of agents $N$**: Varies per scenario in nuPlan. Paper likely caps at a fixed number; standard is 32–80 — marked **Low impact**.

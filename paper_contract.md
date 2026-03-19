# Paper Contract

## Dataset Contract

**Name:** nuPlan
**Source:** To be downloaded (requires account at nuplan.org)
**Sample format:** 
- Type: Database logs
- Shape: T20-frame trajectory (T timesteps) × (T + T actions (T + action_dim)
- Dtype: float32
- Value range: Normalized coordinates
**Coordinate frame / units:** Ego-centric frame, units in meters
**Preprocessing:** Normalize coordinates, extract BE features
**Split sizes:** 
- Train: Boston, Pittsburgh, Singapore, Las Vegas (or mini: 72 logs)
- Val: Same cities (test set is also public)
- Test: Held out for competition

## Architecture Contract

| Module | Input shape | Output shape | Notes |
|--------|-------------|--------------|-------|
| Non-Reactive Transition Model | `(B, C, H, W)` | `(B, C, H, W)` | Previews future scene |
| Mode Selector | `(B, C, H, W)` | `(B,)` | Predicts mode `c` |
| Auto-regressive Policy | `(B, C, H, W)`, `(B,)` | `(B, T_future, A)`, `(B, T_future)` | Generates trajectory |
| Rule Selector | `(B, K, T_future, A)`, `(B, C, H, W)` | `(B, T_future, A)`, `(B,)` | Selects best trajectory |

## Training Contract

**Optimizer:** AdamW
**Learning rate schedule:** ReduceLROnPlateau, patience=0, factor=0.3
**Batch size:** 64 (per GPU)
**Steps / epochs:** 50 epochs

### Loss Mapping

| Paper symbol | Code variable | Semantic meaning | Value |
|--------------|---------------|------------------|-------|
| \( L_{\text{tm}} \) | transition_loss | Transition Model Loss | L1 Loss |
| \( L_{\text{CE}} \) | mode_selector_loss | Mode Selector Loss | Cross-Entropy |
| \( L_{\text{generator}} \) | generator_loss | Generator Loss | L1 Loss |
| \( L_{\text{RL}} \) | rl_loss | RL Total Loss | Weighted sum of policy, value, entropy, generator losses |

## Evaluation Contract

**Metric name:** CLS-NR (Closed-Loop Score Non-Reactive)
**Direction:** Higher is better
**Single-sample or best-of-K?** Best-of-K
**GT source:** Raw data
**Normalization:** None
**Thresholds / filtering:** None specified
**Sampling count / seed policy:** K candidates
**Expected value:** CLS-NR = 95.14
**Tolerance:** ~2-3× worse due to smaller scale and less data diversity

## Inference Contract

**Sampling procedure:** Mean of Gaussian (deterministic, no sampling)
**Number of steps:** T=8
**Schedule:** Must match training endpoints
**Post-processing:** Rule-based selection of best trajectory

## Ambiguity Register

| # | Paper section | Detail | Impact | Default chosen | Rationale | Alternative |
|---|---------------|--------|--------|----------------|-----------|-------------|
| 1 | PPO clip \(\epsilon\) | Not stated | Medium | 0.2 | Standard value | None |
| 2 | History length \(H\) | Not explicitly stated | Medium | 20 | nuPlan standard | None |
| 3 | Feature dimension \(D\) | Not stated | Low | 256 | Transformer practice | None |
| 4 | Number of agents \(N\) | Varies per scenario | Low | 32–80 | nuPlan standard | None |

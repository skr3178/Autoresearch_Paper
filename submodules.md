# CarPlanner Submodules — Build Order

Implement submodules in strict order. **Never start submodule N+1 until submodule N's gate is fully met.**

After each submodule gate passes:
1. Mark the checkbox `✅` in `progress.md`

---

## Submodule 1: `data_loader`

**Paper section**: Dataset / Data Processing
**Description**: Reads nuPlan SQLite scenario files and produces batched Bird's-Eye View (BEV) tensors ready for model input.

**Input → Output**:
- Input: nuPlan SQLite `.db` files (scenario paths from `requirements.md`)
- Output: batched dict with keys:
  - `bev` — `(B, C, H, W)` float32 BEV raster (agent boxes, map layers, ego history)
  - `ego_history` — `(B, T_hist, 3)` float32 (x, y, heading) in ego-centric frame
  - `gt_trajectory` — `(B, T_future, 3)` float32 ground-truth future (x, y, heading)
  - `scenario_id` — list[str]

**Files to create**:
- `implementation/data_loader.py` — `NuPlanDataset`, `collate_fn`, `build_dataloader`
- `implementation/test_data_loader.py`

**Verification gate**:
- [ ] Shape test: `bev` shape matches paper BEV spec (channels, resolution)
- [ ] Dtype test: all tensors float32, no NaN
- [ ] Unit range test: BEV pixel values in [0, 1]
- [ ] Coordinate test: ego position at t=0 is at BEV center within 1 pixel
- [ ] Overfit test: dataloader returns identical batch for same scenario_id
- [ ] Throughput: 10 batches (B=2) load in <5s

---

## Submodule 2: `transition_model`

**Paper section**: World Model / Non-Reactive Simulation
**Description**: Predicts future scene states given the current BEV and a proposed ego action. Used for imagination rollouts during RL training. Frozen after pre-training.

**Input → Output**:
- Input: `bev` `(B, C, H, W)`, `action` `(B, A)` (ego control or waypoint)
- Output: `next_bev` `(B, C, H, W)`, `done` `(B,)` bool

**Files to create**:
- `implementation/transition_model.py` — `TransitionModel(nn.Module)`
- `implementation/pretrain_transition.py` — pre-training script (supervised on logged rollouts)
- `implementation/test_transition_model.py`

**Verification gate**:
- [ ] Shape test: `next_bev` same shape as `bev`
- [ ] Forward + backward at debug config, no NaN
- [ ] Equation oracle: manually verify at least one BEV update equation
- [ ] Overfit test: can memorize 2 transitions (loss < 0.01)
- [ ] Diversity test: different actions → different `next_bev` (cosine sim < 0.95)

---

## Submodule 3: `mode_selector`

**Paper section**: Mode / Modality Prediction
**Description**: Imitation-learning module that predicts a discrete trajectory modality `c` from the current observation. Trained with supervised IL before RL phase.

**Input → Output**:
- Input: `bev` `(B, C, H, W)`, `ego_history` `(B, T_hist, 3)`
- Output: `c` `(B,)` int64 (modality index), `c_logits` `(B, K)` float32

**Files to create**:
- `implementation/mode_selector.py` — `ModeSelector(nn.Module)`
- `implementation/train_mode_selector.py` — IL training script
- `implementation/test_mode_selector.py`

**Verification gate**:
- [ ] Shape test: `c_logits` shape `(B, K)` where K matches paper
- [ ] Forward + backward, no NaN
- [ ] IL overfit: 100% train accuracy on 4 samples within 200 steps
- [ ] Diversity: at least 2 distinct modes predicted on a batch of 8 diverse scenarios
- [ ] Calibration sanity: softmax probabilities sum to 1.0

---

## Submodule 4: `autoregressive_policy`

**Paper section**: Iterative Vision-based Motion planning (IVM) / Autoregressive Decoding
**Description**: Core module. Given the current observation and selected mode `c`, autoregressively generates a future trajectory action sequence. This is the primary RL policy.

**Input → Output**:
- Input: `bev` `(B, C, H, W)`, `ego_history` `(B, T_hist, 3)`, `c` `(B,)`
- Output: `actions` `(B, T_future, A)` float32, `log_probs` `(B, T_future)` float32

**Files to create**:
- `implementation/autoregressive_policy.py` — `AutoregressivePolicy(nn.Module)`
- `implementation/test_autoregressive_policy.py`

**Verification gate**:
- [ ] Shape test: `actions` and `log_probs` shapes match paper spec
- [ ] Forward + backward (with `log_probs.sum().backward()`), no NaN
- [ ] Equation oracle: verify log-prob computation for 1 step manually
- [ ] Overfit test: can fit 2 fixed trajectories (MSE < 0.01 after 500 steps)
- [ ] Autoregressive test: output at step t depends on step t-1 (perturb t-1 → different t)
- [ ] `c` conditioning: same obs, different `c` → meaningfully different trajectories (MSE > 0.1)

---

## Submodule 5: `consistency_module`

**Paper section**: Temporal Consistency / Coherence
**Description**: Enforces temporal coherence across the autoregressive trajectory. Penalizes physically implausible transitions (large jerk, heading jumps). Applied as an auxiliary loss during RL training.

**Input → Output**:
- Input: `actions` `(B, T_future, A)` float32
- Output: `consistency_loss` `(B,)` float32 (per-sample penalty)

**Files to create**:
- `implementation/consistency_module.py` — `ConsistencyModule(nn.Module)`
- `implementation/test_consistency_module.py`

**Verification gate**:
- [ ] Shape test: output `(B,)` scalar per sample
- [ ] Sign test: smooth trajectory → lower penalty than jittery trajectory
- [ ] Equation oracle: hand-compute penalty for a 3-step trajectory, assert match
- [ ] Gradient test: `consistency_loss.mean().backward()` flows back to `actions`
- [ ] Zero test: perfectly smooth (constant velocity) trajectory → penalty ≈ 0

---

## Submodule 6: `critic`

**Paper section**: Value Function / Advantage Estimation
**Description**: Estimates the expected return V(s) for advantage computation in PPO. Trained jointly with the policy but with separate parameters.

**Input → Output**:
- Input: `bev` `(B, C, H, W)`, `ego_history` `(B, T_hist, 3)`, `c` `(B,)`
- Output: `value` `(B,)` float32

**Files to create**:
- `implementation/critic.py` — `Critic(nn.Module)`
- `implementation/test_critic.py`

**Verification gate**:
- [ ] Shape test: output `(B,)` scalar per sample
- [ ] Forward + backward, no NaN
- [ ] Overfit test: can memorize 2 fixed values (MSE < 0.001 after 200 steps)
- [ ] Independence: critic parameters do not share gradients with policy (verify via `id()` check)
- [ ] `c` conditioning: different `c` → different values for same obs

---

## Submodule 7: `ppo_loop`

**Paper section**: RL Training / PPO
**Description**: Orchestrates one PPO update step: rollout collection via `transition_model`, advantage estimation via `critic`, clipped policy gradient loss + value loss + entropy bonus + consistency loss.

**Input → Output**:
- Input: batch of initial observations from `data_loader`
- Output: dict with `policy_loss`, `value_loss`, `entropy`, `consistency_loss`, `total_loss` — all scalar tensors

**Files to create**:
- `implementation/ppo_loop.py` — `PPOTrainer` class with `collect_rollout()` and `update()` methods
- `implementation/test_ppo_loop.py`

**Verification gate**:
- [ ] Shape test: all loss outputs are scalar (0-dim tensors)
- [ ] Sign test: entropy term has correct sign (should be subtracted from loss)
- [ ] Clipping test: policy ratio `r_t` is clipped to `[1-ε, 1+ε]`; verify with a synthetic ratio=2.0
- [ ] Backward test: `total_loss.backward()` populates gradients for policy and critic parameters
- [ ] Overfit test: 20 PPO steps on 2 fixed scenarios → policy loss decreases monotonically
- [ ] Loss decomposition: `total_loss ≈ policy_loss + value_coef * value_loss - entropy_coef * entropy + consistency_coef * consistency_loss` (verify coefficients from paper)

---

## Submodule 8: `expert_refinement`

**Paper section**: Expert / IL Refinement
**Description**: Optional IL fine-tuning step applied on top of the RL-trained policy. Distills expert demonstrations into the policy via behavior cloning loss on a curated subset.

**Input → Output**:
- Input: `bev`, `ego_history`, `c`, `gt_trajectory` (from `data_loader`)
- Output: `bc_loss` scalar float32 (negative log-likelihood of expert actions under policy)

**Files to create**:
- `implementation/expert_refinement.py` — `ExpertRefinement` class
- `implementation/test_expert_refinement.py`

**Verification gate**:
- [ ] Shape test: `bc_loss` is scalar
- [ ] Equation oracle: manually verify NLL for 1 step (tiny policy, known log_probs)
- [ ] Overfit test: BC loss → 0 when `gt_trajectory` is also the mode of the policy distribution
- [ ] Gradient test: `bc_loss.backward()` flows to policy parameters only (not critic)
- [ ] Integration test: run one IL refinement step after one PPO step without errors

---

## Submodule 9: `rule_selector`

**Paper section**: Rule-Based / Inference Selection
**Description**: Inference-only module. Given K candidate trajectories sampled from the policy, selects the one that best satisfies traffic rules (collision-free, on-road, comfort bounds).

**Input → Output**:
- Input: `candidate_trajectories` `(B, K, T_future, A)`, `bev` `(B, C, H, W)`
- Output: `selected_trajectory` `(B, T_future, A)`, `selected_idx` `(B,)` int64

**Files to create**:
- `implementation/rule_selector.py` — `RuleSelector` class (pure Python, no learned params)
- `implementation/test_rule_selector.py`

**Verification gate**:
- [ ] Shape test: `selected_trajectory` shape `(B, T_future, A)`
- [ ] Determinism: same input → same output (no randomness in selector)
- [ ] Collision preference: when one candidate collides and one doesn't, selector picks collision-free
- [ ] Comfort preference: among collision-free candidates, selector prefers lower jerk
- [ ] Fallback test: if all candidates violate rules, selector still returns a trajectory (argmin score)

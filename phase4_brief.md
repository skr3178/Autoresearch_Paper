# Phase 4 Brief — Integration Proof

## 1) Data flow diagram (end-to-end)

**Batch from `data_loader`** (`implementation/data_loader.py`):
- `bev`: `(B, C=7, H=224, W=224)` float32 → consumed by `ModeSelector`, `AutoregressivePolicy`, `Critic`
- `ego_history`: `(B, T_hist=5, 3)` float32 → consumed by `ModeSelector`, `AutoregressivePolicy`, `Critic`
- `gt_trajectory`: `(B, T_future=8, 3)` float32 → (not used in PPO integration proof; used later for IL / metric)
- `scenario_id`: `list[str]` → logging only

**Mode selection** (`implementation/mode_selector.py`):
- Inputs: `bev`, `ego_history`
- Outputs: `ModeSelectorOutput.c_logits`: `(B, K_modes=6)` float32, `ModeSelectorOutput.c`: `(B,)` int64
- In integration: `c = ms_out.c` is passed to policy/critic.

**Policy rollout** (`implementation/autoregressive_policy.py` via `implementation/ppo_loop.py`):
- Inputs: `bev`, `ego_history`, `c`
- Outputs (rollout dict):
  - `actions`: `(B, T_future=8, A=3)` float32
  - `log_probs`: `(B, T_future=8)` float32

**Critic** (`implementation/critic.py` via `implementation/ppo_loop.py`):
- Inputs: `bev`, `ego_history`, `c`
- Output: `value`: `(B,)` float32

**Consistency loss** (`implementation/consistency_module.py` via `implementation/ppo_loop.py`):
- Input: `actions` `(B, T_future, A)`
- Output: `consistency_loss_per_sample`: `(B,)` float32 → reduced to scalar mean in PPO loss.

**PPO update** (`implementation/ppo_loop.py`):
- Consumes rollout + critic values + consistency penalty
- Produces scalar losses (0-dim tensors): `policy_loss`, `value_loss`, `entropy`, `consistency_loss`, `total_loss`

## 2) Loss assembly

In this repo’s PPO integration proof, the total loss is assembled inside `PPOTrainer.update()` as:

- `total_loss = policy_loss + λ_value * value_loss - λ_entropy * entropy + λ_consistency * consistency_loss`

Coefficients are provided by config in `implementation/train.py`:
- `λ_value = cfg['ppo']['value_coef']`
- `λ_entropy = cfg['ppo']['entropy_coef']` (subtracted)
- `λ_consistency = cfg['ppo']['consistency_coef']`

(Exact paper equation numbers for PPO/consistency are referenced in Phase 3 artifacts; Phase 4 focuses on wiring and scalar loss decomposition correctness.)

## 3) Training stages

Integration proof uses a **single-stage** loop:
- Mode selector is used in `no_grad()` to provide `c` (not trained here).
- Policy + critic are trained jointly with PPO-style losses.
- Transition model is instantiated but not used in the current PPOTrainer (kept for later imagination rollouts).

## 4) Figures referenced

Full-system figure (paper Figure 2 per program.md) is the reference for end-to-end data flow:
- Observation → mode selection → autoregressive generation → consistency constraint → RL update.

(Any arrows involving non-reactive world model / generation-selection are not exercised in this minimal integration proof runner.)

## 5) Integration risks

Most likely mismatch points:
1. **Mode selector output type**: returns a dataclass (`ModeSelectorOutput`), not a tuple; integration must access `.c` / `.c_logits`.
2. **Batch collation**: `scenario_id` and `agent_boxes` are lists; overfit batching must concatenate tensors but extend lists.
3. **Shape agreement**: `T_future` and `action_dim` must match between policy and consistency module.
4. **Determinism**: cuDNN nondeterminism can break reproducibility; integration sets deterministic flags.

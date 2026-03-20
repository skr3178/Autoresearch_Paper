# Phase 4 Brief — Integration Proof

## 1) Data flow diagram (end-to-end)

Batch from `implementation/data_loader.py` (Submodule 1):
- `bev`: (B, C=7, H=224, W=224) float32 → consumed by ModeSelector, Policy, Critic
- `ego_history`: (B, T_hist=5, 3) float32 → consumed by ModeSelector, Policy, Critic
- `gt_trajectory`: (B, T_future=8, 3) float32 → used only in Phase-4 shaped reward (integration harness)
- `scenario_id`: list[str]

Forward path:
1. `ModeSelector(bev, ego_history)` →
   - `c_logits`: (B, K=6) float32
   - `c`: (B,) int64
   Consumed by Policy and Critic.

2. `AutoregressivePolicy(bev, ego_history, c)` →
   - `actions`: (B, T_future=8, A=3) float32
   - `log_probs`: (B, T_future=8) float32
   Consumed by PPO loss and ConsistencyModule.

3. `Critic(bev, ego_history, c)` →
   - `value`: (B,) float32
   Used for PPO value loss / advantage estimation.

4. `ConsistencyModule(actions)` →
   - `consistency_loss`: (B,) float32
   Included in total loss with coefficient `lambda_consistency` (set to 0.0 for Phase-4 stability).

5. `PPOTrainer.update(rollout)` → scalar losses:
   - `policy_loss`: ()
   - `value_loss`: ()
   - `entropy`: ()
   - `consistency_loss`: ()
   - `total_loss`: ()

## 2) Loss assembly

Implemented in `implementation/ppo_loop.py`:
- PPO clipped policy loss: uses ratio r_t = exp(logp_new - logp_old) and clip to [1-ε, 1+ε]
- Value loss: MSE(V(s), return)
- Entropy bonus: subtracted from total loss
- Consistency loss: added

Total loss (integration):
`total_loss = λ_policy * L_policy + λ_value * L_value - λ_entropy * H + λ_consistency * L_consistency`

(Equation numbers referenced in code comments: Eq (8)-(10) placeholders; exact paper equation IDs to be aligned in Phase 5 once metric reproduction begins.)

## 3) Training stages

For Phase 4 only (integration proof):
- ModeSelector is used in eval mode (frozen) to provide `c`.
- Policy + Critic are trained jointly with PPO losses.
- Transition model and expert refinement are not used.

## 4) Figures referenced

Full system figure (paper Figure 2) is expected to show the overall pipeline: observation encoder → mode selection → autoregressive generation → selection/refinement → RL training loop.
Phase 4 implements the core forward path and PPO loss wiring; transition model imagination and rule selector are not yet integrated here.

## 5) Integration risks

- Shape mismatches between `gt_trajectory` and `actions` (both must be (B,T_future,3)).
- `c` dtype must be int64 for embedding/conditioning.
- Loss scale instability: consistency penalty can dominate; entropy sign/magnitude must be consistent.

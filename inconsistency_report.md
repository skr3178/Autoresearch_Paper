# Inconsistency Report

Date: 2026-03-20
Scope: Phase 4.5 Integration Audit & Cross-Phase Fix Loop

## Equation Mismatches

| Eq # | Paper formula (summary) | Code location | Status | Issue |
|------|--------------------------|---------------|--------|-------|
| Eq 1 | RL objective max E[sum gamma^t R] | implementation/ppo_loop.py (reward shaping stub) | PARTIAL | Integration harness uses supervised proxy reward from `gt_trajectory`; not true environment reward. Acceptable for Phase 4, but not paper-faithful. |
| Eq 3 | Factorization with policy π and transition model P_τ | implementation/ppo_loop.py:collect_rollout() | MISMATCH | Transition model is not used; rollouts are single-step on logged data. Paper uses non-reactive transition model for imagination rollouts. |
| Eq 4 | Mode marginalization with P(c|s0) and π(a|s,c) | implementation/train.py / ppo_loop.py | PARTIAL | ModeSelector provides c, but no integration over c (no sampling / expectation). Likely OK if paper uses sampled mode. Need confirm. |
| Eq 5 | Transition model L_tm = (1/T) Σ_t Σ_n ||s_t^n - s_t^{n,gt}||_1 | implementation/pretrain_transition.py | UNKNOWN | Not audited yet (need code mapping). |
| Eq 6 | CrossEntropyLoss(σ, c*) = -Σ I(c_i=c*) log σ_i | implementation/train_mode_selector.py | UNKNOWN | Not audited yet (need code mapping). |
| Eq 7 | SideTaskLoss = (1/T) Σ_t ||s̄_t^0 - s_t^{0,gt}||_1 | implementation/train_mode_selector.py / expert_refinement.py | UNKNOWN | Not audited yet (need code mapping). |
| Eq 8 | PPO policy loss with clipped ratio r_t | implementation/ppo_loop.py:update() | UNKNOWN | Needs line-by-line check; Phase 4 report notes entropy sign inconsistency and value_loss=0. |
| Eq 9 | Value loss = (1/T) Σ ||V_new - R_hat||^2 | implementation/ppo_loop.py:update() | MISMATCH | `value_loss` printed as 0.0 in Phase 4 single-step; likely not computed or returns are zero. |
| Eq 10 | Entropy = (1/T) Σ H(d_t,new) | implementation/ppo_loop.py:update() | MISMATCH | Phase 4 report: entropy sign differs between prints; likely sign/logging confusion or entropy computed on wrong distribution. |
| Eq 11 | Generator loss = (1/T) Σ ||s_t^0 - s_t^{0,gt}||_1 | implementation/ppo_loop.py / expert_refinement.py | PARTIAL | Generator loss not clearly included in RL total loss in Phase 4 harness (paper includes + L_generator). |

## Loss Term Audit

| Loss term | Paper expectation | Actual value (Phase4 debug single-step) | Coefficient (paper) | Status |
|-----------|------------------|-----------------------------------------|---------------------|--------|
| policy_loss (Eq 8) | non-zero; scaled by λ_policy=100 | 0.354 | 100 | MISMATCH — coefficient likely not applied (total_loss ~ policy_loss). |
| value_loss (Eq 9) | non-zero; scaled by λ_value=3 | 0.0 | 3 | BROKEN — value loss not contributing. |
| entropy (Eq 10) | positive entropy; subtracted with λ_entropy=0.001 | -3.743 (already negative) | 0.001 | MISMATCH — entropy sign/logging inconsistent. |
| consistency_loss (Sec 3.2) | auxiliary, should be >0 and weighted | 182.98 | (paper nonzero) | DISABLED — lambda_consistency=0.0 in Phase 4. |
| generator_loss (Eq 11) | included in RL total loss (+1×) | not logged | 1 | MISSING — not part of total_loss in Phase 4. |

## Disabled/Bypassed Components

| Component | Why disabled/bypassed | Paper says |
|-----------|------------------------|------------|
| ConsistencyModule loss | `lambda_consistency=0.0` for integration stability | Consistency constraints are a core contribution (Sec 3.2) and should be active in RL training. |
| TransitionModel imagination rollouts | not used in PPO loop | Paper factorizes dynamics with P_τ and uses non-reactive transition model for rollouts (Sec 3.1/4). |
| Generator loss in RL | not included | Paper RL loss includes + L_generator (Eq 11) in addition to PPO terms. |

## Data Scale Issues

Not yet re-checked in Phase 4.5. Need to print min/max/mean for:
- bev
- ego_history
- gt_trajectory
- actions

## Prioritized Fix List

1. **Critical**: PPO coefficients not applied (λ_policy=100, λ_value=3, λ_entropy=0.001) → total_loss scale mismatch.
2. **Critical**: value_loss is 0.0 → critic not learning / returns not computed.
3. **High**: entropy sign inconsistency → wrong optimization direction possible.
4. **High**: generator_loss missing from RL objective.
5. **Medium**: consistency_loss disabled (large magnitude suggests needs normalization/weighting).
6. **Low (for now)**: transition model rollouts not integrated.

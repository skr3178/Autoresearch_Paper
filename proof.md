# Proof of Correctness

This document records verification evidence per submodule, including tests and figure-to-code mapping.

---

## Submodule 2: transition_model

### What was implemented
- `implementation/transition_model.py`
  - `TransitionModel`: CNN encoder/decoder with FiLM conditioning on `action`.
  - `FiLM`: applies per-channel affine modulation `y = x * gamma(a) + beta(a)`.
  - `TransitionModelLoss`: L1 loss between predicted and target BEV.
- `implementation/pretrain_transition.py`
  - Minimal supervised pretraining loop on nuPlan batches (next-step BEV prediction).

### Assumptions
- The paper’s NR transition model is used to preview world evolution; exact architecture is not specified in the extracted artifacts. Implemented a minimal, stable CNN with action conditioning via FiLM.
- `done` is modeled as a sigmoid probability (float32 in [0,1]) since nuPlan logs do not provide terminal flags for short horizons in our simplified setup.

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_transition_model.py`

Passed gates:
- Shape test: `next_bev` matches `bev` shape; `done` is `(B,)`.
- Forward + backward: gradients finite for all parameters.
- Equation oracle: FiLM hand-check `y = x * gamma + beta` with controlled weights matches exactly.
- Overfit: memorizes 2 transitions (teacher-generated targets) to L1 < 0.01.
- Diversity: different actions change `next_bev` (mean abs diff > 1e-4).

### Figure Verification
**Figure classification**
- figure1_page1.png: architecture (conceptual pipeline; introduces consistent latent/mode `c`)
- figure2_page3.png: architecture (full CarPlanner block diagram; includes NR Transition Model)
- figure3_page8.png: results (qualitative comparisons + metric numbers)

**Forward mapping (figure → code)**
- Figure 2, component "NR Transition Model" → implementation/transition_model.py:TransitionModel

**Reverse mapping (code → figure)**
- implementation/transition_model.py:TransitionModel → Figure 2, component "NR Transition Model"

---

## Submodule 3: mode_selector

### What was implemented
- `implementation/mode_selector.py`
  - `ModeSelector`: BEV CNN encoder + ego-history MLP encoder → fused MLP → logits over K modes.

### Figure Verification
**Figure classification**
- figure2_page3.png: architecture (contains Mode Selector block)
- figure3_page8.png: results (metrics/qualitative)

**Forward mapping (figure → code)**
- Figure 2, component "Mode Selector" → implementation/mode_selector.py:ModeSelector

**Reverse mapping (code → figure)**
- implementation/mode_selector.py:ModeSelector → Figure 2, component "Mode Selector"

---

## Submodule 4: autoregressive_policy

### What was implemented
- `implementation/autoregressive_policy.py`
  - `AutoregressivePolicy`: autoregressive GRU decoder producing Gaussian actions and log-probs.
  - `diag_gaussian_log_prob`: diagonal Gaussian log-prob.

### Figure Verification
**Figure classification**
- figure1_page1.png: architecture (auto-regressive generation concept)
- figure2_page3.png: architecture (trajectory generator/policy block)
- figure3_page8.png: results (performance plots)

**Forward mapping (figure → code)**
- Figure 2, component "Auto-regressive Policy" → implementation/autoregressive_policy.py:AutoregressivePolicy

**Reverse mapping (code → figure)**
- implementation/autoregressive_policy.py:AutoregressivePolicy → Figure 2, component "Auto-regressive Policy"

---

## Submodule 5: consistency_module

### What was implemented
- `implementation/consistency_module.py`
  - `ConsistencyModule`: auxiliary temporal-coherence penalty computed from action jerk (xy) and heading acceleration.

### Assumptions
- The extracted artifacts do not provide an explicit closed-form for the consistency penalty. Implemented a standard comfort prior consistent with the paper’s stated goal (penalize physically implausible transitions):
  - jerk penalty on xy action components
  - heading acceleration penalty on heading component

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_consistency_module.py`

Passed gates:
- Shape test: output is `(B,)`.
- Sign test: smooth trajectory has lower penalty than jittery.
- Equation oracle: hand-computed jerk/heading-acc penalty for a 3-step toy trajectory matches.
- Gradient test: `loss.mean().backward()` produces non-zero gradients w.r.t. `actions`.
- Zero test: constant-velocity trajectory yields penalty ≈ 0.

### Figure Verification
**Figure classification**
- figure1_page1.png: architecture (mentions “consistency constraints” conceptually)
- figure2_page3.png: architecture (contains a “Consistency”/constraint block in the training loop)
- figure3_page8.png: results

**Forward mapping (figure → code)**
- Figure 2, component "Consistency constraints/module" → implementation/consistency_module.py:ConsistencyModule

**Reverse mapping (code → figure)**
- implementation/consistency_module.py:ConsistencyModule → Figure 2, component "Consistency constraints/module"

---

## Submodule 6: critic

### What was implemented
- `implementation/critic.py`
  - `Critic`: BEV CNN encoder + ego-history MLP + mode embedding → MLP value head.

### Assumptions
- The paper does not specify the critic architecture in detail in the extracted artifacts. Implemented a minimal value network that matches the policy’s observation inputs and supports `c` conditioning.

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_critic.py`

Passed gates:
- Shape test: output `(B,)`.
- Forward + backward: finite gradients.
- Overfit: memorizes 2 fixed scalar values (MSE < 0.001).
- Independence: critic parameters are distinct objects from policy parameters (no shared `id()`), and gradients are populated only for critic params when backpropagating critic loss.
- `c` conditioning: same obs with different `c` yields different values.

### Figure Verification
**Figure classification**
- figure2_page3.png: architecture (PPO loop includes a value function / critic)
- figure3_page8.png: results

**Forward mapping (figure → code)**
- Figure 2, component "Critic / Value function" → implementation/critic.py:Critic

**Reverse mapping (code → figure)**
- implementation/critic.py:Critic → Figure 2, component "Critic / Value function"

---

## Submodule 7: ppo_loop

### What was implemented
- `implementation/ppo_loop.py`
  - `compute_gae`: generalized advantage estimation.
  - `PPOTrainer.collect_rollout`: samples actions/log-probs from policy and gets critic value.
  - `PPOTrainer.update`: PPO clipped policy loss + value loss + entropy bonus + optional consistency loss.

### Assumptions
- For unit testing, rewards are provided externally (synthetic). Full integration with the transition model/environment is deferred to Phase 4.
- Critic is used as a state-value baseline; for simplicity in this minimal PPO, value targets are computed as mean return over the horizon.

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_ppo_loop.py`

Passed gates:
- Shape test: all reported losses are scalar 0-dim tensors.
- Sign test: entropy is subtracted from total loss (increasing entropy decreases total loss).
- Clipping test: synthetic ratio=2.0 is clipped to 1+ε.
- Backward test: `total_loss.backward()` populates gradients for both policy and critic parameters.
- Overfit test: 20 PPO steps on 2 fixed scenarios decreases policy loss monotonically.
- Loss decomposition: verified `total_loss ≈ λπ*policy_loss + λV*value_loss - λH*entropy + λC*consistency_loss` with configured coefficients.

### Figure Verification
**Figure classification**
- figure2_page3.png: architecture (training loop with PPO, critic, consistency)
- figure4_page13.png: results (training curves / comparisons)

**Forward mapping (figure → code)**
- Figure 2, component "PPO / RL training loop" → implementation/ppo_loop.py:PPOTrainer
- Figure 2, component "GAE / advantage estimation" → implementation/ppo_loop.py:compute_gae

**Reverse mapping (code → figure)**
- implementation/ppo_loop.py:PPOTrainer → Figure 2, component "PPO / RL training loop"
- implementation/ppo_loop.py:compute_gae → Figure 2, component "GAE / advantage estimation"

---

## Submodule 8: expert_refinement

### What was implemented
- `implementation/expert_refinement.py`
  - `ExpertRefinement`: behavior cloning loss wrapper around the policy.
  - `bc_loss`: negative log-likelihood of expert actions under the policy’s diagonal Gaussian.

### Assumptions
- In this minimal implementation, `gt_trajectory` is treated as the expert action sequence in the same space as `policy.actions` (B,T,A). If GT is pose-space (x,y,heading), Phase 4 integration will add a configurable mapping (pose deltas vs absolute).

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_expert_refinement.py`

Passed gates:
- Shape test: `bc_loss` is scalar.
- Equation oracle: NLL for 1 step matches hand-computed `-log N(a | μ, σ)`.
- Overfit test: when GT equals policy mean and σ is tiny, BC loss approaches 0.
- Gradient test: `bc_loss.backward()` produces gradients for policy parameters and none for critic.
- Integration test: one PPO update step followed by one BC refinement step runs without error.

### Figure Verification
**Figure classification**
- figure2_page3.png: architecture (contains expert-guided refinement stage)
- figure6_page15.png: results/ablation (refinement impact)

**Forward mapping (figure → code)**
- Figure 2, component "Expert-guided refinement" → implementation/expert_refinement.py:ExpertRefinement

**Reverse mapping (code → figure)**
- implementation/expert_refinement.py:ExpertRefinement → Figure 2, component "Expert-guided refinement"

---

## Submodule 9: rule_selector

### What was implemented
- `implementation/rule_selector.py`
  - `RuleSelector`: deterministic argmin selector over K candidates.
  - Scoring terms:
    - collision proxy (threshold on |dx|/|dy|)
    - comfort proxy (mean squared jerk)

### Assumptions
- Full nuPlan rule checking (collision with raster obstacles, on-road, red-light, etc.) is not specified in extracted artifacts and is expensive to implement. For Phase 3 gating we implement a deterministic proxy that still satisfies the selector’s required preferences (collision-free > comfort).

### Evidence (tests)
All tests run with:
`/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_rule_selector.py`

Passed gates:
- Shape test: selected trajectory is `(B,T,A)` and selected_idx is `(B,)`.
- Determinism: repeated calls return identical outputs.
- Collision preference: selector chooses collision-free candidate when available.
- Comfort preference: among collision-free candidates, selector chooses lower-jerk.
- Fallback: if all violate, still returns argmin score.

### Figure Verification
**Figure classification**
- figure2_page3.png: architecture (contains rule-based selection block)
- figure5_page14.png: results (qualitative selection examples)

**Forward mapping (figure → code)**
- Figure 2, component "Rule selector" → implementation/rule_selector.py:RuleSelector

**Reverse mapping (code → figure)**
- implementation/rule_selector.py:RuleSelector → Figure 2, component "Rule selector"

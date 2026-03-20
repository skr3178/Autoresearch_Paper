# Phase 3 Brief — Component Implementation (CarPlanner)

This brief grounds each Phase 3 submodule in the paper artifacts extracted in Phase 0.

⚠️ Note: The Phase 0 extraction artifacts in this repo are incomplete/partially inconsistent (e.g., `paper_contract.md` still references CLS-NR while `requirements.md` specifies L1/L2 displacement errors). For Phase 3 we implement submodules with minimal, configurable assumptions and explicitly flag any missing paper grounding as INVENTION in code and later in `proof.md`.

## Submodule 1: data_loader

**Paper grounding**
- Equations: N/A (data pipeline)
- Figures: Fig(s) showing BEV raster / input representation (to be mapped after loading all figures in Step 1 of Phase 3)
- Algorithm: N/A

**Input → Output** (from paper, not from submodules.md)
- Input: nuPlan scenario logs + map layers; ego past states and future GT trajectory in ego-centric coordinates.
- Output: BEV raster tensor + ego history + GT future trajectory.

**Key design decisions**
- BEV channel semantics and rasterization details are not fully specified in extracted artifacts → implement a minimal BEV with configurable channels and value range [0,1].

**Verification plan**
- Equation oracle: N/A
- Overfit target: same `scenario_id` yields identical tensors (deterministic rasterization).

## Submodule 2: transition_model

**Paper grounding**
- Equations: transition/world model loss (if specified in extracted equations; otherwise treated as supervised next-BEV regression)
- Figures: world model / non-reactive simulation block in architecture figure(s)
- Algorithm: rollout/imagination steps in RL algorithm (if present)

**Input → Output**
- Input: current BEV and an ego action/waypoint.
- Output: predicted next BEV.

**Key design decisions**
- Paper likely uses a learned non-reactive transition; exact update equation may be absent in extraction → implement a CNN conditioned on action via FiLM.

**Verification plan**
- Equation oracle: verify FiLM conditioning math on a tiny tensor.
- Overfit target: memorize 2 (bev, action)→next_bev pairs.

## Submodule 3: mode_selector

**Paper grounding**
- Equations: cross-entropy over discrete mode c (if specified)
- Figures: mode selection head in architecture figure(s)
- Algorithm: IL pretraining stage (if present)

**Input → Output**
- Input: BEV + ego history.
- Output: logits over K modes and argmax mode index.

**Key design decisions**
- K (number of modes) not confirmed in extracted artifacts → configurable `K`.

**Verification plan**
- Equation oracle: softmax + CE on a tiny example.
- Overfit target: 100% accuracy on 4 samples.

## Submodule 4: autoregressive_policy

**Paper grounding**
- Equations: autoregressive factorization of trajectory distribution and log-prob computation.
- Figures: autoregressive decoder/policy block.
- Algorithm: trajectory generation loop.

**Input → Output**
- Input: BEV + ego history + mode c.
- Output: action sequence and per-step log-probs.

**Key design decisions**
- Action parameterization (A dims) and distribution (Gaussian) not fully specified in extracted artifacts → implement diagonal Gaussian with configurable std (or learned log_std).

**Verification plan**
- Equation oracle: manual 1-step Gaussian log-prob.
- Overfit target: fit 2 fixed trajectories.

## Submodule 5: consistency_module

**Paper grounding**
- Equations: temporal consistency penalty (jerk/heading smoothness) if specified.
- Figures: consistency constraint block.
- Algorithm: added as auxiliary loss.

**Input → Output**
- Input: actions/trajectory.
- Output: per-sample penalty.

**Key design decisions**
- Exact penalty form may be underspecified → implement squared jerk (second difference) + heading rate penalty, both configurable.

**Verification plan**
- Equation oracle: hand compute second-difference penalty for 3 steps.
- Overfit target: N/A (loss module).

## Submodule 6: critic

**Paper grounding**
- Equations: value function regression loss.
- Figures: critic/value head.
- Algorithm: advantage estimation.

**Input → Output**
- Input: BEV + ego history + mode c.
- Output: scalar V(s).

**Key design decisions**
- Shared encoder with policy is ambiguous; implement separate encoder to satisfy independence gate.

**Verification plan**
- Equation oracle: MSE on tiny values.
- Overfit target: memorize 2 values.

## Submodule 7: ppo_loop

**Paper grounding**
- Equations: PPO clipped objective, value loss, entropy bonus, total loss.
- Figures: training loop diagram.
- Algorithm: PPO update algorithm.

**Input → Output**
- Input: batch of observations.
- Output: scalar losses.

**Key design decisions**
- Reward definition and rollout horizon may be paper-specific; for gate tests use synthetic rewards / placeholder until full integration.

**Verification plan**
- Equation oracle: clipping behavior for synthetic ratio.
- Overfit target: monotonic decrease on fixed batch.

## Submodule 8: expert_refinement

**Paper grounding**
- Equations: behavior cloning NLL.
- Figures: expert-guided refinement block.
- Algorithm: refinement stage.

**Input → Output**
- Input: obs + mode + GT trajectory.
- Output: scalar BC loss.

**Key design decisions**
- Mapping GT trajectory to action space ambiguous; implement direct regression/NLL in same parameterization as policy.

**Verification plan**
- Equation oracle: NLL for 1 step.
- Overfit target: BC loss → 0 on trivial case.

## Submodule 9: rule_selector

**Paper grounding**
- Equations: rule-based scoring (if specified)
- Figures: generation-selection framework.
- Algorithm: selection step.

**Input → Output**
- Input: K candidate trajectories + BEV.
- Output: selected trajectory + index.

**Key design decisions**
- Full traffic rule checking requires map semantics; implement minimal collision/comfort scoring from BEV occupancy + jerk.

**Verification plan**
- Equation oracle: N/A
- Overfit target: deterministic argmin selection.

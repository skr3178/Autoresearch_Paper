# Notation Map

<!-- Maintained throughout implementation. Updated in Phase 3, Step 1 for each new component. -->
<!-- Maps paper notation to code. This is the primary defense against math-to-code translation errors. -->

| Paper symbol | Code variable | File:line | Meaning | Value/Range |
|-------------|---------------|-----------|---------|-------------|
| $\pi(a_t\mid \mathbf{s}_t,\mathbf{c})$ | `AutoregressivePolicy.forward(...)->PolicyOutput` | `implementation/autoregressive_policy.py` | Autoregressive policy distribution over action at step t conditioned on state and mode | diagonal Gaussian |
| $a_t$ | `actions[:, t]` | `implementation/autoregressive_policy.py` | action at step t (proxy: delta pose) | R^A |
| $d_t$ | `(mu_t, log_std_t)` | `implementation/autoregressive_policy.py` | parameters of Gaussian distribution at step t | mu/log_std |
| $\log \pi(a_t\mid d_t)$ | `diag_gaussian_log_prob(a, mu, log_std)` | `implementation/autoregressive_policy.py` | log-prob used in PPO ratio (Eq 8) | scalar per step |
| $\mathbf{c}$ | `c` | `implementation/autoregressive_policy.py` | discrete mode index | {0..K-1} |
| consistency / comfort | `ConsistencyModule.forward(actions)` | `implementation/consistency_module.py` | auxiliary penalty encouraging smoothness (jerk + heading accel) | >=0 |
| $V(\mathbf{s}_t)$ | `Critic.forward(...)->value` | `implementation/critic.py` | value function estimate for PPO/GAE | R |
| $r_t$ | `ratio = exp(logp_new - logp_old)` | `implementation/ppo_loop.py` | PPO probability ratio (Eq 8) | >0 |
| $\epsilon$ | `ppo_clip_eps` | `implementation/ppo_loop.py` | PPO clipping parameter | default 0.2 |
| PolicyLoss | `policy_loss` | `implementation/ppo_loop.py` | clipped surrogate loss (Eq 8) | scalar |
| ValueLoss | `value_loss` | `implementation/ppo_loop.py` | squared error to return target (Eq 9) | scalar |
| Entropy | `entropy` | `implementation/ppo_loop.py` | mean Gaussian entropy (Eq 10) | scalar |
| $\lambda_{policy},\lambda_{value},\lambda_{entropy}$ | `lambda_policy/value/entropy` | `implementation/ppo_loop.py` | RL loss coefficients (Sec 4.1) | 100,3,0.001 |
| Expert NLL | `ExpertRefinement.bc_loss(...)` | `implementation/expert_refinement.py` | behavior cloning negative log-likelihood of expert actions | scalar |
| Rule selector | `RuleSelector.select(...)` | `implementation/rule_selector.py` | deterministic selection among K candidates (collision + comfort proxy) | argmin score |

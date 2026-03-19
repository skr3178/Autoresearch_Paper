# CarPlanner equations

Extracted from **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving** and lightly cleaned into Markdown/LaTeX.

## Equation (1)

$$
\max_{\pi} \mathbb{E}_{\mathbf{s}_t \sim P_{\tau},\, a_t \sim \pi}
\left[
\sum_{t=0}^{T} \gamma^t R(\mathbf{s}_t, a_t)
\right]
$$

## Equation (2)

$$P(\mathbf{s}_0, a_0, \mathbf{s}_1, a_1, \dots, \mathbf{s}_T)$$

$$= P(m, s^{0:N}_{-H:0}, s^0_1, m, s^{0:N}_{1-H:1}, s^0_2, \dots, m, s^{0:N}_{T-H:T})$$

$$= P(m, s^{0:N}_{-H:0}, m, s^{0:N}_{1-H:1}, \dots, m, s^{0:N}_{T-H:T})$$

$$= P(\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_T)$$

## Equation (3)

$$P(\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_T) = \rho_0(\mathbf{s}_0)\prod_{t=0}^{T-1} P(\mathbf{s}_{t+1} \mid \mathbf{s}_t)$$

$$= \rho_0(\mathbf{s}_0)\prod_{t=0}^{T-1} P(s^0_{t+1}, s^{1:N}_{t+1} \mid \mathbf{s}_t)$$

$$= \rho_0(\mathbf{s}_0)\prod_{t=0}^{T-1} \underbrace{\pi(a_t \mid \mathbf{s}_t)}_{\text{Policy}} \cdot \underbrace{P_{\tau}(s^{1:N}_{t+1} \mid \mathbf{s}_t)}_{\text{Transition Model}}$$

## Equation (4)

$$P(\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_T) = \int_{\mathbf{c}} P(\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_T, \mathbf{c})\, d\mathbf{c}$$

$$= \rho_0(\mathbf{s}_0)\int_{\mathbf{c}} P(\mathbf{c} \mid \mathbf{s}_0)\, P(\mathbf{s}_1, \dots, \mathbf{s}_T \mid \mathbf{c})\, d\mathbf{c}$$

$$= \rho_0(\mathbf{s}_0)\prod_{t=0}^{T-1} \underbrace{P_{\tau}(s^{1:N}_{t+1} \mid \mathbf{s}_t)}_{\text{Transition Model}} \int_{\mathbf{c}} \underbrace{P(\mathbf{c} \mid \mathbf{s}_0)}_{\text{Mode Selector}} \prod_{t=0}^{T-1} \underbrace{\pi(a_t \mid \mathbf{s}_t, \mathbf{c})}_{\text{Policy}} \, d\mathbf{c}$$

> Terms: $P_\tau$ = Transition Model, $P(\mathbf{c} \mid \mathbf{s}_0)$ = Mode Selector, $\pi(a_t \mid \mathbf{s}_t, \mathbf{c})$ = Policy

## Equation (5) — Transition Model Loss

$$
L_{\text{tm}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{n=1}^{N}
\left\| s_t^n - s_t^{n,\text{gt}} \right\|_1
$$

## Equation (6) — Mode Selector Loss

$$
\text{CrossEntropyLoss}(\mathbf{\sigma}, c^*)
=
-\sum_{i=1}^{N_{\text{mode}}}
\mathbb{I}(c_i = c^*) \log \sigma_i
$$

## Equation (7) — Side Task Loss

$$
\text{SideTaskLoss}(\bar{s}^{0}_{1:T}, s^{0,\text{gt}}_{1:T})
=
\frac{1}{T}\sum_{t=1}^{T}
\left\| \bar{s}_t^0 - s_t^{0,\text{gt}} \right\|_1
$$

## Equation (8) — PPO Policy Loss

$$\text{PolicyLoss}(a_{0:T-1}, d_{0:T-1,\text{new}}, d_{0:T-1}, A_{0:T-1})$$

$$= -\frac{1}{T}\sum_{t=0}^{T-1} \min\left( r_t A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\, A_t \right)$$

with

$$
r_t = \frac{\mathrm{Prob}(a_t, d_{t,\text{new}})}{\mathrm{Prob}(a_t, d_t)}
$$

## Equation (9) — Value Loss

$$
\text{ValueLoss}(V_{0:T-1,\text{new}}, \hat{R}_{0:T-1})
=
\frac{1}{T}\sum_{t=0}^{T-1}
\left\| V_{t,\text{new}} - \hat{R}_t \right\|_2^2
$$

## Equation (10) — Entropy

$$
\text{Entropy}(d_{0:T-1,\text{new}})
=
\frac{1}{T}\sum_{t=0}^{T-1} \mathcal{H}(d_{t,\text{new}})
$$

## Equation (11) — Generator Loss

$$
L_{\text{generator}}
=
\frac{1}{T}\sum_{t=1}^{T}
\left\| s_t^0 - s_t^{0,\text{gt}} \right\|_1
$$

---

## Total Loss Functions (from Section 3.4 and Section 4.1 — not individually numbered in paper)

### IL Total Loss

$$
L_{\text{IL}} = L_{\text{CE}} + L_{\text{SideTask}} + L_{\text{generator}}
$$

where $L_{\text{CE}}$ is Eq (6), $L_{\text{SideTask}}$ is Eq (7), $L_{\text{generator}}$ is Eq (11).

### RL Total Loss

$$
L_{\text{RL}} = \lambda_{\text{policy}} \cdot \text{PolicyLoss}
              + \lambda_{\text{value}} \cdot \text{ValueLoss}
              - \lambda_{\text{entropy}} \cdot \text{Entropy}
              + L_{\text{generator}}
$$

with coefficients from Section 4.1:
- $\lambda_{\text{policy}} = 100$
- $\lambda_{\text{value}} = 3$
- $\lambda_{\text{entropy}} = 0.001$

### GAE Advantage Estimation (from [30], used in Eq 8)

$$
A_t = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}, \qquad
\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)
$$

with $\gamma = 0.1$ (discount), $\lambda = 0.9$ (GAE parameter).

---

## Key Hyperparameters (Section 4.1)

| Parameter | Value |
|-----------|-------|
| $N_{\text{lon}}$ (longitudinal modes) | 12 |
| $N_{\text{lat}}$ (max lateral modes) | 5 |
| $N_{\text{mode}} = N_{\text{lat}} \times N_{\text{lon}}$ | 60 |
| Discount $\gamma$ | 0.1 |
| GAE $\lambda$ | 0.9 |
| $\lambda_{\text{policy}}$ | 100 |
| $\lambda_{\text{value}}$ | 3 |
| $\lambda_{\text{entropy}}$ | 0.001 |
| Batch size (per GPU) | 64 |
| GPUs | 2 × NVIDIA 3090 |
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| LR schedule | ReduceLROnPlateau, patience=0, factor=0.3 |
| Epochs | 50 |
| Training scenarios | 176,218 |
| Validation scenarios | 1,118 (100 per type, 14 types) |

---

## Notes

- Equation numbering follows the paper.
- The ratio definition for PPO is included immediately below Equation (8).
- Eq (1): paper uses $P_r$ (transition probability from MDP definition) in the subscript; functionally equivalent to $P_\tau$ used elsewhere.

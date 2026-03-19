# CarPlanner equations

Extracted from **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving** and lightly cleaned into Markdown/LaTeX.

## Equation (1)

```latex
\max_{\pi} \mathbb{E}_{\boldsymbol{s}_t \sim P_{\tau},\, a_t \sim \pi}
\left[
\sum_{t=0}^{T} \gamma^t R(\boldsymbol{s}_t, a_t)
\right].
```

## Equation (2)

```latex
\begin{aligned}
&P(\boldsymbol{s}_0, a_0, \boldsymbol{s}_1, a_1, \dots, \boldsymbol{s}_T) \\
&= P(m, s^{0:N}_{-H:0}, s^0_1, m, s^{0:N}_{1-H:1}, s^0_2, \dots, m, s^{0:N}_{T-H:T}) \\
&= P(m, s^{0:N}_{-H:0}, m, s^{0:N}_{1-H:1}, \dots, m, s^{0:N}_{T-H:T}) \\
&= P(\boldsymbol{s}_0, \boldsymbol{s}_1, \dots, \boldsymbol{s}_T).
\end{aligned}
```

## Equation (3)

```latex
\begin{aligned}
&P(\boldsymbol{s}_0, \boldsymbol{s}_1, \dots, \boldsymbol{s}_T)
= \rho_0(\boldsymbol{s}_0)\prod_{t=0}^{T-1} P(\boldsymbol{s}_{t+1} \mid \boldsymbol{s}_t) \\
&= \rho_0(\boldsymbol{s}_0)\prod_{t=0}^{T-1} P(s^0_{t+1}, s^{1:N}_{t+1} \mid \boldsymbol{s}_t) \\
&= \rho_0(\boldsymbol{s}_0)\prod_{t=0}^{T-1}
\underbrace{\pi(a_t \mid \boldsymbol{s}_t)}_{\text{Policy}}
\underbrace{P_{\tau}(s^{1:N}_{t+1} \mid \boldsymbol{s}_t)}_{\text{Transition Model}}.
\end{aligned}
```

## Equation (4)

```latex
\begin{aligned}
&P(\boldsymbol{s}_0, \boldsymbol{s}_1, \dots, \boldsymbol{s}_T)
= \int_{\boldsymbol{c}} P(\boldsymbol{s}_0, \boldsymbol{s}_1, \dots, \boldsymbol{s}_T, \boldsymbol{c})\, d\boldsymbol{c} \\
&= \rho_0(\boldsymbol{s}_0)\int_{\boldsymbol{c}} P(\boldsymbol{c} \mid \boldsymbol{s}_0)\,
P(\boldsymbol{s}_1, \dots, \boldsymbol{s}_T \mid \boldsymbol{c})\, d\boldsymbol{c} \\
&= \rho_0(\boldsymbol{s}_0)\prod_{t=0}^{T-1}
\underbrace{P_{\tau}(s^{1:N}_{t+1} \mid \boldsymbol{s}_t)}_{\text{Transition Model}}
\int_{\boldsymbol{c}}
\underbrace{P(\boldsymbol{c} \mid \boldsymbol{s}_0)}_{\text{Mode Selector}}
\prod_{t=0}^{T-1}
\underbrace{\pi(a_t \mid \boldsymbol{s}_t, \boldsymbol{c})}_{\text{Policy}}
\, d\boldsymbol{c}.
\end{aligned}
```

## Equation (5)

```latex
L_{\text{tm}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{n=1}^{N}
\left\| s_t^n - s_t^{n,\text{gt}} \right\|_1.
```

## Equation (6)

```latex
\text{CrossEntropyLoss}(\boldsymbol{\sigma}, c^*)
=
-\sum_{i=1}^{N_{\text{mode}}}
\mathbb{I}(c_i = c^*) \log \sigma_i.
```

## Equation (7)

```latex
\text{SideTaskLoss}(\bar{s}^{0}_{1:T}, s^{0,\text{gt}}_{1:T})
=
\frac{1}{T}\sum_{t=1}^{T}
\left\| \bar{s}_t^0 - s_t^{0,\text{gt}} \right\|_1.
```

## Equation (8)

```latex
\begin{aligned}
&\text{PolicyLoss}(a_{0:T-1}, d_{0:T-1,\text{new}}, d_{0:T-1}, A_{0:T-1}) \\
&= -\frac{1}{T}\sum_{t=0}^{T-1}
\min\left(
r_t A_t,\,
\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\, A_t
\right),
\end{aligned}
```

with

```latex
r_t = \frac{\mathrm{Prob}(a_t, d_{t,\text{new}})}{\mathrm{Prob}(a_t, d_t)}.
```

## Equation (9)

```latex
\text{ValueLoss}(V_{0:T-1,\text{new}}, \hat{R}_{0:T-1})
=
\frac{1}{T}\sum_{t=0}^{T-1}
\left\| V_{t,\text{new}} - \hat{R}_t \right\|_2^2.
```

## Equation (10)

```latex
\text{Entropy}(d_{0:T-1,\text{new}})
=
\frac{1}{T}\sum_{t=0}^{T-1} \mathcal{H}(d_{t,\text{new}}).
```

## Equation (11)

```latex
L_{\text{generator}}
=
\frac{1}{T}\sum_{t=1}^{T}
\left\| s_t^0 - s_t^{0,\text{gt}} \right\|_1.
```

## Notes

- Equation numbering follows the paper.
- The PDF text extraction introduced minor spacing artifacts, so this file is a cleaned transcription.
- The ratio definition for PPO is included immediately below Equation (8), as described in the supplementary text.

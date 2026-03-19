# Algorithms

Extracted from **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving**

---

## Algorithm 1: Training Procedure of CarPlanner (Supplementary Material, Page 11)

1. **Input:** Dataset $\mathcal{D}$ containing initial states $s_0$ and ground-truth trajectories $s^{0:N,\text{gt}}_{1:T}$, longitudinal modes $c_{\text{lon}}$, discount factor $\gamma$, GAE parameter $\lambda$, update interval $I$
2. **Require:** Non-reactive transition model $\beta$, mode selector $f_{\text{selector}}$, policy $\pi$, policy old $\pi_{\text{old}}$
3. **Step 1: Training Transition Model**
4. **for** $(s_0, s^{1:N,\text{gt}}_{1:T}) \in \mathcal{D}$ **do**
5. $\quad$ Simulate agent trajectories: $s^{1:N}_{1:T} \leftarrow \beta(s_0)$
6. $\quad$ Calculate loss: $L_{\text{tm}} \leftarrow \text{L1Loss}(s^{1:N}_{1:T},\ s^{1:N,\text{gt}}_{1:T})$
7. $\quad$ Backpropagate and update $\beta$ using $L_{\text{tm}}$
8. **end for**
9. **Step 2: Training Selector and Generator**
10. Initialize $\text{training\_step} \leftarrow 0$
11. Initialize policy old $\pi_{\text{old}} \leftarrow \pi$
12. **for** $(s_0, s^{0,\text{gt}}_{1:T}) \in \mathcal{D}$ **do**
13. $\quad$ **Non-Reactive Transition Model:**
14. $\quad\quad$ Simulate agent trajectories: $s^{1:N}_{1:T} \leftarrow \beta(s_0)$
15. $\quad$ **Mode Assignment:**
16. $\quad\quad$ Determine $c_{\text{lat}}$ based on $s_0$
17. $\quad\quad$ Concatenate $c_{\text{lat}}$ and $c_{\text{lon}}$ to get $c$
18. $\quad\quad$ Determine positive mode $c^*$ based on $s^{0,\text{gt}}_{1:T}$ and $c$
19. $\quad$ **Mode Selector Loss:**
20. $\quad\quad$ Compute scores: $\sigma,\ \bar{s}^0_{1:T} \leftarrow f_{\text{selector}}(s_0, c)$
21. $\quad\quad$ $L_{\text{selector}} \leftarrow \text{CrossEntropyLoss}(\sigma, c^*) + \text{SideTaskLoss}(\bar{s}^0_{1:T},\ s^{0,\text{gt}}_{1:T})$
22. $\quad$ **Generator Loss:**
23. $\quad\quad$ **if** Reinforcement Learning (RL) Training **then**
24. $\quad\quad\quad$ Use $\pi_{\text{old}}, s_0, c^*$, and $s^{1:N}_{1:T}$ to collect rollout data $(s_{0:T-1}, a_{0:T-1}, d_{0:T-1}, V_{0:T-1}, R_{0:T-1})$
25. $\quad\quad\quad$ Compute advantage $A_{0:T-1}$ and return $\hat{R}_{0:T-1}$ using GAE: $A_{0:T-1},\ \hat{R}_{0:T-1} \leftarrow \text{GAE}(R_{0:T-1}, V_{0:T-1}, \gamma, \lambda)$
26. $\quad\quad\quad$ Compute policy distribution and value estimates: $(d_{0:T-1,\text{new}}, V_{0:T-1,\text{new}}) \leftarrow \pi(s_{0:T-1}, a_{0:T-1}, c^*)$
27. $\quad\quad\quad$ $L_{\text{generator}} \leftarrow \text{ValueLoss}(V_{0:T-1,\text{new}}, \hat{R}_{0:T-1}) + \text{PolicyLoss}(d_{0:T-1,\text{new}}, d_{0:T-1}, A_{0:T-1}) - \text{Entropy}(d_{0:T-1,\text{new}})$
28. $\quad\quad$ **else if** Imitation Learning (IL) Training **then**
29. $\quad\quad\quad$ Use $\pi, s_0, c^*$, and $s^{1:N}_{1:T}$ to collect action sequence $a_{0:T-1}$
30. $\quad\quad\quad$ Stack action sequence as ego-planned trajectory: $s^0_{1:T} \leftarrow \text{Stack}(a_{0:T-1})$
31. $\quad\quad\quad$ $L_{\text{generator}} \leftarrow \text{L1Loss}(s^0_{1:T},\ s^{0,\text{gt}}_{1:T})$
32. $\quad\quad$ **end if**
33. $\quad$ **Overall Loss:**
34. $\quad\quad$ $L \leftarrow L_{\text{selector}} + L_{\text{generator}}$
35. $\quad\quad$ Backpropagate and update $f_{\text{selector}}, \pi$ using $L$
36. $\quad$ **Policy Update:**
37. $\quad\quad$ Increment $\text{training\_step} \leftarrow \text{training\_step} + 1$
38. $\quad\quad$ **if** $\text{training\_step}\ \%\ I = 0$ **then**
39. $\quad\quad\quad$ Update $\pi_{\text{old}} \leftarrow \pi$
40. $\quad\quad$ **end if**
41. **end for**

**Key observations:**
- Steps 1 and 2 are separate training stages. $\beta$ is frozen before Step 2 begins.
- The same loop handles both IL and RL training; the branch at line 19 selects the generator loss formula.
- $\pi_{\text{old}}$ is updated every $I$ steps (PPO-style delayed policy update), not every iteration.
- The selector loss $L_{\text{selector}}$ is identical in both IL and RL; only $L_{\text{generator}}$ differs.
- In RL, the RL loss coefficients from Section 4.1 ($\lambda_{\text{policy}}=100$, $\lambda_{\text{value}}=3$, $\lambda_{\text{entropy}}=0.001$) are applied inside `PolicyLoss`, `ValueLoss`, `Entropy` — the algorithm shows the unscaled form.

---

## Note on Reconstructed Algorithms

The following algorithms (2–3) are reconstructed from paper prose (Sections 3.3–3.4). Algorithm 1 above is verbatim from the supplementary material.

---

## Algorithm 1: CarPlanner Training (Reconstructed from Section 3.4)

**Inputs:**
- nuPlan dataset D of scenarios
- Non-reactive transition model $P_\tau$ (pre-trained, frozen)
- Mode selector $f_c$ (parameters $\theta_c$)
- Trajectory generator / autoregressive policy $\pi$ (parameters $\theta_\pi$)
- Value function $V$ (parameters $\theta_v$)

**Stage 1: Pre-train Transition Model**
1. For each scenario in D:
   - Sample initial state $s_0$, ground-truth agent trajectories $s^{1:N}_{1:T}$
   - Compute $L_{\text{tm}} = \frac{1}{T}\sum_{t=1}^T \sum_{n=1}^N \|s_t^n - s_t^{n,\text{gt}}\|_1$
   - Update $P_\tau$ via AdamW
2. Freeze $P_\tau$

**Stage 2: IL Pre-training (Mode Selector + Trajectory Generator)**
1. For each scenario in D:
   - Compute agent/map features from $s_0$
   - Assign positive lateral mode: route closest to GT endpoint
   - Assign positive longitudinal mode: interval containing GT endpoint speed
   - Compute mode selector loss $L_{\text{CE}}$ (Eq 6)
   - Compute side task loss $L_{\text{SideTask}}$ (Eq 7)
   - Compute generator loss $L_{\text{generator}}$ (Eq 11)
   - Total: $L_{\text{IL}} = L_{\text{CE}} + L_{\text{SideTask}} + L_{\text{generator}}$
   - Update $\theta_c$, $\theta_\pi$ via AdamW

**Stage 3: RL Fine-tuning (PPO)**
1. For each scenario batch:
   - Use frozen $P_\tau$ to preview world states $s_1, \dots, s_T$
   - Sample mode $c \sim f_c(s_0)$ (winner-takes-all: use GT-assigned positive mode during training)
   - Replicate $s_0$ for $N_{\text{mode}}$ parallel rollouts
   - For $t = 0, \dots, T-1$:
     - Sample action $a_t \sim \pi(a_t | s_t, c)$, record $\log \pi$
     - Compute next state via $P_\tau$
   - Compute rewards $R_t$: displacement error (DE) + collision + drivable area
   - Compute GAE advantages $A_t$ with $\gamma=0.1$, $\lambda=0.9$
   - Compute PPO losses (Eq 8, 9, 10, 11)
   - Total: $L_{\text{RL}} = 100 \cdot \text{PolicyLoss} + 3 \cdot \text{ValueLoss} - 0.001 \cdot \text{Entropy} + L_{\text{generator}}$
   - Update $\theta_\pi$, $\theta_v$ via AdamW (no backbone sharing in RL)

---

## Algorithm 2: CarPlanner Inference (Reconstructed from Section 3.3.4)

**Inputs:** Initial state $s_0$, number of candidates $N_{\text{mode}}$

1. Encode $s_0$ with agent and map encoders (PointNet + Transformer)
2. Run Mode Selector: compute scores for all $N_{\text{mode}} = N_{\text{lat}} \times N_{\text{lon}}$ modes
3. Apply IVM: transform agent/map/mode features to ego-centric coordinate frame at each step
4. For each mode $c_i$ ($i = 1 \dots N_{\text{mode}}$):
   - Autoregressively decode trajectory using $\pi(a_t | s_t, c_i)$ for $t = 0 \dots T-1$
   - Use mean of Gaussian (not sample) for deterministic inference
5. Run Rule Selector:
   - Compute rule-based scores: collision, drivable area, comfort, progress
   - Combine with mode selector scores (weighted sum)
   - Select trajectory with highest combined score
6. Return selected ego trajectory

---

## Algorithm 3: Invariant-View Module (IVM) — Section 3.3.3

**Purpose:** Eliminate absolute time information from inputs to make the policy time-agnostic.

**For map and agent features in state $s_t$:**
1. Select K-nearest neighbors to ego current pose (K = half of map/agent elements)
2. For routes (lateral mode): filter segments where the point closest to ego is the starting point, retaining $K_r = N_r / 4$ points per route
3. Transform all agent, map, and route poses to ego's current coordinate frame at time $t$
4. Subtract historical time steps $t-H:t$ from current time $t$, yielding time range $-H:0$
5. Output: time-agnostic feature representation

**Query-based Transformer decoder:**
- Mode $c$ acts as query (dimension $1 \times D$)
- Map + agent features act as keys/values (dimension $(N + N_m) \times D$)
- Output: updated mode feature (dimension $1 \times D$)
- Decoded through MLP → policy head (action distribution params) + value head (scalar)

# Tables

Extracted from **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving**

---

## Table 1: State-of-the-Art Comparison on nuPlan Test14-Hard

**Caption**: Comparison of CarPlanner against state-of-the-art planners on the nuPlan Test14-Hard benchmark (closed-loop, non-reactive).

**Benchmark**: nuPlan Test14-Hard, closed-loop non-reactive evaluation.

| Method | Type | CLS-NR | S-CR | S-TTC | S-Area | S-PR |
|--------|------|--------|------|-------|--------|------|
| PDM-Closed | Rule-based | 47.40 | 100.0 | 100.0 | 100.0 | 31.08 |
| CarPlanner-IL (Ours) | IL | 47.50 | 100.0 | 0.0 | 100.0 | 31.98 |
| CarPlanner (Ours) | IL+RL | **95.14** | **100.0** | **100.0** | **100.0** | **85.07** |

**Notes**:
- CLS-NR = Closed-Loop Score Non-Reactive (primary metric, higher is better)
- S-CR = Collision Rate score (higher is better)
- S-TTC = Time-to-Collision score (higher is better)
- S-Area = Drivable area adherence score (higher is better)
- S-PR = Progress score (higher is better)
- CarPlanner-IL's S-TTC=0 indicates collision-risk violations not fixed by RL training
- CarPlanner (full RL) achieves ~2× improvement over both baselines on CLS-NR
- Source: Figure 3 metric tables (qualitative comparison figure)

**Paper's recommended configuration**: CarPlanner (full IL+RL)

---

## Table 2: Comparison on nuPlan Test14-Random (Non-Reactive)

**Caption**: Comparison on the Test14-Random non-reactive benchmark (broader scenario coverage than Test14-Hard).

**Benchmark**: nuPlan Test14-Random, closed-loop non-reactive evaluation.

| Method | CLS-NR | Notes |
|--------|--------|-------|
| Baseline (no Mode Dropout, no Side Task, no Ego-hist Drop, no BB Share) | 91.67 | RL, all components off |
| CarPlanner RL best config | **94.07** | Mode Dropout + Selector Side Task only |
| CarPlanner IL best config | 93.41 | All four components on |

**Notes**:
- Source: Table 4 ablation (RL section, various row configurations)
- Test14-Random = 14 scenario types × 100 scenarios each = 1,400 scenarios
- This benchmark is used throughout the ablation study

---

## Table 3: Training Horizon Analysis

**Caption**: Performance of different training time horizons $T$ under different testing time horizons. Values are CLS-NR on Test14-Random non-reactive benchmark.

**Source**: Figure 4 (heatmap, page 13)

| Train \ Test T | 1 | 3 | 5 | 8 |
|----------------|-------|-------|-------|-------|
| 1 | 75.79 | 86.08 | 85.09 | 87.43 |
| 3 | 73.04 | 88.65 | 91.14 | 93.93 |
| 5 | 71.55 | 89.39 | 91.72 | 93.94 |
| **8** | 71.55 | 89.36 | 91.80 | **94.07** |

**Notes**:
- Diagonal (train=test=T) improves monotonically with T
- Training T=8, testing T=8 is the best configuration: CLS-NR=94.07
- Short training horizon (T=1) particularly hurts long-horizon test performance
- Paper default: T=8 for both training and testing
- Short training with T=1 forces testing T=1 → only 75.79 CLS-NR

**Paper's recommended configuration**: T=8 (train and test)

---

## Table 4: Ablation Study — Effect of Design Components

**Caption**: Effect of different components on IL and RL loss using CarPlanner. Results on Test14-Random non-reactive benchmark.

**Source**: table4_page8.png (page 8)

**Columns**:
- Design choices: Mode Dropout, Selector Side Task, Ego-history Dropout, Backbone Sharing
- Metrics (↑): CLS-NR, S-CR, S-Area, S-PR, S-Comfort
- Open-loop losses (↓): Loss Selector, Loss Generator

### IL Section

| Mode Drop | Sel. Side | Ego-hist Drop | BB Share | CLS-NR | S-CR | S-Area | S-PR | S-Comfort | L-Sel | L-Gen |
|-----------|-----------|---------------|----------|--------|------|--------|------|-----------|-------|-------|
| ✗ | ✗ | ✗ | ✗ | 92.05 | 97.29 | 98.46 | 91.52 | 1.04 | — | 147.5 |
| ✓ | ✗ | ✗ | ✗ | 91.21 | 96.54 | 98.46 | 91.44 | **96.92** | 1.07 | 153.0 |
| ✓ | ✓ | ✗ | ✗ | 91.21 | 96.91 | 98.46 | **95.30** | 93.86 | 1.04 | 147.5 |
| ✓ | ✓ | ✓ | ✗ | 92.72 | 98.06 | 98.46 | 94.88 | 95.53 | 1.04 | 167.5 |
| ✓ | ✓ | ✓ | ✓ | **93.41** | **98.85** | **98.85** | 93.87 | 96.15 | **1.04** | **174.3** |

**IL best config**: all four components on → CLS-NR=93.41

### RL Section

| Mode Drop | Sel. Side | Ego-hist Drop | BB Share | CLS-NR | S-CR | S-Area | S-PR | S-Comfort | L-Sel | L-Gen |
|-----------|-----------|---------------|----------|--------|------|--------|------|-----------|-------|-------|
| ✗ | ✗ | ✗ | ✗ | 91.67 | 98.85 | 98.46 | 91.69 | 90.73 | 1.04 | 1812.6 |
| ✓ | ✗ | ✗ | ✗ | 93.46 | 98.07 | **99.61** | 94.26 | 92.28 | 1.09 | 2254.6 |
| ✓ | ✓ | ✗ | ✗ | **94.07** | **99.22** | 98.85 | **95.30** | 93.37 | **1.05** | **540.3** |
| ✓ | ✓ | ✓ | ✗ | 89.51 | 97.27 | 98.46 | 90.93 | 83.20 | 1.05 | 524.3 |
| ✓ | ✓ | ✓ | ✓ | 88.66 | 95.54 | 98.84 | 92.82 | 86.05 | 1.21 | 1928.1 |

**RL best config**: Mode Dropout + Selector Side Task (NO Ego-history Dropout, NO Backbone Sharing) → CLS-NR=94.07

### Key Findings

| Finding | IL | RL |
|---------|----|----|
| Mode Dropout | Critical (necessary for both) | Critical |
| Selector Side Task | Improves S-PR significantly | Improves CLS-NR |
| Ego-history Dropout | Helps (+1.2 CLS-NR) | **Hurts** (-4.6 CLS-NR) |
| Backbone Sharing | Helps (+0.7 CLS-NR) | **Hurts** (-5.4 CLS-NR) |

**Implementation note**: Use separate configs for IL and RL training — the optimal component set differs substantially.

---

## Table 5 (Figure 6 fragment): Effect of Mode Representations

**Caption**: Effect of different mode representations on RL performance.

**Source**: Table fragment visible in figure6_page15.png

| RL Config | IL | Consistent | Vanilla | Lon-Lat |
|-----------|----|-----------:|--------:|--------:|
| Consistent | ✓ | ✓ | — | — |
| Vanilla | ✓ | — | ✓ | — |
| Lon-Lat | ✓ | — | — | ✓ |

**Notes**:
- "Consistent" = CarPlanner's proposed approach (mode $c$ fixed across all auto-regressive steps)
- "Vanilla" = vanilla auto-regressive (no consistent mode)
- "Lon-Lat" = longitudinal-lateral decomposed modes
- Full metric values not fully visible in extracted figure; consistent mode variant is the paper's proposed best

---

## Metric Definitions

| Metric | Full Name | Direction | Description |
|--------|-----------|-----------|-------------|
| CLS-NR | Closed-Loop Score Non-Reactive | ↑ higher | Primary metric. Composite score from collision, TTC, area, progress |
| S-CR | Collision Rate score | ↑ higher | Score component: absence of collisions |
| S-TTC | Time-to-Collision score | ↑ higher | Score component: safety margin |
| S-Area | Drivable Area score | ↑ higher | Score component: staying in drivable area |
| S-PR | Progress score | ↑ higher | Score component: forward progress toward goal |
| S-Comfort | Comfort score | ↑ higher | Score component: smooth acceleration/jerk |
| L-Sel | Loss Selector | ↓ lower | Cross-entropy loss for mode selector (open-loop proxy) |
| L-Gen | Loss Generator | ↓ lower | L1 waypoint loss for trajectory generator (open-loop proxy) |

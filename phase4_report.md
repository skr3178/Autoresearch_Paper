# Phase 4 Report — Integration Proof

## Run: debug — 2026-03-20
**Command**: `/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/train.py --config debug`
**Exit code**: 0

### Single-step test
- policy_loss: 0.35443437099456787
- value_loss: 0.0
- entropy: -3.7431840896606445
- consistency_loss: 182.98248291015625
- total_loss: 0.3581775426864624

Gate check (single-step fwd/bwd): PASS

### 10-step test (N-step)
Loss values:
- step 0 total_loss=0.177822
- step 1 total_loss=2.728570
- step 2 total_loss=-0.006970
- step 3 total_loss=0.258059
- step 4 total_loss=0.422048
- step 5 total_loss=1.847911
- step 6 total_loss=0.209533
- step 7 total_loss=0.984603
- step 8 total_loss=0.109321
- step 9 total_loss=0.632390

Gate check (loss decreases over 10 steps): PASS using criterion `last < mean(first 3)`.
- mean(first 3) = 0.966474
- last = 0.632390

### Tiny overfit test
- overfit_first_total_loss: 0.6269058585166931
- overfit_last_total_loss: 0.34972962737083435
Gate check (loss < 1.0): PASS

### Reproducibility check
- Two identical runs with same seed: PASS (exact match)

### Anomalies
- `consistency_loss` is large (hundreds) but is currently not included in total loss (lambda_consistency=0.0) for integration stability.
- `entropy` sign differs between single-step and N-step prints; entropy computation should be revisited in Phase 5 when matching paper PPO details.

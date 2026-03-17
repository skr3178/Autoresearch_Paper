# Implementation Plan

Agent reads this file during Phase 1.

## Decomposition Strategy
- **Start with simplest ablation** (single modality, no auxiliary losses)
- **Validate pipeline on simple/synthetic data first**
- **Profile before optimizing**

### Component Dependency graph
```
data → tokenizer → encoder → policy → critic → consistency_module → expert_guided_refinement → training_loop → evaluation
```

### Phase 1: Components (in order)

1. **data** - nuPlan dataset loader
   - Input: database files, map features
   - Output: normalized ego-centric representation (BEV features or agents, traffic lights)
 routes)
   - Exit criteria: shapes match contract, loader <2s/10 batches at batch_size=2
   
2. **encoder** - BEV feature encoder
   - Input: BEV features from data
   - Output: conditioning vector for policy
   - Exit criteria: forward/backward passes, equation oracle test
   
3. **policy** - Auto-regressive trajectory generator
   - Input: conditioning vector from encoder
   - Output: trajectory waypoints (auto-regressive)
   - Exit criteria: can generate diverse trajectories, temporal consistency enforced, gradients flow
   
4. **consistency_module** - Temporal coherence across timesteps
   - Input: trajectory, hidden state
   - Output: consistency loss, gradient penalty
   - Exit criteria: different inputs produce different outputs, gradients non-zero
   
5. **critic** - Predict Q-value for each trajectory
   - Input: trajectory, hidden state
   - Output: Q-values
   - Exit criteria: Q-values distinguish good/bail trajectories
   
6. **expert_guided_refinement** - Expert-guided refinement network
   - Input: generator output, expert trajectories
   - Output: refined trajectories
   - Exit criteria: trajectories improve after refinement
   
7. **training_loop** - RL training loop
   - Input: batch from dataset
   - Output: loss, metrics
   - Exit criteria: loss decreases, no NaN
   
8. **evaluation** - Metric computation
   - Input: model, test data
   - Output: L1, L2, minADE, minFDE, HitRate
   - Exit criteria: metrics computed correctly, match paper's evaluation protocol

## Config Tiers

| Config | Description | When to use |
|-------|-------------|---------------|
| debug | Smallest: 2 layers, 64 hidden, batch 2, 10 steps | Initial component testing |
| smoke | Small but realistic: fewer layers, small batch | Full pipeline validation |
| full | Paper-scale or hardware-adapted | Real training runs |

## Notes
- Start with debug config for all components
- Use smoke config for integration testing
- Paper uses RL training. Metrics will be worse on mini, but training pipeline should still work.
- Full training may require significant compute resources (paper uses 8×A100s)

# Progress Checklist

## Component Checklist

1. **Data Loader**
   - Exit Criteria: Successfully loads and preprocesses data, matches expected shapes and types.

2. **Transition Model**
   - Exit Criteria: Pre-trained, outputs match expected future states.

3. **Mode Selector**
   - Exit Criteria: Correctly predicts trajectory modality, high accuracy on validation.

4. **Auto-regressive Policy**
   - Exit Criteria: Generates plausible trajectories, passes overfit test.

5. **Consistency Module**
   - Exit Criteria: Penalizes implausible transitions, improves trajectory coherence.

6. **Critic**
   - Exit Criteria: Accurately estimates value function, aids in advantage computation.

7. **PPO Loop**
   - Exit Criteria: Successfully orchestrates PPO updates, improves policy performance.

8. **Expert Refinement**
   - Exit Criteria: Fine-tunes policy using expert demonstrations, improves performance.

9. **Rule Selector**
   - Exit Criteria: Selects best trajectory based on rule-based criteria.

## Exit Gate

- Each component must meet its exit criteria before proceeding to the next.
- **Debug Config**: Verify basic functionality and shape compliance.
- **Smoke Config**: Ensure loss decreases and model trains without errors.
- **Full Config**: Achieve performance metrics close to those reported in the paper.

## Current Progress
- [ ] Data Loader
- [ ] Transition Model
- [ ] Mode Selector
- [ ] Auto-regressive Policy
- [ ] Consistency Module
- [ ] Critic
- [ ] PPO Loop
- [ ] Expert Refinement
- [ ] Rule Selector

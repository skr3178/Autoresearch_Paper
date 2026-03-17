# Paper Requirements

Agent reads this file as the primary specification.

---

## Paper

**Title:** CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving

**Summary:** CarPlanner introduces an auto-regressive planner using RL to generate multi-modal trajectories. The auto-regressive structure enables efficient large-scale RL training, while consistency constraints ensure stable policy learning by maintaining temporal coherence across timesteps. A generation-selection framework with expert-guided refinement improves training. To the best of our knowledge, CarPlanner is the first RL-based planner to surpass IL- and rule-based Sotas on nuPlan.

 the **Key architectural idea:** Auto-regressive trajectory generation with consistency constraints
** **Novel vs prior work:** First to use RL for trajectory planning; combines generation and selection with expert guidance
** **Key paper sections:**
- Section 3.1: Auto-regressive trajectory generation
- Section 3.2: Consistency constraints  
- Section 3.3: Expert-guided refinement
- Section 4: Training procedure (RL + IL hybrid)
  
**Key components:**
- Auto-regressive trajectory generator
- Consistency module
- Expert-guided refinement network
- RL training loop

- Evaluation

**Key paper sections to implement:**
- Section 3.1: Auto-regressive trajectory generation
- Section 3.2: Consistency constraints  
- Section 3.3: Expert-guided refinement
- Section 4: Training procedure

- Appendix (implementation details)

---

## Dataset

**Name:** nuPlan
**Location:** To be downloaded (requires account at nuplan.org)
**Format:** Database logs with ego pose, surrounding agents, traffic lights, and route information. 20Hz frequency. One sample = T20-frame trajectory (T timesteps) × (T + T actions (T + action_dim)
**Preprocessing required:** Normalize coordinates, extract BE features
** **Train / val split:** 
- Train: Boston, Pittsburgh, Singapore, Las Vegas (or mini: 72 logs)
- Val: Same cities (test set is also public)
- Test: held out for competition

- Mini: 72 logs for quick prototyping

**Notes:** Mini split has ~72 logs, much smaller than full train. Good for initial development and faster iteration.
---
## Evaluation Metric
**Metric name:** L1 (L2) and L2 (displacement error),**Direction:** Lower is better
**How to compute:** 
1. Generate K trajectories from the model
2. Compute L1 and l2 errors for each trajectory
3. Select best trajectory (lowest l1+l2 or L2)
4 4. Compute final displacement error (l1+l2)/ 2
5. Average over K samples

**Log format:**
```
metric: <value>
l1_loss: <value>
...
```

**Paper's Reported baseline**
**Metric value:** L1 (l2+l2) = 0.92, l1 = 0.92, l2 (better than rule-based Sota)
**Model scale:** TBD (need to check paper)
**Notes:** Paper uses full nuPlan dataset with RL training. We using mini split for initial development. Metrics will be ~2-3× worse due to smaller scale and less data diversity. expect training to use slightly higher learning rates.
---
## Training Budget
**Time budget per run:** 30 minutes for initial experiments,**Hardware available:** 1× RTX 4090 (24GB VRAM)
**Batch size guidance:** Paper uses batch=64, but we may need to adjust. Start with batch=16 for 24GB

---
## Additional Constraints
**Immutable files:** None (building from scratch)
**Available packages:** Standard PyTorch ecosystem (nuplan-devkit dependencies)

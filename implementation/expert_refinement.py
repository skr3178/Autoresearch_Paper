"""Expert-guided refinement (behavior cloning) for CarPlanner.

Submodule 8 contract:
- Input: bev, ego_history, c, gt_trajectory
- Output: bc_loss scalar (negative log-likelihood of expert actions under policy)

We interpret gt_trajectory as desired actions directly (same shape as policy actions).
In later integration, a mapping from GT poses to action deltas may be needed.
"""

from __future__ import annotations

from typing import Dict

import torch

from implementation.autoregressive_policy import AutoregressivePolicy, diag_gaussian_log_prob


class ExpertRefinement:
    def __init__(self, cfg: Dict, policy: AutoregressivePolicy):
        self.cfg = cfg
        self.policy = policy

    def bc_loss(
        self,
        bev: torch.Tensor,
        ego_history: torch.Tensor,
        c: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute behavior cloning NLL.

        Args:
            gt_trajectory: (B,T,A) expert actions (or mapped from poses)

        Returns:
            scalar loss
        """
        out = self.policy(
            bev,
            ego_history,
            c,
            deterministic=False,
            teacher_forcing_actions=gt_trajectory,
        )
        # NLL = -log_prob(expert_action)
        logp = diag_gaussian_log_prob(gt_trajectory, out.mu, out.log_std)  # (B,T)
        return (-logp).mean()

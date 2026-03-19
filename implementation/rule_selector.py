"""Rule-based selector for inference.

Submodule 9 contract:
- Input: candidate_trajectories (B,K,T,A), bev (B,C,H,W)
- Output: selected_trajectory (B,T,A), selected_idx (B,)

We implement a simple deterministic scoring:
- Collision proxy: if any step has |dx| or |dy| > collision_threshold -> collision penalty.
  (In real nuPlan, collision is computed against rasterized obstacles; deferred.)
- Comfort: mean squared jerk on xy dims.

Score = collision_weight * collision + comfort_weight * comfort
Select argmin score.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _jerk_cost(traj: torch.Tensor) -> torch.Tensor:
    """traj: (B,K,T,A) -> (B,K)"""
    B, K, T, A = traj.shape
    if T < 3:
        return torch.zeros(B, K, device=traj.device, dtype=traj.dtype)
    acc = traj[:, :, 1:] - traj[:, :, :-1]
    jerk = acc[:, :, 1:] - acc[:, :, :-1]
    xy = jerk[..., : min(2, A)]
    return (xy ** 2).mean(dim=(2, 3))


class RuleSelector:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.collision_threshold = float(cfg.get("collision_threshold", 3.0))
        self.collision_weight = float(cfg.get("collision_weight", 1000.0))
        self.comfort_weight = float(cfg.get("comfort_weight", 1.0))

    def score(self, candidate_trajectories: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
        # bev unused in this proxy implementation.
        B, K, T, A = candidate_trajectories.shape
        xy = candidate_trajectories[..., : min(2, A)]
        collision = (xy.abs().max(dim=-1).values > self.collision_threshold).any(dim=-1).float()  # (B,K)
        comfort = _jerk_cost(candidate_trajectories)  # (B,K)
        return self.collision_weight * collision + self.comfort_weight * comfort

    def select(self, candidate_trajectories: torch.Tensor, bev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(candidate_trajectories, bev)
        idx = torch.argmin(scores, dim=1)  # (B,)
        B = candidate_trajectories.shape[0]
        selected = candidate_trajectories[torch.arange(B, device=idx.device), idx]
        return selected, idx

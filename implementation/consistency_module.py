"""Temporal consistency / comfort penalty.

Submodule 5 contract:
- Input: actions (B, T_future, A)
- Output: consistency_loss (B,)

We implement a simple jerk penalty on the first two action dims (dx, dy) and
heading acceleration penalty on heading dim (d_heading).

For actions interpreted as delta pose per step:
- velocity ~ a_t
- acceleration ~ a_t - a_{t-1}
- jerk ~ (a_t - a_{t-1}) - (a_{t-1} - a_{t-2})

A perfectly constant-velocity trajectory => zero acceleration and jerk => loss ~ 0.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ConsistencyModule(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.t_future = int(cfg["t_future"])
        self.action_dim = int(cfg["action_dim"])
        self.jerk_weight = float(cfg.get("jerk_weight", 1.0))
        self.heading_acc_weight = float(cfg.get("heading_acc_weight", 1.0))
        self.eps = float(cfg.get("consistency_eps", 1e-8))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        assert actions.dim() == 3, actions.shape
        B, T, A = actions.shape
        assert T == self.t_future
        assert A == self.action_dim

        if T < 3:
            return torch.zeros(B, device=actions.device, dtype=actions.dtype)

        # acceleration: (B, T-1, A)
        acc = actions[:, 1:] - actions[:, :-1]
        # jerk: (B, T-2, A)
        jerk = acc[:, 1:] - acc[:, :-1]

        # Penalize jerk on xy dims if available.
        xy_dims = min(2, A)
        jerk_xy = jerk[..., :xy_dims]
        jerk_loss = (jerk_xy ** 2).mean(dim=(1, 2))  # (B,)

        # Penalize heading acceleration if heading dim exists.
        if A >= 3:
            heading_acc = acc[..., 2]
            heading_acc_loss = (heading_acc ** 2).mean(dim=1)
        else:
            heading_acc_loss = torch.zeros(B, device=actions.device, dtype=actions.dtype)

        return self.jerk_weight * jerk_loss + self.heading_acc_weight * heading_acc_loss

"""Critic network for PPO value estimation.

Submodule 6 contract:
- Input: bev (B,C,H,W), ego_history (B,T_hist,3), c (B,)
- Output: value (B,)

Important: critic parameters must be independent from policy parameters.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.bev_channels = int(cfg["bev_channels"])
        self.t_hist = int(cfg["t_hist"])
        self.num_modes = int(cfg["num_modes"])

        d_model = int(cfg.get("critic_d_model", cfg.get("d_model", 128)))

        self.bev_encoder = nn.Sequential(
            nn.Conv2d(self.bev_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model),
            nn.ReLU(),
        )

        self.ego_hist_encoder = _mlp(self.t_hist * 3, (128,), d_model)
        self.mode_embed = nn.Embedding(self.num_modes, d_model)

        self.value_head = _mlp(d_model * 3, (256, 256), 1)

    def forward(self, bev: torch.Tensor, ego_history: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = bev.shape[0]
        obs_emb = self.bev_encoder(bev)
        ego_emb = self.ego_hist_encoder(ego_history.reshape(B, -1))
        mode_emb = self.mode_embed(c.long())
        feat = torch.cat([obs_emb, ego_emb, mode_emb], dim=-1)
        v = self.value_head(feat).squeeze(-1)
        return v.float()

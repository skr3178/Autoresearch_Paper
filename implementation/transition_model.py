"""Transition model (non-reactive world model).

Minimal implementation to satisfy submodules.md gates.

Public API expected by tests:
- TransitionModel(bev_channels, bev_height, bev_width, action_dim, hidden_dim)
- TransitionModelLoss
- TransitionModel.film with .gamma_net and .beta_net for oracle test

Note: film.forward supports both (B,D) and (B,D,H,W) feature tensors.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SimpleFiLM(nn.Module):
    """FiLM with separate gamma/beta linear nets for test controllability.

    Implements: y = x * gamma(cond) + beta(cond)

    The test oracle sets gamma_net.bias=1, gamma_net.weight=0 and expects gamma=1.
    """

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        if d > self.feat_dim:
            rep = (d + self.feat_dim - 1) // self.feat_dim
            gamma = self.gamma_net(cond).repeat(1, rep)[:, :d]
            beta = self.beta_net(cond).repeat(1, rep)[:, :d]
        else:
            gamma = self.gamma_net(cond)[:, :d]
            beta = self.beta_net(cond)[:, :d]

        if x.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


class TransitionModel(nn.Module):
    def __init__(
        self,
        bev_channels: int,
        bev_height: int,
        bev_width: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.bev_channels = int(bev_channels)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        c = self.bev_channels
        a = self.action_dim
        h = self.hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(c, h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.film = SimpleFiLM(feat_dim=h, cond_dim=a)

        self.decoder = nn.Sequential(
            nn.Conv2d(h, h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, bev: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(bev)
        h = self.film(h, action)
        next_bev = self.decoder(h)
        done = torch.zeros((bev.shape[0],), device=bev.device, dtype=torch.float32)
        return next_bev, done


class TransitionModelLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_next_bev: torch.Tensor, target_next_bev: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(pred_next_bev, target_next_bev, reduction=self.reduction)

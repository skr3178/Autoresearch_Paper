#!/usr/bin/env python3
"""implementation/mode_selector.py

Submodule 3: Mode Selector (Section 3.1 / 3.3, Eq 6 + Eq 7 side task)

Predicts discrete mode c from observation.

Contract (submodules.md):
- Input:
  - bev: (B, C, H, W) float32
  - ego_history: (B, T_hist, 3) float32
- Output:
  - c: (B,) int64
  - c_logits: (B, K) float32

We implement a simple CNN encoder for BEV + MLP for ego history, fused by
concatenation, then a classifier head.

Note: Side-task head (predicting coarse ego future) is part of paper's selector
training, but submodule gate only requires logits + classification behavior.
We include an optional side-task head for later IL training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModeSelectorOutput:
    c: torch.Tensor  # (B,) int64
    c_logits: torch.Tensor  # (B, K) float32
    c_probs: torch.Tensor  # (B, K) float32
    side_task: Optional[torch.Tensor] = None  # (B, T_future, 3) optional


class ModeSelector(nn.Module):
    def __init__(
        self,
        bev_channels: int = 7,
        num_modes: int = 60,
        ego_hist_len: int = 5,
        ego_feat_dim: int = 32,
        bev_feat_dim: int = 128,
        hidden_dim: int = 256,
        side_task: bool = True,
        side_task_T: int = 8,
        side_task_dim: int = 3,
    ):
        super().__init__()
        self.num_modes = int(num_modes)
        self.ego_hist_len = int(ego_hist_len)
        self.side_task_enabled = bool(side_task)
        self.side_task_T = int(side_task_T)
        self.side_task_dim = int(side_task_dim)

        # BEV encoder -> (B, bev_feat_dim)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, bev_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Ego history encoder -> (B, ego_feat_dim)
        self.ego_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ego_hist_len * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ego_feat_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = bev_feat_dim + ego_feat_dim

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(hidden_dim, self.num_modes)

        if self.side_task_enabled:
            self.side_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.side_task_T * self.side_task_dim),
            )
        else:
            self.side_head = None

    def forward(self, bev: torch.Tensor, ego_history: torch.Tensor) -> ModeSelectorOutput:
        assert bev.dim() == 4, f"bev must be (B,C,H,W), got {bev.shape}"
        assert ego_history.dim() == 3 and ego_history.shape[-1] == 3, f"ego_history must be (B,T,3), got {ego_history.shape}"

        bev_feat = self.bev_encoder(bev)
        ego_feat = self.ego_encoder(ego_history)
        fused = torch.cat([bev_feat, ego_feat], dim=-1)
        h = self.fuse(fused)

        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        c = torch.argmax(probs, dim=-1).to(torch.int64)

        side = None
        if self.side_head is not None:
            side = self.side_head(h).view(bev.shape[0], self.side_task_T, self.side_task_dim)

        return ModeSelectorOutput(c=c, c_logits=logits, c_probs=probs, side_task=side)


def build_mode_selector(config: Dict) -> ModeSelector:
    return ModeSelector(
        bev_channels=config.get('bev_channels', 7),
        num_modes=config.get('num_modes', 60),
        ego_hist_len=config.get('ego_hist_len', 5),
        ego_feat_dim=config.get('ego_feat_dim', 32),
        bev_feat_dim=config.get('bev_feat_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        side_task=config.get('selector_side_task', True),
        side_task_T=config.get('T_future', 8),
        side_task_dim=3,
    )

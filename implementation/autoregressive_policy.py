"""Autoregressive policy (trajectory generator) for CarPlanner.

This is a minimal, testable implementation matching `submodules.md`.

Paper refs:
- Section 3.1 / 3.3: auto-regressive trajectory generation
- Eq (8): PPO log-prob ratio uses Prob(a_t | d_t) where d_t is policy distribution.

We model each step's action distribution as a diagonal Gaussian:
  a_t ~ Normal(mu_t, sigma_t)

The policy is autoregressive via an RNNCell over previous action (teacher-forcing optional).
Mode c is embedded and concatenated to the observation embedding.

All dimensions are parameterized via `cfg`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, act: nn.Module = nn.ReLU()) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), act]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def diag_gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute log N(x; mu, diag(std^2)) summed over last dim.

    Args:
        x: (..., A)
        mu: (..., A)
        log_std: (..., A)

    Returns:
        log_prob: (...) float32

    Oracle-friendly: matches manual computation.
    """
    x = x.float()
    mu = mu.float()
    log_std = log_std.float()
    var = torch.exp(2.0 * log_std)
    log_2pi = math.log(2.0 * math.pi)
    return -0.5 * (((x - mu) ** 2) / var + 2.0 * log_std + log_2pi).sum(dim=-1)


@dataclass
class PolicyOutput:
    actions: torch.Tensor  # (B, T, A)
    log_probs: torch.Tensor  # (B, T)
    mu: torch.Tensor  # (B, T, A)
    log_std: torch.Tensor  # (B, T, A)


class AutoregressivePolicy(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.bev_channels = int(cfg["bev_channels"])
        self.bev_h = int(cfg["bev_h"])
        self.bev_w = int(cfg["bev_w"])
        self.t_hist = int(cfg["t_hist"])
        self.t_future = int(cfg["t_future"])
        self.action_dim = int(cfg["action_dim"])  # A
        self.num_modes = int(cfg["num_modes"])  # K

        d_model = int(cfg.get("d_model", 128))
        rnn_hidden = int(cfg.get("rnn_hidden", 128))
        min_log_std = float(cfg.get("min_log_std", -5.0))
        max_log_std = float(cfg.get("max_log_std", 2.0))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

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

        self.rnn = nn.GRUCell(input_size=self.action_dim, hidden_size=rnn_hidden)

        self.head = _mlp(d_model * 3 + rnn_hidden, (256, 256), out_dim=self.action_dim * 2)

    def forward(
        self,
        bev: torch.Tensor,
        ego_history: torch.Tensor,
        c: torch.Tensor,
        *,
        deterministic: bool = False,
        teacher_forcing_actions: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        assert bev.dim() == 4
        assert ego_history.dim() == 3
        B = bev.shape[0]
        device = bev.device

        obs_emb = self.bev_encoder(bev)
        ego_emb = self.ego_hist_encoder(ego_history.reshape(B, -1))
        mode_emb = self.mode_embed(c.long())
        ctx = torch.cat([obs_emb, ego_emb, mode_emb], dim=-1)

        prev_a = torch.zeros(B, self.action_dim, device=device, dtype=torch.float32)
        h = torch.zeros(B, self.rnn.hidden_size, device=device, dtype=torch.float32)

        actions = []
        logps = []
        mus = []
        log_stds = []

        for t in range(self.t_future):
            h = self.rnn(prev_a, h)
            feat = torch.cat([ctx, h], dim=-1)
            out = self.head(feat)
            mu, log_std = out.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self._min_log_std, self._max_log_std)

            if deterministic:
                a = mu
            else:
                std = torch.exp(log_std)
                a = mu + std * torch.randn_like(std)

            lp = diag_gaussian_log_prob(a, mu, log_std)

            actions.append(a)
            logps.append(lp)
            mus.append(mu)
            log_stds.append(log_std)

            if teacher_forcing_actions is not None:
                prev_a = teacher_forcing_actions[:, t].detach()
            else:
                prev_a = a

        return PolicyOutput(
            actions=torch.stack(actions, dim=1),
            log_probs=torch.stack(logps, dim=1),
            mu=torch.stack(mus, dim=1),
            log_std=torch.stack(log_stds, dim=1),
        )

"""Tests for Submodule 6: critic.

Run:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_critic.py
"""

from __future__ import annotations

import time

import torch

from implementation.critic import Critic
from implementation.autoregressive_policy import AutoregressivePolicy


def _cfg():
    return {
        "bev_channels": 7,
        "bev_h": 224,
        "bev_w": 224,
        "t_hist": 5,
        "t_future": 8,
        "action_dim": 3,
        "num_modes": 60,
        "d_model": 64,
        "critic_d_model": 64,
    }


def test_shape_and_backward():
    cfg = _cfg()
    m = Critic(cfg)
    B = 3
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64)

    v = m(bev, ego, c)
    assert v.shape == (B,)
    loss = (v ** 2).mean()
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_overfit_two_values():
    torch.manual_seed(0)
    cfg = _cfg()
    m = Critic(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)

    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.tensor([0, 1], dtype=torch.int64)
    target = torch.tensor([1.5, -0.5], dtype=torch.float32)

    for _ in range(200):
        v = m(bev, ego, c)
        loss = (v - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    v = m(bev, ego, c)
    mse = (v - target).pow(2).mean().item()
    assert mse < 1e-3, mse


def test_independence_from_policy():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)

    policy_param_ids = {id(p) for p in policy.parameters()}
    critic_param_ids = {id(p) for p in critic.parameters()}
    assert policy_param_ids.isdisjoint(critic_param_ids)


def test_c_conditioning_changes_value():
    torch.manual_seed(0)
    cfg = _cfg()
    m = Critic(cfg)
    B = 1
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)

    v0 = m(bev, ego, torch.zeros(B, dtype=torch.int64))
    v1 = m(bev, ego, torch.ones(B, dtype=torch.int64) * 10)
    diff = (v0 - v1).abs().item()
    assert diff > 1e-4, diff


def main():
    t0 = time.time()
    test_shape_and_backward()
    test_overfit_two_values()
    test_independence_from_policy()
    test_c_conditioning_changes_value()
    dt = time.time() - t0
    print(f"OK - test_critic.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

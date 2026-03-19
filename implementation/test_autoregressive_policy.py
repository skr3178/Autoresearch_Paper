"""Tests for Submodule 4: autoregressive_policy.

Run with:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_autoregressive_policy.py
"""

from __future__ import annotations

import math
import time

import torch

from implementation.autoregressive_policy import AutoregressivePolicy, diag_gaussian_log_prob


def _cfg():
    return {
        "bev_channels": 7,
        "bev_h": 224,
        "bev_w": 224,
        "t_hist": 5,
        "t_future": 8,
        "action_dim": 3,  # (dx, dy, dheading) as a simple proxy
        "num_modes": 60,
        "d_model": 64,
        "rnn_hidden": 64,
        "min_log_std": -4.0,
        "max_log_std": 1.0,
    }


def test_shapes_and_dtypes():
    cfg = _cfg()
    m = AutoregressivePolicy(cfg)
    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64)

    out = m(bev, ego, c)
    assert out.actions.shape == (B, cfg["t_future"], cfg["action_dim"])
    assert out.log_probs.shape == (B, cfg["t_future"])
    assert out.mu.shape == out.actions.shape
    assert out.log_std.shape == out.actions.shape
    assert out.actions.dtype == torch.float32
    assert out.log_probs.dtype == torch.float32
    assert torch.isfinite(out.actions).all()
    assert torch.isfinite(out.log_probs).all()


def test_forward_backward_no_nan():
    cfg = _cfg()
    m = AutoregressivePolicy(cfg)
    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32, requires_grad=True)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32, requires_grad=True)
    c = torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64)

    out = m(bev, ego, c)
    loss = out.log_probs.sum() + out.actions.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_equation_oracle_log_prob_one_step():
    x = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)
    mu = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
    log_std = torch.log(torch.tensor([[1.0, 0.5, 2.0]], dtype=torch.float32))

    lp = diag_gaussian_log_prob(x, mu, log_std)

    std = torch.exp(log_std)
    var = std ** 2
    manual = -0.5 * (((x - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)
    assert torch.allclose(lp, manual, atol=1e-6), (lp, manual)


def test_overfit_two_trajectories():
    torch.manual_seed(0)
    cfg = _cfg()
    m = AutoregressivePolicy(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.tensor([0, 1], dtype=torch.int64)

    target = torch.zeros(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)
    target[0, :, 0] = 0.5
    target[1, :, 1] = -0.25

    for _ in range(500):
        out = m(bev, ego, c, deterministic=True)
        loss = (out.actions - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    out = m(bev, ego, c, deterministic=True)
    mse = (out.actions - target).pow(2).mean().item()
    assert mse < 1e-2, mse


def test_autoregressive_dependency():
    torch.manual_seed(0)
    cfg = _cfg()
    m = AutoregressivePolicy(cfg)
    B = 1
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.zeros(B, dtype=torch.int64)

    tf1 = torch.zeros(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)
    tf2 = tf1.clone()
    tf2[:, 0, 0] = 5.0

    out1 = m(bev, ego, c, deterministic=True, teacher_forcing_actions=tf1)
    out2 = m(bev, ego, c, deterministic=True, teacher_forcing_actions=tf2)

    diff = (out1.actions[:, 1] - out2.actions[:, 1]).abs().mean().item()
    assert diff > 1e-4, diff


def test_c_conditioning_changes_trajectory():
    torch.manual_seed(0)
    cfg = _cfg()
    m = AutoregressivePolicy(cfg)
    B = 1
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)

    c0 = torch.zeros(B, dtype=torch.int64)
    c1 = torch.ones(B, dtype=torch.int64) * 10

    out0 = m(bev, ego, c0, deterministic=True)
    out1 = m(bev, ego, c1, deterministic=True)

    mse = (out0.actions - out1.actions).pow(2).mean().item()
    # At init, mode embedding influence may be small; require a detectable difference.
    assert mse > 1e-4, mse


def main():
    t0 = time.time()
    test_shapes_and_dtypes()
    test_forward_backward_no_nan()
    test_equation_oracle_log_prob_one_step()
    test_overfit_two_trajectories()
    test_autoregressive_dependency()
    test_c_conditioning_changes_trajectory()
    dt = time.time() - t0
    print(f"OK - test_autoregressive_policy.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

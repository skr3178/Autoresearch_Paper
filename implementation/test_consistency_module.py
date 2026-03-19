"""Tests for Submodule 5: consistency_module.

Run:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_consistency_module.py
"""

from __future__ import annotations

import time

import torch

from implementation.consistency_module import ConsistencyModule


def _cfg():
    return {"t_future": 8, "action_dim": 3, "jerk_weight": 1.0, "heading_acc_weight": 1.0}


def test_shape():
    cfg = _cfg()
    m = ConsistencyModule(cfg)
    actions = torch.zeros(4, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)
    out = m(actions)
    assert out.shape == (4,)


def test_zero_for_constant_velocity():
    cfg = _cfg()
    m = ConsistencyModule(cfg)
    B = 2
    actions = torch.zeros(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)
    actions[..., 0] = 0.5
    actions[..., 1] = -0.25
    actions[..., 2] = 0.1
    out = m(actions)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), out


def test_sign_smooth_lower_than_jitter():
    cfg = _cfg()
    m = ConsistencyModule(cfg)
    B = 1
    smooth = torch.zeros(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)
    smooth[..., 0] = 0.2

    jitter = smooth.clone()
    jitter[:, 2, 0] = 2.0
    jitter[:, 3, 0] = -2.0

    ls = m(smooth).item()
    lj = m(jitter).item()
    assert ls < lj, (ls, lj)


def test_equation_oracle_three_step():
    # Use T=3 for hand computation.
    cfg = {"t_future": 3, "action_dim": 3, "jerk_weight": 1.0, "heading_acc_weight": 1.0}
    m = ConsistencyModule(cfg)

    # actions: a0, a1, a2
    actions = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float32
    )
    # acc0 = a1-a0 = [1,0,0]
    # acc1 = a2-a1 = [2,0,0]
    # jerk0 = acc1-acc0 = [1,0,0]
    # jerk_xy^2 mean over (T-2=1, xy_dims=2): mean([1^2,0^2])=0.5
    # heading_acc is zero.
    expected = torch.tensor([0.5], dtype=torch.float32)
    out = m(actions)
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)


def test_gradients_flow():
    cfg = _cfg()
    m = ConsistencyModule(cfg)
    actions = torch.randn(2, cfg["t_future"], cfg["action_dim"], dtype=torch.float32, requires_grad=True)
    loss = m(actions).mean()
    loss.backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()


def main():
    t0 = time.time()
    test_shape()
    test_zero_for_constant_velocity()
    test_sign_smooth_lower_than_jitter()
    test_equation_oracle_three_step()
    test_gradients_flow()
    dt = time.time() - t0
    print(f"OK - test_consistency_module.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

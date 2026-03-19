"""Tests for Submodule 9: rule_selector.

Run:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_rule_selector.py
"""

from __future__ import annotations

import time

import torch

from implementation.rule_selector import RuleSelector


def _cfg():
    return {
        "collision_threshold": 1.0,
        "collision_weight": 1000.0,
        "comfort_weight": 1.0,
    }


def test_shape():
    cfg = _cfg()
    sel = RuleSelector(cfg)
    B, K, T, A = 2, 5, 8, 3
    cand = torch.zeros(B, K, T, A, dtype=torch.float32)
    bev = torch.zeros(B, 7, 224, 224, dtype=torch.float32)
    traj, idx = sel.select(cand, bev)
    assert traj.shape == (B, T, A)
    assert idx.shape == (B,)


def test_determinism():
    cfg = _cfg()
    sel = RuleSelector(cfg)
    B, K, T, A = 1, 3, 8, 3
    cand = torch.randn(B, K, T, A)
    bev = torch.zeros(B, 7, 224, 224)
    t1, i1 = sel.select(cand, bev)
    t2, i2 = sel.select(cand, bev)
    assert torch.allclose(t1, t2)
    assert torch.equal(i1, i2)


def test_collision_preference():
    cfg = _cfg()
    sel = RuleSelector(cfg)
    B, K, T, A = 1, 2, 8, 3
    cand = torch.zeros(B, K, T, A)
    # candidate 0 collides (large dx)
    cand[0, 0, 0, 0] = 2.0
    # candidate 1 safe
    cand[0, 1, :, 0] = 0.2
    bev = torch.zeros(B, 7, 224, 224)
    traj, idx = sel.select(cand, bev)
    assert idx.item() == 1


def test_comfort_preference_among_safe():
    cfg = _cfg()
    sel = RuleSelector(cfg)
    B, K, T, A = 1, 2, 8, 3
    cand = torch.zeros(B, K, T, A)
    # both safe under threshold
    cand[0, 0, :, 0] = 0.2  # smooth
    cand[0, 1, :, 0] = 0.2
    cand[0, 1, 2, 0] = 0.9  # jitter but still below collision threshold
    cand[0, 1, 3, 0] = -0.9
    bev = torch.zeros(B, 7, 224, 224)
    traj, idx = sel.select(cand, bev)
    assert idx.item() == 0


def test_fallback_all_violate():
    cfg = _cfg()
    sel = RuleSelector(cfg)
    B, K, T, A = 1, 3, 8, 3
    cand = torch.ones(B, K, T, A) * 2.0  # all collide
    # make candidate 2 slightly less bad in comfort (still colliding)
    cand[0, 2] = 2.0
    bev = torch.zeros(B, 7, 224, 224)
    traj, idx = sel.select(cand, bev)
    assert idx.shape == (B,)
    assert traj.shape == (B, T, A)


def main():
    t0 = time.time()
    test_shape()
    test_determinism()
    test_collision_preference()
    test_comfort_preference_among_safe()
    test_fallback_all_violate()
    dt = time.time() - t0
    print(f"OK - test_rule_selector.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

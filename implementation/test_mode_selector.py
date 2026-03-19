#!/usr/bin/env python3
"""implementation/test_mode_selector.py

Verification tests for Submodule 3: mode_selector.

Run with:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_mode_selector.py

Gates (submodules.md):
- Shape test: c_logits (B,K)
- Forward+backward
- IL overfit: 100% train acc on 4 samples within 200 steps
- Diversity: at least 2 distinct modes predicted on a batch of 8 diverse scenarios
- Calibration: softmax probabilities sum to 1.0
"""

import sys

sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/implementation')

import torch
import torch.nn.functional as F

from mode_selector import ModeSelector


def test_shapes_and_calibration():
    print("=" * 80)
    print("Test 1: Shapes + Calibration")
    print("=" * 80)

    B, C, H, W = 8, 7, 224, 224
    T_hist = 5
    K = 16

    model = ModeSelector(bev_channels=C, num_modes=K, ego_hist_len=T_hist, hidden_dim=64, side_task=False)
    bev = torch.rand(B, C, H, W)
    ego_hist = torch.randn(B, T_hist, 3)

    out = model(bev, ego_hist)

    assert out.c_logits.shape == (B, K)
    assert out.c.shape == (B,)
    assert out.c.dtype == torch.int64
    assert out.c_logits.dtype == torch.float32

    probs = F.softmax(out.c_logits, dim=-1)
    s = probs.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

    print("✅ shapes ok")
    print("✅ softmax sums to 1")


def test_forward_backward():
    print("=" * 80)
    print("Test 2: Forward + Backward")
    print("=" * 80)

    model = ModeSelector(bev_channels=7, num_modes=8, ego_hist_len=5, hidden_dim=32, side_task=False)
    bev = torch.rand(2, 7, 224, 224)
    ego_hist = torch.randn(2, 5, 3)

    out = model(bev, ego_hist)
    target = torch.tensor([1, 3], dtype=torch.int64)
    loss = F.cross_entropy(out.c_logits, target)
    loss.backward()

    ok = True
    for p in model.parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            ok = False
            break
    assert ok
    print("✅ backward ok")


def test_overfit_4_samples():
    print("=" * 80)
    print("Test 3: IL Overfit 4 samples")
    print("=" * 80)

    torch.manual_seed(0)
    B, C, H, W = 4, 7, 64, 64
    T_hist = 5
    K = 8

    model = ModeSelector(bev_channels=C, num_modes=K, ego_hist_len=T_hist, hidden_dim=64, side_task=False)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    bev = torch.rand(B, C, H, W)
    ego_hist = torch.randn(B, T_hist, 3)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    model.train()
    for step in range(200):
        opt.zero_grad()
        out = model(bev, ego_hist)
        loss = F.cross_entropy(out.c_logits, labels)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(bev, ego_hist).c
        acc = (pred == labels).float().mean().item()

    print(f"train_acc={acc:.3f}")
    assert acc == 1.0


def test_diversity_modes():
    print("=" * 80)
    print("Test 4: Diversity")
    print("=" * 80)

    torch.manual_seed(1)
    B, C, H, W = 8, 7, 64, 64
    T_hist = 5
    K = 16

    model = ModeSelector(bev_channels=C, num_modes=K, ego_hist_len=T_hist, hidden_dim=64, side_task=False)
    model.eval()

    bev = torch.rand(B, C, H, W)
    ego_hist = torch.randn(B, T_hist, 3)

    with torch.no_grad():
        c = model(bev, ego_hist).c

    unique = torch.unique(c).numel()
    print(f"unique_modes={unique} / {B}")
    assert unique >= 2


def main():
    test_shapes_and_calibration()
    test_forward_backward()
    test_overfit_4_samples()
    test_diversity_modes()
    print("✅✅✅ mode_selector tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

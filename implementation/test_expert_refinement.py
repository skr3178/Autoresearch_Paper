"""Tests for Submodule 8: expert_refinement.

Run:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_expert_refinement.py
"""

from __future__ import annotations

import math
import time

import torch

from implementation.autoregressive_policy import AutoregressivePolicy
from implementation.critic import Critic
from implementation.consistency_module import ConsistencyModule
from implementation.expert_refinement import ExpertRefinement
from implementation.ppo_loop import PPOTrainer


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
        "ppo_clip_eps": 0.2,
        "gamma": 0.1,
        "gae_lambda": 0.9,
        "lambda_policy": 100.0,
        "lambda_value": 3.0,
        "lambda_entropy": 0.001,
        "lambda_consistency": 1.0,
        "normalize_adv": False,
    }


def test_shape_scalar():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    ref = ExpertRefinement(cfg, policy)

    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64)
    gt = torch.randn(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)

    loss = ref.bc_loss(bev, ego, c, gt)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_equation_oracle_nll_one_step():
    # For a 1D Gaussian with mu=0, std=1, x=0 => logp = -0.5*log(2pi)
    cfg = _cfg()
    cfg["t_future"] = 1
    cfg["action_dim"] = 1
    policy = AutoregressivePolicy(cfg)

    # Force head to output mu=0, log_std=0 by zeroing parameters.
    for p in policy.parameters():
        p.data.zero_()

    ref = ExpertRefinement(cfg, policy)

    B = 1
    bev = torch.zeros(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.zeros(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.zeros(B, dtype=torch.int64)
    gt = torch.zeros(B, 1, 1, dtype=torch.float32)

    loss = ref.bc_loss(bev, ego, c, gt)
    expected = 0.5 * math.log(2 * math.pi)
    assert abs(loss.item() - expected) < 1e-4, (loss.item(), expected)


def test_overfit_bc_loss_to_zeroish():
    torch.manual_seed(0)
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    ref = ExpertRefinement(cfg, policy)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.tensor([0, 1], dtype=torch.int64)

    # Use deterministic mean actions as expert; should be high likelihood after training.
    with torch.no_grad():
        expert = policy(bev, ego, c, deterministic=True).actions

    for _ in range(300):
        loss = ref.bc_loss(bev, ego, c, expert)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    loss = ref.bc_loss(bev, ego, c, expert)
    assert loss.item() < 0.2, loss.item()


def test_gradients_flow_to_policy_only():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    ref = ExpertRefinement(cfg, policy)

    B = 2
    bev = torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32)
    ego = torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32)
    c = torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64)
    gt = torch.randn(B, cfg["t_future"], cfg["action_dim"], dtype=torch.float32)

    loss = ref.bc_loss(bev, ego, c, gt)
    loss.backward()

    assert any(p.grad is not None for p in policy.parameters())
    assert all(p.grad is None for p in critic.parameters())


def test_integration_one_ppo_then_one_bc_step():
    torch.manual_seed(0)
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    cons = ConsistencyModule(cfg)
    ppo = PPOTrainer(cfg, policy, critic, cons)
    ref = ExpertRefinement(cfg, policy)

    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=3e-3)

    B = 2
    batch = {
        "bev": torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32),
        "ego_history": torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32),
        "c": torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64),
    }

    rollout = ppo.collect_rollout(batch, rewards=torch.ones(B, cfg["t_future"], dtype=torch.float32))
    losses = ppo.update(rollout)
    opt.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    opt.step()

    with torch.no_grad():
        expert = policy(batch["bev"], batch["ego_history"], batch["c"], deterministic=True).actions

    bc = ref.bc_loss(batch["bev"], batch["ego_history"], batch["c"], expert)
    opt.zero_grad(set_to_none=True)
    bc.backward()
    opt.step()

    assert torch.isfinite(bc)


def main():
    t0 = time.time()
    test_shape_scalar()
    test_equation_oracle_nll_one_step()
    test_overfit_bc_loss_to_zeroish()
    test_gradients_flow_to_policy_only()
    test_integration_one_ppo_then_one_bc_step()
    dt = time.time() - t0
    print(f"OK - test_expert_refinement.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

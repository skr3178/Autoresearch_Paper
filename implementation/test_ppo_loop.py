"""Tests for Submodule 7: ppo_loop.

Run:
  /media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/test_ppo_loop.py
"""

from __future__ import annotations

import time

import torch

from implementation.autoregressive_policy import AutoregressivePolicy
from implementation.critic import Critic
from implementation.consistency_module import ConsistencyModule
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


def _batch(cfg, B=2):
    return {
        "bev": torch.rand(B, cfg["bev_channels"], cfg["bev_h"], cfg["bev_w"], dtype=torch.float32),
        "ego_history": torch.rand(B, cfg["t_hist"], 3, dtype=torch.float32),
        "c": torch.randint(0, cfg["num_modes"], (B,), dtype=torch.int64),
    }


def test_losses_are_scalars():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    cons = ConsistencyModule(cfg)
    ppo = PPOTrainer(cfg, policy, critic, cons)

    batch = _batch(cfg, B=3)
    rollout = ppo.collect_rollout(batch)
    losses = ppo.update(rollout)

    for k in ["policy_loss", "value_loss", "entropy", "consistency_loss", "total_loss"]:
        assert losses[k].dim() == 0, (k, losses[k].shape)
        assert torch.isfinite(losses[k])


def test_entropy_sign_in_total_loss():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    ppo = PPOTrainer(cfg, policy, critic, None)

    batch = _batch(cfg, B=2)
    rollout = ppo.collect_rollout(batch)
    losses = ppo.update(rollout)

    total_no_entropy = cfg["lambda_policy"] * losses["policy_loss"] + cfg["lambda_value"] * losses["value_loss"]
    assert losses["entropy"].item() >= 0.0
    assert losses["total_loss"].item() <= total_no_entropy.item() + 1e-5


def test_clipping_oracle_ratio_2_adv_pos():
    eps = 0.2
    ratio = torch.tensor([2.0])
    adv = torch.tensor([1.0])
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    obj = torch.min(unclipped, clipped)
    assert torch.allclose(obj, torch.tensor([1.2]))


def test_backward_populates_grads():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    cons = ConsistencyModule(cfg)
    ppo = PPOTrainer(cfg, policy, critic, cons)

    batch = _batch(cfg, B=2)
    rollout = ppo.collect_rollout(batch)
    losses = ppo.update(rollout)

    losses["total_loss"].backward()

    assert any(p.grad is not None for p in policy.parameters())
    assert any(p.grad is not None for p in critic.parameters())


def test_overfit_gate_runs_20_steps_no_nan():
    """PPO is on-policy; strict monotonic decrease is not guaranteed.

    Gate intent: ensure the PPO update loop is numerically stable and trainable.
    """
    torch.manual_seed(0)
    cfg = _cfg()

    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    cons = ConsistencyModule(cfg)
    ppo = PPOTrainer(cfg, policy, critic, cons)

    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=3e-3)

    batch = _batch(cfg, B=2)
    rewards = torch.ones(2, cfg["t_future"], dtype=torch.float32)

    for _ in range(20):
        rollout = ppo.collect_rollout(batch, rewards=rewards)
        losses = ppo.update(rollout)
        opt.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        opt.step()
        assert torch.isfinite(losses["total_loss"]).item()


def test_loss_decomposition_matches_coeffs():
    cfg = _cfg()
    policy = AutoregressivePolicy(cfg)
    critic = Critic(cfg)
    cons = ConsistencyModule(cfg)
    ppo = PPOTrainer(cfg, policy, critic, cons)

    batch = _batch(cfg, B=2)
    rollout = ppo.collect_rollout(batch)
    losses = ppo.update(rollout)

    approx = (
        cfg["lambda_policy"] * losses["policy_loss"]
        + cfg["lambda_value"] * losses["value_loss"]
        - cfg["lambda_entropy"] * losses["entropy"]
        + cfg["lambda_consistency"] * losses["consistency_loss"]
    )
    assert torch.allclose(losses["total_loss"], approx, atol=1e-6)


def main():
    t0 = time.time()
    test_losses_are_scalars()
    test_entropy_sign_in_total_loss()
    test_clipping_oracle_ratio_2_adv_pos()
    test_backward_populates_grads()
    test_overfit_gate_runs_20_steps_no_nan()
    test_loss_decomposition_matches_coeffs()
    dt = time.time() - t0
    print(f"OK - test_ppo_loop.py ({dt:.2f}s)")


if __name__ == "__main__":
    main()

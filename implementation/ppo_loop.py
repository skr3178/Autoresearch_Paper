"""PPO training loop utilities (single update step).

Submodule 7 contract:
- Input: batch from data_loader
- Output: dict of scalar losses: policy_loss, value_loss, entropy, consistency_loss, total_loss

This is a minimal PPO implementation sufficient for unit tests and later integration.

Paper refs:
- Eq (8) PolicyLoss with clipping
- Eq (9) ValueLoss
- Eq (10) Entropy
- RL total loss coefficients (Sec 4.1):
  lambda_policy=100, lambda_value=3, lambda_entropy=0.001

We implement:
- collect_rollout(): uses current policy to sample actions and log_probs; uses critic for values.
  Rewards are synthetic in tests; in real training will come from environment/transition model.
- update(): computes GAE advantages, PPO losses, and backprop.

Note: This module does not yet integrate the transition model; tests focus on PPO math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch

from implementation.autoregressive_policy import AutoregressivePolicy, PolicyOutput
from implementation.critic import Critic
from implementation.consistency_module import ConsistencyModule


@dataclass
class Rollout:
    bev: torch.Tensor
    ego_history: torch.Tensor
    c: torch.Tensor
    actions: torch.Tensor  # (B,T,A)
    log_probs_old: torch.Tensor  # (B,T)
    values: torch.Tensor  # (B,T+1)
    rewards: torch.Tensor  # (B,T)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> Dict[str, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: (B,T)
        values: (B,T+1) bootstrap values (last is V_T)

    Returns:
        advantages: (B,T)
        returns: (B,T)
    """
    B, T = rewards.shape
    adv = torch.zeros(B, T, device=rewards.device, dtype=rewards.dtype)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        gae = delta + gamma * lam * gae
        adv[:, t] = gae
    ret = adv + values[:, :-1]
    return {"advantages": adv, "returns": ret}


class PPOTrainer:
    def __init__(
        self,
        cfg: Dict,
        policy: AutoregressivePolicy,
        critic: Critic,
        consistency: Optional[ConsistencyModule] = None,
    ):
        self.cfg = cfg
        self.policy = policy
        self.critic = critic
        self.consistency = consistency

        self.clip_eps = float(cfg.get("ppo_clip_eps", 0.2))
        self.gamma = float(cfg.get("gamma", 0.1))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.9))

        self.lambda_policy = float(cfg.get("lambda_policy", 100.0))
        self.lambda_value = float(cfg.get("lambda_value", 3.0))
        self.lambda_entropy = float(cfg.get("lambda_entropy", 0.001))
        self.lambda_consistency = float(cfg.get("lambda_consistency", 1.0))

        # Stabilization knobs for integration proof (configurable; defaults chosen to keep scales O(1)).
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.reward_clip = cfg.get("reward_clip", None)  # e.g. 10.0
        self.value_clip = cfg.get("value_clip", None)  # e.g. 10.0

    def collect_rollout(self, batch: Dict[str, torch.Tensor], rewards: Optional[torch.Tensor] = None) -> Rollout:
        bev = batch["bev"]
        ego = batch["ego_history"]
        c = batch["c"]

        pol_out: PolicyOutput = self.policy(bev, ego, c, deterministic=False)

        # Value predictions per timestep (minimal: repeat V(s0) across time).
        v0 = self.critic(bev, ego, c)  # (B,)
        B, T = pol_out.log_probs.shape
        values = v0[:, None].expand(B, T + 1).contiguous()

        if rewards is None:
            rewards = torch.zeros(B, T, device=bev.device, dtype=torch.float32)

        # Scale/clip rewards to keep return magnitudes controlled.
        rewards = rewards.to(dtype=torch.float32) * self.reward_scale
        if self.reward_clip is not None:
            rewards = torch.clamp(rewards, -float(self.reward_clip), float(self.reward_clip))

        return Rollout(
            bev=bev,
            ego_history=ego,
            c=c,
            actions=pol_out.actions.detach(),
            log_probs_old=pol_out.log_probs.detach(),
            values=values.detach(),
            rewards=rewards,
        )

    def update(self, rollout: Rollout) -> Dict[str, torch.Tensor]:
        gae = compute_gae(rollout.rewards, rollout.values, gamma=self.gamma, lam=self.gae_lambda)
        adv = gae["advantages"].detach()
        ret = gae["returns"].detach()

        if bool(self.cfg.get("normalize_adv", True)):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pol_new = self.policy(
            rollout.bev,
            rollout.ego_history,
            rollout.c,
            deterministic=False,
            teacher_forcing_actions=rollout.actions,
        )
        logp_new = pol_new.log_probs
        logp_old = rollout.log_probs_old

        ratio = torch.exp(logp_new - logp_old)  # (B,T)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -(torch.min(unclipped, clipped)).mean()

        v_pred = self.critic(rollout.bev, rollout.ego_history, rollout.c)  # (B,)
        target_v = ret.mean(dim=1)  # (B,)

        if self.value_clip is not None:
            v_pred = torch.clamp(v_pred, -float(self.value_clip), float(self.value_clip))
            target_v = torch.clamp(target_v, -float(self.value_clip), float(self.value_clip))

        value_loss = ((v_pred - target_v) ** 2).mean()

        log_std = pol_new.log_std
        entropy_per_dim = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std
        entropy = entropy_per_dim.sum(dim=-1).mean()

        if self.consistency is None:
            consistency_loss = torch.tensor(0.0, device=policy_loss.device)
        else:
            consistency_loss = self.consistency(pol_new.actions).mean()

        total_loss = (
            self.lambda_policy * policy_loss
            + self.lambda_value * value_loss
            - self.lambda_entropy * entropy
            + self.lambda_consistency * consistency_loss
        )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "consistency_loss": consistency_loss,
            "total_loss": total_loss,
            "ratio": ratio.detach(),
        }

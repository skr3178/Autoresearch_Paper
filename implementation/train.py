"""End-to-end integration runner (Phase 4).

Deterministic integration proof that wires:
  data_loader -> mode_selector -> autoregressive_policy -> critic -> consistency -> PPO loss

We use a shaped imitation reward based on distance between predicted actions and
nuPlan ground-truth future trajectory.

Run with nuPlan devkit venv python.
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch

from implementation.data_loader import build_dataloader
from implementation.mode_selector import ModeSelector
from implementation.autoregressive_policy import AutoregressivePolicy
from implementation.critic import Critic
from implementation.consistency_module import ConsistencyModule
from implementation.ppo_loop import PPOTrainer


_NUPLAN_MINI_ROOT = "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini"
_NUPLAN_TRAIN_BOSTON_ROOT = "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/train_boston"
_NUPLAN_MAPS_ROOT = "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs() -> Dict[str, Dict[str, Any]]:
    base = {
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"split": "mini", "batch_size": 2, "num_workers": 0, "max_scenarios": 16},
        "bev": {"channels": 7, "h": 224, "w": 224},
        "timesteps": {"t_hist": 5, "t_future": 8},
        "modes": {"k": 6},
        "action": {"dim": 3},
        "model": {"hidden": 128},
        "ppo": {
            "clip_eps": 0.2,
            "value_coef": 3.0,
            "entropy_coef": 0.001,
            "consistency_coef": 0.0,
            "policy_coef": 1.0,
            "lr": 3e-4,
            "reward_scale": 5.0,
            "reward_clip": 10.0,
            "value_clip": 10.0,
            "normalize_adv": True,
        },
        "tests": {"n_steps": 10, "overfit_steps": 120, "overfit_lr": 3e-3},
    }

    debug = copy.deepcopy(base)
    debug["data"]["max_scenarios"] = 4
    debug["model"]["hidden"] = 64

    smoke = copy.deepcopy(base)
    smoke["data"]["batch_size"] = 4
    smoke["data"]["max_scenarios"] = 16
    smoke["model"]["hidden"] = 128

    full = copy.deepcopy(base)
    full["data"]["split"] = "train_boston"
    full["data"]["batch_size"] = 8
    full["data"]["max_scenarios"] = 128
    full["model"]["hidden"] = 256
    full["tests"]["n_steps"] = 50
    full["tests"]["overfit_steps"] = 200

    return {"debug": debug, "smoke": smoke, "full": full}


def _data_root_for_split(split: str) -> str:
    if split == "mini":
        return _NUPLAN_MINI_ROOT
    if split == "train_boston":
        return _NUPLAN_TRAIN_BOSTON_ROOT
    raise ValueError(split)


def build_components(cfg: Dict[str, Any]) -> Tuple[ModeSelector, AutoregressivePolicy, Critic, ConsistencyModule, PPOTrainer]:
    ms = ModeSelector(
        bev_channels=cfg["bev"]["channels"],
        num_modes=cfg["modes"]["k"],
        ego_hist_len=cfg["timesteps"]["t_hist"],
        hidden_dim=cfg["model"]["hidden"],
        side_task=False,
        side_task_T=cfg["timesteps"]["t_future"],
    )

    policy = AutoregressivePolicy(
        {
            "bev_channels": cfg["bev"]["channels"],
            "bev_h": cfg["bev"]["h"],
            "bev_w": cfg["bev"]["w"],
            "t_hist": cfg["timesteps"]["t_hist"],
            "t_future": cfg["timesteps"]["t_future"],
            "action_dim": cfg["action"]["dim"],
            "num_modes": cfg["modes"]["k"],
            "d_model": cfg["model"]["hidden"],
            "rnn_hidden": cfg["model"]["hidden"],
        }
    )

    critic = Critic(
        {
            "bev_channels": cfg["bev"]["channels"],
            "t_hist": cfg["timesteps"]["t_hist"],
            "num_modes": cfg["modes"]["k"],
            "d_model": cfg["model"]["hidden"],
            "critic_d_model": cfg["model"]["hidden"],
        }
    )

    consistency = ConsistencyModule(
        {
            "t_future": cfg["timesteps"]["t_future"],
            "action_dim": cfg["action"]["dim"],
            "jerk_weight": 1.0,
            "heading_acc_weight": 1.0,
        }
    )

    trainer = PPOTrainer(
        cfg={
            "ppo_clip_eps": cfg["ppo"]["clip_eps"],
            "gamma": 0.9,
            "gae_lambda": 0.95,
            "lambda_policy": cfg["ppo"]["policy_coef"],
            "lambda_value": cfg["ppo"]["value_coef"],
            "lambda_entropy": cfg["ppo"]["entropy_coef"],
            "lambda_consistency": cfg["ppo"]["consistency_coef"],
            "normalize_adv": cfg["ppo"].get("normalize_adv", True),
            "reward_scale": cfg["ppo"].get("reward_scale", 1.0),
            "reward_clip": cfg["ppo"].get("reward_clip", None),
            "value_clip": cfg["ppo"].get("value_clip", None),
        },
        policy=policy,
        critic=critic,
        consistency=consistency,
    )

    return ms, policy, critic, consistency, trainer


@torch.no_grad()
def infer_modes(ms: ModeSelector, batch: Dict[str, Any]) -> torch.Tensor:
    return ms(batch["bev"], batch["ego_history"]).c


def _build_dl(cfg: Dict[str, Any], *, shuffle: bool):
    return build_dataloader(
        data_root=_data_root_for_split(cfg["data"]["split"]),
        map_root=_NUPLAN_MAPS_ROOT,
        map_version="1.0",
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        num_scenarios=cfg["data"]["max_scenarios"],
        num_workers=cfg["data"]["num_workers"],
    )


def _make_optimizer(cfg: Dict[str, Any], policy: torch.nn.Module, critic: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=cfg["ppo"]["lr"])


def _shaped_rewards_from_gt(actions: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    d_xy = torch.norm(actions[..., :2] - gt_traj[..., :2], dim=-1)
    d_h = torch.abs(actions[..., 2] - gt_traj[..., 2])
    return -(d_xy + 0.1 * d_h)


def single_step_test(cfg: Dict[str, Any]) -> float:
    print("== single-step test ==")
    device = torch.device(cfg["device"])
    dl = _build_dl(cfg, shuffle=False)

    ms, policy, critic, _consistency, trainer = build_components(cfg)
    ms.to(device).eval()
    policy.to(device).train()
    critic.to(device).train()
    optim = _make_optimizer(cfg, policy, critic)

    batch = next(iter(dl))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        c = infer_modes(ms, batch)
        pol_out = policy(batch["bev"], batch["ego_history"], c, deterministic=True)
        rewards = _shaped_rewards_from_gt(pol_out.actions, batch["gt_trajectory"])

    rollout = trainer.collect_rollout({"bev": batch["bev"], "ego_history": batch["ego_history"], "c": c}, rewards=rewards)
    losses = trainer.update(rollout)

    optim.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    optim.step()

    for k, v in losses.items():
        if torch.is_tensor(v) and v.dim() == 0:
            print(f"{k}: {float(v.detach().cpu())}")

    return float(losses["total_loss"].detach().cpu())


def n_step_test(cfg: Dict[str, Any]) -> Tuple[float, float]:
    print("== N-step test ==")
    device = torch.device(cfg["device"])
    dl = _build_dl(cfg, shuffle=True)

    ms, policy, critic, _consistency, trainer = build_components(cfg)
    ms.to(device).eval()
    policy.to(device).train()
    critic.to(device).train()
    optim = _make_optimizer(cfg, policy, critic)

    it = iter(dl)
    losses_hist = []
    for step in range(cfg["tests"]["n_steps"]):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with torch.no_grad():
            c = infer_modes(ms, batch)
            pol_out = policy(batch["bev"], batch["ego_history"], c, deterministic=True)
            rewards = _shaped_rewards_from_gt(pol_out.actions, batch["gt_trajectory"])

        rollout = trainer.collect_rollout({"bev": batch["bev"], "ego_history": batch["ego_history"], "c": c}, rewards=rewards)
        losses = trainer.update(rollout)

        optim.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optim.step()

        total = float(losses["total_loss"].detach().cpu())
        losses_hist.append(total)
        print(
            f"step={step} loss={total:.6f} policy={float(losses['policy_loss']):.6f} "
            f"value={float(losses['value_loss']):.6f} entropy={float(losses['entropy']):.6f} "
            f"cons={float(losses['consistency_loss']):.6f}"
        )

    first = losses_hist[0]
    last = losses_hist[-1]
    print("first_total_loss:", first)
    print("last_total_loss: ", last)

    ref = float(np.mean(losses_hist[: min(3, len(losses_hist))]))
    if not (last < ref):
        raise AssertionError(f"Loss did not decrease: last={last} ref(mean first steps)={ref}")

    return first, last


def tiny_overfit_test(cfg: Dict[str, Any]) -> Tuple[float, float]:
    print("== tiny overfit test ==")
    device = torch.device(cfg["device"])

    overfit_cfg = copy.deepcopy(cfg)
    overfit_cfg["data"]["max_scenarios"] = max(2, cfg["data"]["batch_size"])
    dl = _build_dl(overfit_cfg, shuffle=False)

    ms, policy, critic, _consistency, trainer = build_components(cfg)
    ms.to(device).eval()
    policy.to(device).train()
    critic.to(device).train()

    overfit_cfg2 = copy.deepcopy(cfg)
    overfit_cfg2["ppo"]["lr"] = cfg["tests"]["overfit_lr"]
    optim = _make_optimizer(overfit_cfg2, policy, critic)

    batch = next(iter(dl))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        c = infer_modes(ms, batch)

    first = None
    last = None
    for step in range(cfg["tests"]["overfit_steps"]):
        with torch.no_grad():
            pol_out = policy(batch["bev"], batch["ego_history"], c, deterministic=True)
            rewards = _shaped_rewards_from_gt(pol_out.actions, batch["gt_trajectory"])

        rollout = trainer.collect_rollout({"bev": batch["bev"], "ego_history": batch["ego_history"], "c": c}, rewards=rewards)
        losses = trainer.update(rollout)

        optim.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optim.step()

        total = float(losses["total_loss"].detach().cpu())
        if first is None:
            first = total
        last = total
        if step % 20 == 0:
            print(f"overfit_step={step} total_loss={total:.6f}")

    print("overfit_first_total_loss:", first)
    print("overfit_last_total_loss: ", last)

    if not (last < 1.0):
        raise AssertionError(f"Overfit test failed: final loss not < 1.0 (got {last})")

    return float(first), float(last)


def reproducibility_check(cfg: Dict[str, Any]) -> None:
    print("== reproducibility check ==")
    device = torch.device(cfg["device"])

    rep_cfg = copy.deepcopy(cfg)
    rep_cfg["data"]["max_scenarios"] = max(2, cfg["data"]["batch_size"])
    dl = _build_dl(rep_cfg, shuffle=False)
    batch = next(iter(dl))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def run_once() -> Dict[str, float]:
        set_determinism(cfg["seed"])
        ms, policy, critic, _consistency, trainer = build_components(cfg)
        ms.to(device).eval()
        policy.to(device).train()
        critic.to(device).train()
        optim = _make_optimizer(cfg, policy, critic)
        with torch.no_grad():
            c = infer_modes(ms, batch)
            pol_out = policy(batch["bev"], batch["ego_history"], c, deterministic=True)
            rewards = _shaped_rewards_from_gt(pol_out.actions, batch["gt_trajectory"])
        rollout = trainer.collect_rollout({"bev": batch["bev"], "ego_history": batch["ego_history"], "c": c}, rewards=rewards)
        losses = trainer.update(rollout)
        optim.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optim.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items() if torch.is_tensor(v) and v.dim() == 0}

    out1 = run_once()
    out2 = run_once()

    for k in out1.keys():
        if abs(out1[k] - out2[k]) > 0.0:
            raise AssertionError(f"Non-deterministic for key={k}: {out1[k]} vs {out2[k]}")
    print("reproducibility: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="debug", choices=["debug", "smoke", "full"])
    args = parser.parse_args()

    cfg = get_configs()[args.config]
    set_determinism(cfg["seed"])

    _ = single_step_test(cfg)
    _ = n_step_test(cfg)
    _overfit_first, overfit_last = tiny_overfit_test(cfg)
    reproducibility_check(cfg)

    print(f"metric: {overfit_last}")


if __name__ == "__main__":
    main()

"""End-to-end training entrypoint (Phase 4 Integration Proof).

Wires all Phase-3 submodules into a minimal end-to-end training harness.
This is an integration proof runner (not a full nuPlan benchmark).
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch

from implementation.data_loader import build_dataloader
from implementation.transition_model import TransitionModel
from implementation.mode_selector import ModeSelector
from implementation.autoregressive_policy import AutoregressivePolicy
from implementation.consistency_module import ConsistencyModule
from implementation.critic import Critic
from implementation.ppo_loop import PPOTrainer


DATA_ROOTS = {
    "mini": "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini",
    "train_boston": "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/train_boston",
}
MAP_ROOT = "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0"
MAP_VERSION = "1.0"


def _base_config() -> Dict[str, Any]:
    return {
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"split": "mini", "batch_size": 2, "num_workers": 0, "max_scenarios": 8},
        "bev": {"C": 7, "H": 224, "W": 224},
        "T_hist": 5,
        "T_future": 8,
        "K_modes": 6,
        "action_dim": 3,
        "model": {"hidden_dim": 128},
        "ppo": {
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "consistency_coef": 0.1,
            "lr": 3e-4,
            "grad_clip": 1.0,
        },
        "train": {"steps": 10, "overfit_steps": 200, "log_every": 1},
    }


def get_config(name: str) -> Dict[str, Any]:
    cfg = _base_config()
    if name == "debug":
        cfg["data"].update({"batch_size": 2, "num_workers": 0, "max_scenarios": 4})
        cfg["model"].update({"hidden_dim": 64})
        cfg["train"].update({"steps": 1, "overfit_steps": 50})
        return cfg
    if name == "smoke":
        cfg["data"].update({"batch_size": 4, "num_workers": 0, "max_scenarios": 16})
        cfg["model"].update({"hidden_dim": 128})
        cfg["train"].update({"steps": 10, "overfit_steps": 200})
        return cfg
    if name == "full":
        cfg["data"].update({"split": "train_boston", "batch_size": 8, "num_workers": 2, "max_scenarios": 256})
        cfg["model"].update({"hidden_dim": 256})
        cfg["train"].update({"steps": 100, "overfit_steps": 400})
        return cfg
    raise ValueError(f"Unknown config: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _flat_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bev_channels": int(cfg["bev"]["C"]),
        "bev_h": int(cfg["bev"]["H"]),
        "bev_w": int(cfg["bev"]["W"]),
        "t_hist": int(cfg["T_hist"]),
        "t_future": int(cfg["T_future"]),
        "action_dim": int(cfg["action_dim"]),
        "num_modes": int(cfg["K_modes"]),
        "d_model": int(cfg["model"]["hidden_dim"]),
        "rnn_hidden": int(cfg["model"]["hidden_dim"]),
        "critic_d_model": int(cfg["model"]["hidden_dim"]),
    }


def _consistency_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "t_future": int(cfg["T_future"]),
        "action_dim": int(cfg["action_dim"]),
        "jerk_weight": 1.0,
        "heading_acc_weight": 1.0,
        "consistency_eps": 1e-8,
    }


def build_system(cfg: Dict[str, Any]):
    device = torch.device(cfg["device"])

    bevC = int(cfg["bev"]["C"])
    bevH = int(cfg["bev"]["H"])
    bevW = int(cfg["bev"]["W"])
    action_dim = int(cfg["action_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])

    _ = TransitionModel(bevC, bevH, bevW, action_dim, hidden_dim=hidden_dim).to(device)

    mode_selector = ModeSelector(
        bev_channels=bevC,
        num_modes=int(cfg["K_modes"]),
        ego_hist_len=int(cfg["T_hist"]),
        hidden_dim=hidden_dim,
        side_task=False,
        side_task_T=int(cfg["T_future"]),
    ).to(device)

    flat = _flat_model_cfg(cfg)
    policy = AutoregressivePolicy(flat).to(device)
    critic = Critic(flat).to(device)
    consistency = ConsistencyModule(_consistency_cfg(cfg)).to(device)

    trainer = PPOTrainer(
        cfg={
            "ppo_clip_eps": float(cfg["ppo"]["clip_eps"]),
            "lambda_policy": 1.0,
            "lambda_value": float(cfg["ppo"]["value_coef"]),
            "lambda_entropy": float(cfg["ppo"]["entropy_coef"]),
            "lambda_consistency": float(cfg["ppo"]["consistency_coef"]),
            "normalize_adv": True,
            "gamma": 0.1,
            "gae_lambda": 0.9,
        },
        policy=policy,
        critic=critic,
        consistency=consistency,
    )

    opt = torch.optim.AdamW(
        list(policy.parameters()) + list(critic.parameters()),
        lr=float(cfg["ppo"]["lr"]),
        weight_decay=1e-4,
    )

    return {"trainer": trainer, "mode_selector": mode_selector, "opt": opt, "device": device}


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def update_step(system: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    device = system["device"]
    batch = _batch_to_device(batch, device)

    with torch.no_grad():
        ms_out = system["mode_selector"](batch["bev"], batch["ego_history"])
        c = ms_out.c

    rollout = system["trainer"].collect_rollout({"bev": batch["bev"], "ego_history": batch["ego_history"], "c": c})
    losses = system["trainer"].update(rollout)

    system["opt"].zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        list(system["trainer"].policy.parameters()) + list(system["trainer"].critic.parameters()),
        max_norm=float(1.0),
    )
    system["opt"].step()

    return {k: v for k, v in losses.items() if k in {"policy_loss", "value_loss", "entropy", "consistency_loss", "total_loss"}}


def _build_dl(cfg: Dict[str, Any], shuffle: bool) -> torch.utils.data.DataLoader:
    split = cfg["data"]["split"]
    return build_dataloader(
        data_root=DATA_ROOTS[split],
        map_root=MAP_ROOT,
        map_version=MAP_VERSION,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=shuffle,
        num_scenarios=int(cfg["data"]["max_scenarios"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )


def run_single_step(cfg: Dict[str, Any]) -> Dict[str, float]:
    set_seed(cfg["seed"])
    system = build_system(cfg)
    dl = _build_dl(cfg, shuffle=False)
    batch = next(iter(dl))
    out = update_step(system, batch)
    assert torch.isfinite(out["total_loss"]).item()
    return {k: float(v.detach().cpu().item()) for k, v in out.items()}


def run_n_steps(cfg: Dict[str, Any], steps: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    set_seed(cfg["seed"])
    system = build_system(cfg)
    dl = _build_dl(cfg, shuffle=True)
    it = iter(dl)
    first = None
    last = None
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        out = update_step(system, batch)
        metrics = {k: float(v.detach().cpu().item()) for k, v in out.items()}
        if first is None:
            first = metrics
        last = metrics
        if step % cfg["train"]["log_every"] == 0:
            print(
                f"step={step} loss={metrics['total_loss']:.6f} "
                f"policy={metrics['policy_loss']:.6f} value={metrics['value_loss']:.6f} "
                f"entropy={metrics['entropy']:.6f} cons={metrics['consistency_loss']:.6f}"
            )
    return first, last


def _merge_batches(batches):
    merged = {}
    keys = batches[0].keys()
    for k in keys:
        v0 = batches[0][k]
        if torch.is_tensor(v0):
            merged[k] = torch.cat([b[k] for b in batches], dim=0)
        elif isinstance(v0, list):
            merged[k] = sum([b[k] for b in batches], [])
        else:
            merged[k] = [b[k] for b in batches]
    return merged


def run_tiny_overfit(cfg: Dict[str, Any], n_samples: int = 2) -> Tuple[float, float]:
    set_seed(cfg["seed"])
    system = build_system(cfg)
    dl = _build_dl(cfg, shuffle=False)
    it = iter(dl)
    batches = [next(it) for _ in range(n_samples)]
    merged = _merge_batches(batches)

    first_loss = None
    last_loss = None
    for step in range(cfg["train"]["overfit_steps"]):
        out = update_step(system, merged)
        loss = float(out["total_loss"].detach().cpu().item())
        if first_loss is None:
            first_loss = loss
        last_loss = loss
        if step % 20 == 0:
            print(f"overfit_step={step} total_loss={loss:.6f}")
    return first_loss, last_loss


def reproducibility_check(cfg: Dict[str, Any]) -> None:
    m1 = run_single_step(cfg)
    m2 = run_single_step(cfg)
    for k in m1:
        if m1[k] != m2[k]:
            raise AssertionError(f"Non-deterministic {k}: {m1[k]} vs {m2[k]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="debug", choices=["debug", "smoke", "full"])
    ap.add_argument("--print_config", action="store_true")
    args = ap.parse_args()

    cfg = get_config(args.config)
    if args.print_config:
        print(json.dumps(cfg, indent=2))

    print("== single-step test ==")
    m = run_single_step(cfg)
    for k, v in m.items():
        print(f"{k}: {v}")

    if cfg["train"]["steps"] >= 2:
        print("== N-step test ==")
        first, last = run_n_steps(cfg, cfg["train"]["steps"])
        print(f"first_total_loss: {first['total_loss']}")
        print(f"last_total_loss:  {last['total_loss']}")
        if not (last["total_loss"] < first["total_loss"]):
            raise AssertionError("Loss did not decrease")

    print("== tiny overfit test ==")
    f, l = run_tiny_overfit(cfg, n_samples=2)
    print(f"overfit_first_total_loss: {f}")
    print(f"overfit_last_total_loss:  {l}")
    if not (l < 0.5 * f):
        raise AssertionError("Overfit did not reduce loss by 50%")

    print("== reproducibility check ==")
    reproducibility_check(cfg)
    print("reproducibility: OK")

    print(f"metric: {l}")


if __name__ == "__main__":
    main()

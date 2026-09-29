#!/usr/bin/env python3
"""
Offline renderer for reward-ablation 2D trajectory plots and GIFs.

It evaluates each saved final checkpoint once, then writes:
  - figures/trajectories_2d.png
  - figures/trajectories_2d.gif
into the latest log directory for each experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.shared.quant.config import Config
from du_iibtd_based_fading_delta.MAPPO_quant.networks import MAPPOPolicy
from du_iibtd_based_fading_delta.shared.quant.runner import make_env_factory
from du_iibtd_based_fading_delta.shared.quant.utils import evaluate_policy
from du_iibtd_based_fading_delta.shared.quant.visualize import (
    _apply_config_overrides,
    plot_planner_trajectory_2d,
    plot_planner_trajectory_2d_gif,
)


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT_DIR / "checkpoints" / "reward_ablation"
LOG_ROOT = ROOT_DIR / "logs" / "reward_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reward-ablation 2D GIFs")
    parser.add_argument("--run_tag", type=str, default=None, help="Reward-ablation run tag, e.g. 20260316_223639")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiments, e.g. full_reward tx_only",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Evaluation device")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional saved config.json; required only when the run has no config snapshot.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Max eval steps; default uses config")
    parser.add_argument("--fps", type=int, default=4, help="GIF frames per second")
    return parser.parse_args()


def latest_run_tag() -> str:
    run_dirs = sorted([p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No reward-ablation checkpoint runs found under {CHECKPOINT_ROOT}")
    return run_dirs[-1].name


def resolve_experiments(run_tag: str, selected: Iterable[str] | None) -> List[str]:
    run_root = CHECKPOINT_ROOT / run_tag
    if not run_root.exists():
        raise FileNotFoundError(f"Checkpoint run tag not found: {run_root}")
    all_experiments = sorted([p.name for p in run_root.iterdir() if p.is_dir()])
    if selected is None:
        return all_experiments

    selected = list(selected)
    missing = [name for name in selected if name not in all_experiments]
    if missing:
        raise FileNotFoundError(f"Unknown experiments under {run_root}: {missing}")
    return selected


def find_output_log_dir(run_tag: str, experiment: str) -> Path:
    exp_log_root = LOG_ROOT / run_tag / experiment
    metrics_candidates = sorted(exp_log_root.glob("*/metrics.json"))
    if metrics_candidates:
        return metrics_candidates[-1].parent

    offline_dir = exp_log_root / f"offline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    offline_dir.mkdir(parents=True, exist_ok=True)
    return offline_dir


def load_run_config(
    run_tag: str,
    experiment: str,
    device: str,
    config_path: Path | None,
) -> Config:
    """Hydrate the environment from the checkpoint's saved scene configuration."""
    if config_path is None:
        candidates = sorted((LOG_ROOT / run_tag / experiment).glob("*/config.json"))
        if not candidates:
            raise FileNotFoundError(
                "No saved config.json was found for "
                f"{run_tag}/{experiment}. Refusing to render with Config() defaults; "
                "pass --config_path to the original run configuration."
            )
        config_path = candidates[-1]

    with config_path.open("r") as f:
        raw_config = json.load(f)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")

    config = Config()
    _apply_config_overrides(config, raw_config)
    config.training.device = device
    config.__post_init__()
    return config


def render_one(
    run_tag: str,
    experiment: str,
    device: str,
    max_steps: int | None,
    fps: int,
    config_path: Path | None,
) -> None:
    checkpoint_path = CHECKPOINT_ROOT / run_tag / experiment / "final_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_run_config(run_tag, experiment, device, config_path)
    if max_steps is None:
        max_steps = int(config.training.episode_max_steps)

    env = make_env_factory(config)()
    try:
        policy = MAPPOPolicy(env.get_obs_dims(), env.get_action_dims(), config.training)
        policy.load(str(checkpoint_path))

        eval_results = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=1,
            max_steps=max_steps,
        )

        uav_traj = np.asarray(eval_results["eval_uav_trajectory"], dtype=float)
        ugv_traj = np.asarray(eval_results["eval_ugv_trajectory"], dtype=float)
        step_details = eval_results["eval_step_details"]
        occupancy = env.scene.get_occupancy_grid()

        output_dir = find_output_log_dir(run_tag, experiment)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        title = f"{experiment} 2D Trajectory"
        plot_planner_trajectory_2d(
            uav_trajectory=uav_traj,
            ugv_trajectory=ugv_traj,
            step_details=step_details,
            grid_size=config.scene.grid_size,
            occupancy=occupancy,
            save_path=str(fig_dir / "trajectories_2d.png"),
            title=title,
        )
        plot_planner_trajectory_2d_gif(
            uav_trajectory=uav_traj,
            ugv_trajectory=ugv_traj,
            step_details=step_details,
            grid_size=config.scene.grid_size,
            occupancy=occupancy,
            save_path=str(fig_dir / "trajectories_2d.gif"),
            title=title,
            fps=fps,
        )
        print(f"[Render] {experiment}: saved 2D plot + GIF to {fig_dir}")
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or latest_run_tag()
    experiments = resolve_experiments(run_tag, args.experiments)

    print(f"[Render] run_tag={run_tag}")
    print(f"[Render] experiments={experiments}")

    for experiment in experiments:
        render_one(
            run_tag=run_tag,
            experiment=experiment,
            device=args.device,
            max_steps=args.max_steps,
            fps=args.fps,
            config_path=args.config_path,
        )


if __name__ == "__main__":
    main()

"""Run greedy UAV path planning in the UAV-UGV active sensing environment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional


import numpy as np

import environment as env_module
from config import Config
from greedy_policy import GreedyPathPolicy, GreedyPolicyConfig
from sim_models import GridScene, IIBTD_opt, SimDataGen
from utils import MetricsLogger, evaluate_policy, json_default, print_config_summary, set_seeds
from du_iibtd_based.scene_suite import (
    apply_scene_cli_overrides,
    EVAL_SCENE_SEED_STRIDE,
    aggregate_eval_results,
    clone_config_for_scene,
    configure_scene_suite_config,
    format_scene_suite,
)


DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _parse_bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _parse_float_list(value: str):
    return [float(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Greedy UAV-UGV active sensing")

    parser.add_argument("--eval_episodes", type=int, default=defaults.mappo.eval_episodes)
    parser.add_argument("--episode_max_steps", type=int, default=defaults.mappo.episode_max_steps)
    parser.add_argument("--seed", type=int, default=defaults.mappo.seed)
    parser.add_argument("--device", type=str, default=defaults.mappo.device)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)

    parser.add_argument("--radioseer_root", type=str, default=defaults.scene.radioseer_root)
    parser.add_argument(
        "--radioseer_sample_index",
        type=int,
        default=None,
        help="Single-scene override; mutually exclusive with --radioseer_scene_indices.",
    )
    parser.add_argument(
        "--radioseer_scene_indices",
        type=str,
        default=None,
        help="Comma-separated DPM100PSD manifest rows used as the shared evaluation suite.",
    )
    parser.add_argument("--uav_height", type=float, default=defaults.scene.uav_height)
    parser.add_argument("--ugv_height", type=float, default=defaults.scene.ugv_height)
    parser.add_argument("--building_height_m", type=float, default=defaults.scene.building_height_m)

    parser.add_argument("--uav_step_size", type=float, default=defaults.uav.step_size)
    parser.add_argument("--ugv_step_size", type=float, default=defaults.ugv.step_size)
    parser.add_argument("--bandwidth_ratios", type=str, default=",".join(str(v) for v in defaults.uav.bandwidth_ratios))
    parser.add_argument("--uav_default_bw_ratio", type=float, default=defaults.uav.default_bw_ratio)
    parser.add_argument("--queue_capacity_packets", type=int, default=defaults.uav.queue_capacity_packets)

    parser.add_argument("--planner_target_mode", type=str, default=defaults.planner.target_mode)
    parser.add_argument("--initial_observation_mode", type=str, default=defaults.planner.initial_observation_mode)
    parser.add_argument("--local_planner_radius", type=int, default=defaults.planner.local_planner_radius)
    parser.add_argument("--ensemble_refresh_interval", type=int, default=defaults.planner.ensemble_refresh_interval)
    parser.add_argument("--ensemble_full_refresh_interval", type=int, default=defaults.planner.ensemble_full_refresh_interval)
    parser.add_argument("--nmse_refresh_delta", type=float, default=defaults.planner.nmse_refresh_delta)
    parser.add_argument("--incremental_outer_iters", type=int, default=defaults.planner.incremental_outer_iters)
    parser.add_argument("--incremental_max_svt_iters", type=int, default=defaults.planner.incremental_max_svt_iters)
    parser.add_argument("--ensemble_kernel_bandwidth_mode", type=str, default=defaults.planner.ensemble_kernel_bandwidth_mode, choices=["fixed", "same", "base_pm_delta", "pm_delta", "plus_minus"])
    parser.add_argument("--ensemble_kernel_bandwidth_delta", type=float, default=defaults.planner.ensemble_kernel_bandwidth_delta)
    parser.add_argument("--ensemble_init_jitter_scale", type=float, default=defaults.planner.ensemble_init_jitter_scale)
    parser.add_argument("--ensemble_quality_weighted", type=_parse_bool_arg, default=defaults.planner.ensemble_quality_weighted)
    parser.add_argument("--prefill_percent", type=float, default=defaults.planner.prefill_percent)
    parser.add_argument("--prefill_budget_basis", type=int, default=defaults.planner.prefill_budget_basis)
    parser.add_argument("--init_building_clearance", type=int, default=defaults.planner.init_building_clearance)
    parser.add_argument("--bootstrap_building_clearance", type=int, default=defaults.planner.bootstrap_building_clearance)
    parser.add_argument("--flush_reconstruction_on_episode_end", type=_parse_bool_arg, default=defaults.planner.flush_reconstruction_on_episode_end)
    parser.add_argument("--iibtd_mu", type=float, default=defaults.planner.iibtd_mu)
    parser.add_argument("--iibtd_nu", type=float, default=defaults.planner.iibtd_nu)
    parser.add_argument("--iibtd_kernel_bandwidth", type=float, default=defaults.planner.iibtd_kernel_bandwidth)
    parser.add_argument("--iibtd_backend", type=str, default=defaults.planner.iibtd_backend)
    parser.add_argument("--iibtd_device", type=str, default=defaults.planner.iibtd_device)
    parser.add_argument("--iibtd_gpu_phi_solver", type=str, default=defaults.planner.iibtd_gpu_phi_solver)
    parser.add_argument(
        "--du_iibtd_checkpoints",
        nargs="+",
        default=None,
        help=(
            "Optional residual-Sr learn-nu checkpoint list used when "
            "--iibtd_backend is du_iibtd_res_sr or du_iibtd_res_sr_learn_nu."
        ),
    )

    parser.add_argument("--accuracy_target_nmse", type=float, default=defaults.reward.accuracy_target_nmse)
    parser.add_argument("--terminal_failure_penalty", type=float, default=defaults.reward.terminal_failure_penalty)

    parser.add_argument("--greedy_beam_width", type=int, default=512)
    parser.add_argument("--greedy_max_extra_actions", type=int, default=2)
    parser.add_argument("--greedy_revisit_penalty", type=float, default=0.25)
    parser.add_argument("--greedy_sampled_revisit_penalty", type=float, default=0.15)
    parser.add_argument("--greedy_progress_weight", type=float, default=0.05)
    parser.add_argument("--greedy_arrival_bonus", type=float, default=1.0)
    parser.add_argument("--greedy_building_uncertainty_scale", type=float, default=0.0)
    parser.add_argument("--greedy_score_traversed_cells", type=_parse_bool_arg, default=True)

    return parser.parse_args()

def build_config(args: argparse.Namespace) -> Config:
    config = Config()

    config.mappo.seed = int(args.seed)
    config.mappo.device = str(args.device)
    config.mappo.episode_max_steps = int(args.episode_max_steps)
    config.mappo.eval_episodes = int(args.eval_episodes)
    config.mappo.log_dir = str(args.log_dir)
    config.mappo.num_envs = 1
    config.mappo.vec_backend = "sync"

    apply_scene_cli_overrides(config, args)
    config.scene.uav_height = float(args.uav_height)
    config.scene.ugv_height = float(args.ugv_height)
    config.scene.building_height_m = float(args.building_height_m)

    config.uav.step_size = float(args.uav_step_size)
    config.ugv.step_size = float(args.ugv_step_size)
    config.uav.bandwidth_ratios = _parse_float_list(args.bandwidth_ratios)
    config.uav.default_bw_ratio = float(args.uav_default_bw_ratio)
    config.uav.queue_capacity_packets = int(args.queue_capacity_packets)

    config.planner.target_mode = str(args.planner_target_mode).strip().lower()
    config.planner.initial_observation_mode = str(args.initial_observation_mode).strip().lower()
    config.planner.local_planner_radius = max(1, int(args.local_planner_radius))
    config.planner.ensemble_refresh_interval = max(1, int(args.ensemble_refresh_interval))
    config.planner.ensemble_full_refresh_interval = max(0, int(args.ensemble_full_refresh_interval))
    config.planner.nmse_refresh_delta = max(0.0, float(args.nmse_refresh_delta))
    config.planner.incremental_outer_iters = max(1, int(args.incremental_outer_iters))
    config.planner.incremental_max_svt_iters = max(1, int(args.incremental_max_svt_iters))
    config.planner.ensemble_kernel_bandwidth_mode = (
        str(args.ensemble_kernel_bandwidth_mode).strip().lower()
        or config.planner.ensemble_kernel_bandwidth_mode
    )
    config.planner.ensemble_kernel_bandwidth_delta = abs(float(args.ensemble_kernel_bandwidth_delta))
    config.planner.ensemble_init_jitter_scale = max(0.0, float(args.ensemble_init_jitter_scale))
    config.planner.ensemble_quality_weighted = bool(args.ensemble_quality_weighted)
    config.planner.prefill_percent = float(args.prefill_percent)
    config.planner.prefill_budget_basis = max(0, int(args.prefill_budget_basis))
    config.planner.init_building_clearance = max(0, int(args.init_building_clearance))
    config.planner.bootstrap_building_clearance = max(0, int(args.bootstrap_building_clearance))
    config.planner.flush_reconstruction_on_episode_end = bool(args.flush_reconstruction_on_episode_end)
    config.planner.iibtd_mu = float(args.iibtd_mu)
    config.planner.iibtd_nu = float(args.iibtd_nu)
    config.planner.iibtd_kernel_bandwidth = float(args.iibtd_kernel_bandwidth)
    config.planner.iibtd_backend = str(args.iibtd_backend).strip().lower()
    config.planner.iibtd_device = str(args.iibtd_device).strip()
    config.planner.iibtd_gpu_phi_solver = str(args.iibtd_gpu_phi_solver).strip().lower()
    if args.du_iibtd_checkpoints is not None:
        config.planner.du_iibtd_checkpoints = [
            str(path).strip()
            for path in args.du_iibtd_checkpoints
            if str(path).strip()
        ]

    config.reward.accuracy_target_nmse = float(args.accuracy_target_nmse)
    config.reward.terminal_failure_penalty = float(args.terminal_failure_penalty)

    config.__post_init__()
    return config


def build_greedy_config(args: argparse.Namespace) -> GreedyPolicyConfig:
    return GreedyPolicyConfig(
        beam_width=max(1, int(args.greedy_beam_width)),
        max_extra_actions=max(0, int(args.greedy_max_extra_actions)),
        revisit_penalty=max(0.0, float(args.greedy_revisit_penalty)),
        sampled_revisit_penalty=max(0.0, float(args.greedy_sampled_revisit_penalty)),
        progress_weight=max(0.0, float(args.greedy_progress_weight)),
        arrival_bonus=float(args.greedy_arrival_bonus),
        building_uncertainty_scale=float(np.clip(args.greedy_building_uncertainty_scale, 0.0, 1.0)),
        score_traversed_cells=bool(args.greedy_score_traversed_cells),
    )


def make_scene_shared_env_data(config: Config, sample_index: Optional[int] = None) -> Dict:
    scene_config = clone_config_for_scene(config, sample_index) if sample_index is not None else config
    sim_data = SimDataGen(config=scene_config, seed=int(scene_config.mappo.seed))
    return sim_data.export_data()


def sync_scene_grid_size_from_shared_data(config: Config, shared_data: Dict) -> None:
    """Keep logged scene config aligned with the actual dataset-backed grid."""
    if "H" not in shared_data:
        raise KeyError("shared_data must contain H to resolve scene.grid_size")
    h_shape = np.asarray(shared_data["H"]).shape
    if len(h_shape) < 2:
        raise ValueError(f"shared_data['H'] must have at least 2 dimensions, got shape {h_shape!r}")
    config.scene.grid_size = (int(h_shape[0]), int(h_shape[1]))


def make_env(config: Config, shared_data: Optional[Dict] = None, env_idx: int = 0):
    sim_data = SimDataGen(
        config=config,
        seed=int(config.mappo.seed + int(env_idx)),
        precomputed_data=shared_data,
    )
    td = IIBTD_opt(
        config=config,
        grid_coords=sim_data.grid_coords,
        bounds=sim_data.bounds,
        i_mask=sim_data.I_mask,
        n_sources=1,
    )
    return env_module.UAVUGVEnvironment(
        config=config,
        tensor_decomp=td,
        sim_data=sim_data,
        scene_map=GridScene(
            config,
            occupancy_grid=sim_data.get_building_mask(),
            building_heights=sim_data.get_building_heights(),
        ),
    )


def build_run_metadata(
    config: Config,
    greedy_config: GreedyPolicyConfig,
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "config": asdict(config),
        "greedy_config": asdict(greedy_config),
        "run_info": {
            "argv": list(os.sys.argv),
            "cwd": os.getcwd(),
        },
        "cli_args": vars(args).copy(),
    }


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)
    return path


def save_outputs(
    logger: MetricsLogger,
    target_dir: str,
    metadata: Dict[str, object],
) -> None:
    write_json(os.path.join(target_dir, "config.json"), metadata["config"])
    write_json(os.path.join(target_dir, "greedy_config.json"), metadata["greedy_config"])
    write_json(os.path.join(target_dir, "cli_args.json"), metadata["cli_args"])
    write_json(os.path.join(target_dir, "run_info.json"), metadata["run_info"])
    metrics_path = logger.save(os.path.join(target_dir, "metrics.json"))
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    greedy_config = build_greedy_config(args)
    set_seeds(config.mappo.seed)

    scene_indices = configure_scene_suite_config(config)
    shared_data_suite = [
        make_scene_shared_env_data(config, sample_index=sample_index)
        for sample_index in scene_indices
    ]
    sync_scene_grid_size_from_shared_data(config, shared_data_suite[0])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = os.path.join(config.mappo.log_dir, timestamp)
    os.makedirs(target_dir, exist_ok=True)
    metadata = build_run_metadata(config, greedy_config, args)
    metadata.setdefault("config", {}).setdefault("scene", {})[
        "radioseer_scene_indices"
    ] = list(scene_indices)
    logger = MetricsLogger(target_dir, metadata=metadata)

    print_config_summary(config)
    print(f"[Scene Suite] DPM100PSD manifest rows: {format_scene_suite(scene_indices)}")
    print("\n--- Greedy UAV Planner ---")
    print(f"  Beam width:             {greedy_config.beam_width}")
    print(f"  Max extra actions:      {greedy_config.max_extra_actions}")
    print(f"  Score traversed cells:  {greedy_config.score_traversed_cells}")
    print(f"  Revisit penalty:        {greedy_config.revisit_penalty}")
    print(f"  Sampled revisit penalty:{greedy_config.sampled_revisit_penalty}")
    print("  BW selection:           uncertainty-ranked bins")
    print("-" * 60 + "\n")

    eval_envs = []

    try:
        scene_results = []
        for scene_pos, (sample_index, shared_data) in enumerate(zip(scene_indices, shared_data_suite)):
            scene_config = clone_config_for_scene(config, sample_index)
            sync_scene_grid_size_from_shared_data(scene_config, shared_data)
            env = make_env(config=scene_config, shared_data=shared_data, env_idx=scene_pos)
            eval_envs.append(env)
            policy = GreedyPathPolicy(config=greedy_config, env=env)
            scene_results.append(
                evaluate_policy(
                    env=env,
                    policy=policy,
                    num_episodes=int(config.mappo.eval_episodes),
                    max_steps=int(args.episode_max_steps),
                    seed_base=int(config.mappo.seed) + scene_pos * EVAL_SCENE_SEED_STRIDE,
                )
            )
        results = aggregate_eval_results(scene_results, seed_base=int(config.mappo.seed))
        logger.log_evaluation(results)
        save_outputs(
            logger=logger,
            target_dir=target_dir,
            metadata=metadata,
        )
        print(
            "[Greedy] Done | "
            f"episodes={int(config.mappo.eval_episodes)} | "
            f"mean_return={float(results.get('eval_mean_return', 0.0)):.4f} | "
            f"mean_nmse={float(results.get('eval_mean_nmse', 0.0)):.6f}"
        )
    finally:
        for env in eval_envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


if __name__ == "__main__":
    main()

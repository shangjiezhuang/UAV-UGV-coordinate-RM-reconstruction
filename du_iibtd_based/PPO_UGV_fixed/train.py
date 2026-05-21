"""
Main training script for UAV-UGV Cooperative MAPPO.
"""

import argparse
import __main__
import json
import os
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def _configure_matplotlib_cache_dir() -> None:
    """Avoid matplotlib cache writes to an unwritable HOME during training/export."""
    if os.environ.get("MPLCONFIGDIR"):
        return

    default_cache_dir = os.path.join(os.path.expanduser("~"), ".config", "matplotlib")
    if os.path.isdir(default_cache_dir) and os.access(default_cache_dir, os.W_OK):
        return

    uid = getattr(os, "getuid", lambda: "user")()
    fallback_dir = os.path.join(tempfile.gettempdir(), f"matplotlib-{uid}")
    os.makedirs(fallback_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = fallback_dir


_configure_matplotlib_cache_dir()

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None

from buffer import RolloutBuffer
from config import Config
from environment import (
    SubprocVecUAVUGVEnvironment,
    UAVUGVEnvironment,
    VecUAVUGVEnvironment,
)
from mappo import MAPPO
from networks import MAPPOPolicy
from utils import MetricsLogger, evaluate_policy, json_default, print_config_summary, set_seeds
from du_iibtd_based.scene_suite import (
    apply_scene_cli_overrides,
    build_shared_data_suite,
    configure_scene_suite_config,
    evaluate_scene_suite,
    format_scene_suite,
    scene_config_from_shared_data,
    select_shared_data,
)


EPISODE_SUM_KEYS = [
    "r_nmse",
    "r_unc",
    "r_new_freq",
    "r_new_spatial",
    "r_tx",
    "r_queue",
    "r_progress",
    "r_revisit",
    "r_terminal",
    "r_uav_progress",
    "r_goal_arrival",
    "ensemble_triggered",
    "team_reward",
    "data_produced_bits",
    "data_delivered_bits",
    "novel_data_delivered_bits",
    "uav_move_dist",
    "target_gap_penalty_diag",
    "newly_sampled_freqs",
    "newly_visited_spatial",
]

_EVAL_DATA_SEED_OFFSET = 10_000
_EVAL_RESET_SEED_OFFSET = 100_000


def _parse_bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO UAV-UGV Training")
    defaults = Config()

    parser.add_argument("--num_envs", type=int, default=defaults.mappo.num_envs)
    parser.add_argument("--total_timesteps", type=int, default=defaults.mappo.total_timesteps)
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=None,
        help="Deprecated compatibility option; accepted but ignored. Rollout length follows --episode_max_steps.",
    )
    parser.add_argument("--episode_max_steps", type=int, default=defaults.mappo.episode_max_steps)
    parser.add_argument("--num_minibatches", type=int, default=defaults.mappo.num_minibatches)
    parser.add_argument("--num_epochs", type=int, default=defaults.mappo.num_epochs)
    parser.add_argument("--lr_actor", type=float, default=defaults.mappo.lr_actor)
    parser.add_argument("--lr_critic", type=float, default=defaults.mappo.lr_critic)
    parser.add_argument("--seed", type=int, default=defaults.mappo.seed)
    parser.add_argument("--device", type=str, default=defaults.mappo.device)
    parser.add_argument("--vec_backend", type=str, default=defaults.mappo.vec_backend)
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
        help="Comma-separated DPM100PSD manifest rows used as the shared scene suite.",
    )
    parser.add_argument("--uav_height", type=float, default=defaults.scene.uav_height)
    parser.add_argument("--ugv_height", type=float, default=defaults.scene.ugv_height)
    parser.add_argument("--building_height_m", type=float, default=defaults.scene.building_height_m)

    parser.add_argument("--alpha_nmse", type=float, default=defaults.reward.alpha_nmse)
    parser.add_argument("--nmse_signed_clip", type=float, default=defaults.reward.nmse_signed_clip)
    parser.add_argument("--target_gap_penalty_coef", type=float, default=defaults.reward.target_gap_penalty_coef)
    parser.add_argument("--alpha_unc", type=float, default=defaults.reward.alpha_unc)
    parser.add_argument("--lambda_new_freq", type=float, default=defaults.reward.lambda_new_freq)
    parser.add_argument("--lambda_new_spatial", type=float, default=defaults.reward.lambda_new_spatial)
    parser.add_argument("--beta_tx", type=float, default=defaults.reward.beta_tx)
    parser.add_argument("--gamma_queue", type=float, default=defaults.reward.gamma_queue)
    parser.add_argument("--lambda_uav_progress", type=float, default=defaults.reward.lambda_uav_progress)
    parser.add_argument("--lambda_uav_backtrack", type=float, default=defaults.reward.lambda_uav_backtrack)
    parser.add_argument("--bootstrap_progress_scale", type=float, default=defaults.reward.bootstrap_progress_scale)
    parser.add_argument("--lambda_spatial_revisit", type=float, default=defaults.reward.lambda_spatial_revisit)
    parser.add_argument("--local_goal_arrival_bonus", type=float, default=defaults.reward.local_goal_arrival_bonus)
    parser.add_argument("--terminal_failure_penalty", type=float, default=defaults.reward.terminal_failure_penalty)

    parser.add_argument(
        "--reconstruct_every_n_delivered",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reconstruct_every_n_uav_moves",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ensemble_refresh_interval",
        type=int,
        default=defaults.planner.ensemble_refresh_interval,
        help="Trigger one ensemble refresh/map update after this many newly delivered packets",
    )
    parser.add_argument(
        "--ensemble_full_refresh_interval",
        type=int,
        default=defaults.planner.ensemble_full_refresh_interval,
        help="Run one full ensemble refit every N map updates; use 0 to disable periodic full refresh.",
    )
    parser.add_argument(
        "--nmse_refresh_delta",
        type=float,
        default=defaults.planner.nmse_refresh_delta,
        help="Trigger a full ensemble refit when an incremental update worsens NMSE by at least this amount; use 0 to disable.",
    )
    parser.add_argument(
        "--incremental_outer_iters",
        type=int,
        default=defaults.planner.incremental_outer_iters,
        help="Outer iterations used by fit_incremental between full refreshes.",
    )
    parser.add_argument(
        "--incremental_max_svt_iters",
        type=int,
        default=defaults.planner.incremental_max_svt_iters,
        help="Maximum SVT iterations used by fit_incremental between full refreshes.",
    )
    parser.add_argument(
        "--ensemble_kernel_bandwidth_mode",
        type=str,
        default=defaults.planner.ensemble_kernel_bandwidth_mode,
        choices=["fixed", "same", "base_pm_delta", "pm_delta", "plus_minus"],
        help="Shared-member ensemble diversity mode for II-BTD kernel bandwidth.",
    )
    parser.add_argument(
        "--ensemble_kernel_bandwidth_delta",
        type=float,
        default=defaults.planner.ensemble_kernel_bandwidth_delta,
        help="Absolute bandwidth offset used by base_pm_delta shared-member ensemble mode.",
    )
    parser.add_argument(
        "--ensemble_init_jitter_scale",
        type=float,
        default=defaults.planner.ensemble_init_jitter_scale,
        help="Initial-state jitter scale used to diversify shared-observation ensemble members.",
    )
    parser.add_argument(
        "--ensemble_quality_weighted",
        type=_parse_bool_arg,
        default=defaults.planner.ensemble_quality_weighted,
        help="Whether to weight ensemble members by observed-entry NMSE.",
    )
    parser.add_argument(
        "--planner_target_mode",
        type=str,
        default=defaults.planner.target_mode,
        help="Planner target scope: local, global, or hybrid.",
    )
    parser.add_argument(
        "--initial_observation_mode",
        type=str,
        default=defaults.planner.initial_observation_mode,
        help="Planner warmup mode before active planning: bootstrap or prefill.",
    )
    parser.add_argument(
        "--local_planner_radius",
        type=int,
        default=defaults.planner.local_planner_radius,
        help="Planner only searches goals within this Manhattan radius of the UAV",
    )
    parser.add_argument(
        "--hybrid_stall_update_count",
        type=int,
        default=defaults.planner.hybrid_stall_update_count,
        help="In hybrid mode, switch local->global after this many consecutive weak map updates.",
    )
    parser.add_argument(
        "--hybrid_stall_nmse_threshold",
        type=float,
        default=defaults.planner.hybrid_stall_nmse_threshold,
        help="In hybrid mode, local-mode map updates below this absolute NMSE improvement count as stalled.",
    )
    parser.add_argument(
        "--hybrid_global_step_multiplier",
        type=int,
        default=defaults.planner.hybrid_global_step_multiplier,
        help="In hybrid mode, stay global for this many ensemble intervals before allowing a return to local mode.",
    )
    parser.add_argument(
        "--hybrid_local_min_candidate_count",
        type=int,
        default=defaults.planner.hybrid_local_min_candidate_count,
        help="In hybrid mode, local planning resumes once local viable targets reach max(this value, planner.target_count).",
    )
    parser.add_argument(
        "--prefill_percent",
        type=float,
        default=defaults.planner.prefill_percent,
        help="Prefill observations as a percentage of the sensing-budget basis when initial_observation_mode=prefill.",
    )
    parser.add_argument(
        "--prefill_budget_basis",
        type=int,
        default=defaults.planner.prefill_budget_basis,
        help="Prefill sensing-budget basis. Use <= 0 to fall back to episode_max_steps.",
    )
    parser.add_argument(
        "--init_building_clearance",
        type=int,
        default=defaults.planner.init_building_clearance,
        help="Preferred building-clearance radius for initial UAV/UGV positions.",
    )
    parser.add_argument(
        "--bootstrap_building_clearance",
        type=int,
        default=defaults.planner.bootstrap_building_clearance,
        help="Preferred building-clearance radius for bootstrap targets.",
    )
    parser.add_argument(
        "--flush_reconstruction_on_episode_end",
        type=_parse_bool_arg,
        default=defaults.planner.flush_reconstruction_on_episode_end,
        help="Whether to force one last reconstruction flush when an episode terminates.",
    )
    parser.add_argument(
        "--iibtd_mu",
        type=float,
        default=defaults.planner.iibtd_mu,
        help="II-BTD solver penalty parameter mu.",
    )
    parser.add_argument(
        "--iibtd_nu",
        type=float,
        default=defaults.planner.iibtd_nu,
        help="II-BTD solver penalty parameter nu.",
    )
    parser.add_argument(
        "--iibtd_kernel_bandwidth",
        type=float,
        default=defaults.planner.iibtd_kernel_bandwidth,
        help="Kernel bandwidth used by the II-BTD reconstruction solver.",
    )
    parser.add_argument(
        "--iibtd_backend",
        type=str,
        default=defaults.planner.iibtd_backend,
        help="II-BTD reconstruction backend: auto, cpu, gpu, du_iibtd_res_sr, or du_iibtd_res_sr_learn_nu.",
    )
    parser.add_argument(
        "--iibtd_device",
        type=str,
        default=defaults.planner.iibtd_device,
        help="Device used by the GPU II-BTD backend. Use auto to inherit runtime selection.",
    )
    parser.add_argument(
        "--iibtd_gpu_phi_solver",
        type=str,
        default=defaults.planner.iibtd_gpu_phi_solver,
        help="NNLS backend inside the GPU II-BTD solver: scipy or pgd.",
    )
    parser.add_argument(
        "--du_iibtd_checkpoints",
        nargs="+",
        default=None,
        help=(
            "Optional residual-Sr learn-nu checkpoint list used when "
            "--iibtd_backend is du_iibtd_res_sr or du_iibtd_res_sr_learn_nu."
        ),
    )

    parser.add_argument("--log_dir", type=str, default=defaults.mappo.log_dir)
    parser.add_argument("--model_dir", type=str, default=defaults.mappo.model_dir)
    parser.add_argument("--log_interval", type=int, default=defaults.mappo.log_interval)
    parser.add_argument("--eval_interval", type=int, default=defaults.mappo.eval_interval)
    parser.add_argument("--eval_seed_stride", type=int, default=defaults.mappo.eval_seed_stride)
    parser.add_argument("--save_interval", type=int, default=defaults.mappo.save_interval)

    return parser.parse_args()


def build_config(args) -> Config:
    config = Config()

    config.mappo.num_envs = args.num_envs
    config.mappo.total_timesteps = args.total_timesteps
    config.mappo.episode_max_steps = int(args.episode_max_steps)
    config.mappo.rollout_length = config.mappo.episode_max_steps
    config.mappo.num_minibatches = args.num_minibatches
    config.mappo.num_epochs = args.num_epochs
    config.mappo.lr_actor = args.lr_actor
    config.mappo.lr_critic = args.lr_critic
    config.mappo.seed = args.seed
    config.mappo.device = args.device
    config.mappo.vec_backend = str(args.vec_backend).strip().lower()
    config.mappo.log_dir = args.log_dir
    config.mappo.model_dir = args.model_dir
    config.mappo.log_interval = args.log_interval
    config.mappo.eval_interval = args.eval_interval
    config.mappo.eval_seed_stride = max(0, int(args.eval_seed_stride))
    config.mappo.save_interval = args.save_interval
    apply_scene_cli_overrides(config, args)
    config.scene.uav_height = float(args.uav_height)
    config.scene.ugv_height = float(args.ugv_height)
    config.scene.building_height_m = float(args.building_height_m)

    config.reward.alpha_nmse = args.alpha_nmse
    config.reward.nmse_signed_clip = args.nmse_signed_clip
    config.reward.target_gap_penalty_coef = args.target_gap_penalty_coef
    config.reward.alpha_unc = args.alpha_unc
    config.reward.lambda_new_freq = args.lambda_new_freq
    config.reward.lambda_new_spatial = args.lambda_new_spatial
    config.reward.beta_tx = args.beta_tx
    config.reward.gamma_queue = args.gamma_queue
    config.reward.lambda_uav_progress = args.lambda_uav_progress
    config.reward.lambda_uav_backtrack = args.lambda_uav_backtrack
    config.reward.bootstrap_progress_scale = args.bootstrap_progress_scale
    config.reward.lambda_spatial_revisit = args.lambda_spatial_revisit
    config.reward.local_goal_arrival_bonus = args.local_goal_arrival_bonus
    config.reward.terminal_failure_penalty = args.terminal_failure_penalty

    interval = int(args.ensemble_refresh_interval)
    default_interval = int(config.planner.ensemble_refresh_interval)
    if (
        args.reconstruct_every_n_delivered is not None
        and interval == default_interval
    ):
        interval = int(args.reconstruct_every_n_delivered)
    if args.reconstruct_every_n_uav_moves is not None and interval == default_interval:
        interval = int(args.reconstruct_every_n_uav_moves)
    config.planner.ensemble_refresh_interval = max(1, interval)
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
    config.planner.target_mode = str(args.planner_target_mode).strip().lower() or config.planner.target_mode
    config.planner.initial_observation_mode = (
        str(args.initial_observation_mode).strip().lower() or config.planner.initial_observation_mode
    )
    config.planner.local_planner_radius = max(1, int(args.local_planner_radius))
    config.planner.hybrid_stall_update_count = max(1, int(args.hybrid_stall_update_count))
    config.planner.hybrid_stall_nmse_threshold = max(0.0, float(args.hybrid_stall_nmse_threshold))
    config.planner.hybrid_global_step_multiplier = max(1, int(args.hybrid_global_step_multiplier))
    config.planner.hybrid_local_min_candidate_count = max(1, int(args.hybrid_local_min_candidate_count))
    config.planner.prefill_percent = float(args.prefill_percent)
    config.planner.prefill_budget_basis = max(0, int(args.prefill_budget_basis))
    config.planner.init_building_clearance = max(0, int(args.init_building_clearance))
    config.planner.bootstrap_building_clearance = max(0, int(args.bootstrap_building_clearance))
    config.planner.flush_reconstruction_on_episode_end = bool(
        args.flush_reconstruction_on_episode_end
    )
    config.planner.iibtd_mu = float(args.iibtd_mu)
    config.planner.iibtd_nu = float(args.iibtd_nu)
    config.planner.iibtd_kernel_bandwidth = float(args.iibtd_kernel_bandwidth)
    config.planner.iibtd_backend = str(args.iibtd_backend).strip().lower() or config.planner.iibtd_backend
    config.planner.iibtd_device = str(args.iibtd_device).strip() or config.planner.iibtd_device
    config.planner.iibtd_gpu_phi_solver = (
        str(args.iibtd_gpu_phi_solver).strip().lower() or config.planner.iibtd_gpu_phi_solver
    )
    if args.du_iibtd_checkpoints is not None:
        config.planner.du_iibtd_checkpoints = [
            str(path).strip()
            for path in args.du_iibtd_checkpoints
            if str(path).strip()
        ]

    config.__post_init__()

    return config


def make_env_factory(
    config: Config,
    data_seed: Optional[int] = None,
    shared_data: Optional[Dict] = None,
    minimal_info: bool = False,
):
    from sim_models import GridScene, IIBTD_opt, SimDataGen

    next_auto_idx = {"value": 0}
    if shared_data is None:
        shared_data_seed = int(config.mappo.seed if data_seed is None else data_seed)
        shared_data = build_shared_data_suite(config, SimDataGen, shared_data_seed)

    def factory(idx: Optional[int] = None):
        if idx is None:
            idx = next_auto_idx["value"]
        idx = int(idx)
        next_auto_idx["value"] = max(int(next_auto_idx["value"]), idx + 1)

        env_shared_data = select_shared_data(shared_data, idx)
        env_config = scene_config_from_shared_data(config, env_shared_data)
        sim_data = SimDataGen(
            config=env_config,
            seed=env_config.mappo.seed + idx,
            precomputed_data=env_shared_data,
        )
        td = IIBTD_opt(
            config=env_config,
            grid_coords=sim_data.grid_coords,
            bounds=sim_data.bounds,
            i_mask=sim_data.I_mask,
            n_sources=1,
        )
        return UAVUGVEnvironment(
            config=env_config,
            tensor_decomp=td,
            sim_data=sim_data,
            scene_map=GridScene(
                env_config,
                occupancy_grid=sim_data.get_building_mask(),
                building_heights=sim_data.get_building_heights(),
            ),
            minimal_info=minimal_info,
        )

    return factory


def make_shared_env_data_suite(config: Config, data_seed: Optional[int] = None) -> List[Dict]:
    from sim_models import SimDataGen

    shared_data_seed = int(config.mappo.seed if data_seed is None else data_seed)
    return build_shared_data_suite(
        config,
        SimDataGen,
        shared_data_seed,
    )


def make_episode_trackers(
    num_envs: int,
    keys: List[str],
) -> Tuple[List[Dict[str, float]], List[int]]:
    """
        收集个episode的信息
    """
    accumulators: List[Dict[str, float]] = []
    step_counts: List[int] = []
    for _ in range(num_envs):
        item: Dict[str, float] = {}
        for key in keys:
            item[key] = 0.0
        accumulators.append(item)
        step_counts.append(0)
    return accumulators, step_counts


def reset_episode_tracker_entry(
    accumulators: List[Dict[str, float]],
    step_counts: List[int],
    env_idx: int,
    keys: List[str],
) -> None:
    for key in keys:
        accumulators[env_idx][key] = 0.0
    step_counts[env_idx] = 0


def update_episode_trackers(
    logger: MetricsLogger,
    infos: List[dict],
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    accumulators: List[Dict[str, float]],
    step_counts: List[int],
    keys: List[str],
) -> int:
    episodes_finished = 0
    num_envs = len(infos)
    for env_idx in range(num_envs):
        info = infos[env_idx]
        step_counts[env_idx] = step_counts[env_idx] + 1

        for key in keys:
            accumulators[env_idx][key] = accumulators[env_idx][key] + float(info.get(key, 0.0))

        if bool(terminateds[env_idx]) or bool(truncateds[env_idx]):
            episode_info = dict(info)
            for key in keys:
                episode_info[key] = float(accumulators[env_idx][key])
            episode_info["episode_steps"] = int(step_counts[env_idx])
            logger.log_episode(episode_info)
            reset_episode_tracker_entry(accumulators, step_counts, env_idx, keys)
            episodes_finished = episodes_finished + 1
    return episodes_finished


def make_done_array(terminateds: np.ndarray, truncateds: np.ndarray) -> np.ndarray:
    return np.logical_or(terminateds, truncateds).astype(np.float32)


def compute_timeout_bootstrap_values(
    policy: MAPPOPolicy,
    infos: List[dict],
    terminateds: np.ndarray,
    truncateds: np.ndarray,
) -> np.ndarray:
    timeout_values = np.zeros(len(infos), dtype=np.float32)
    timeout_indices: List[int] = []
    timeout_states: List[np.ndarray] = []

    for env_idx, info in enumerate(infos):
        if not bool(truncateds[env_idx]) or bool(terminateds[env_idx]):
            continue
        terminal_obs = info.get("terminal_obs")
        if not isinstance(terminal_obs, dict) or "critic_state" not in terminal_obs:
            continue
        timeout_indices.append(env_idx)
        timeout_states.append(np.asarray(terminal_obs["critic_state"], dtype=np.float32))

    if timeout_indices:
        timeout_state_batch = np.stack(timeout_states, axis=0)
        timeout_value_batch = np.asarray(policy.get_value(timeout_state_batch), dtype=np.float32)
        timeout_values[np.asarray(timeout_indices, dtype=int)] = timeout_value_batch

    return timeout_values


def print_progress(
    update: int,
    total_updates: int,
    global_step: int,
    total_transitions: int,
    update_metrics: Dict[str, float],
    episodes_finished: int,
) -> None:
    line = (
        f"[Progress] update {update}/{total_updates} | "
        f"steps {global_step}/{total_transitions} | "
        f"uav_pi {update_metrics['uav_policy_loss']:.3f} | "
        f"v {update_metrics['value_loss']:.3f}"
    )
    if episodes_finished > 0:
        line = line + f" | ep {episodes_finished}"
    print(line)


def build_run_metadata(config: Config, cli_args: Optional[argparse.Namespace]) -> Dict[str, object]:
    """Build a JSON-serializable snapshot of the run settings."""
    metadata: Dict[str, object] = {
        "config": asdict(config),
        "run_info": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        },
    }
    if cli_args is not None:
        metadata["cli_args"] = vars(cli_args).copy()
    return metadata


def write_json(path: str, payload: Dict[str, object]) -> str:
    """Write a JSON payload with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    return path


def export_timestamped_figures(
    logger: MetricsLogger,
    source_log_dir: str,
    target_nmse: float,
    run_metadata: Optional[Dict[str, object]] = None,
    render_figures: bool = True,
) -> Tuple[str, str]:
    """
    Save metrics.json and run settings to <source_log_dir>/<timestamp>/, then auto-generate figures there.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = os.path.join(source_log_dir, timestamp)
    os.makedirs(target_dir, exist_ok=True)

    dst_metrics = logger.save(os.path.join(target_dir, "metrics.json"))
    if run_metadata:
        write_json(os.path.join(target_dir, "config.json"), run_metadata["config"])
        if "cli_args" in run_metadata:
            write_json(os.path.join(target_dir, "cli_args.json"), run_metadata["cli_args"])
        if "run_info" in run_metadata:
            write_json(os.path.join(target_dir, "run_info.json"), run_metadata["run_info"])

    if render_figures:
        from visualize import plot_all
        plot_all(log_dir=target_dir, target_nmse=target_nmse)
    return target_dir, dst_metrics


def save_best_eval_checkpoint(
    policy: MAPPOPolicy,
    model_dir: str,
    checkpoint_name: str,
    metric_name: str,
    metric_value: float,
    best_value: float,
    update: int,
    higher_is_better: bool,
) -> Tuple[float, Optional[str]]:
    """Persist a best-so-far checkpoint when an eval metric improves."""
    if not np.isfinite(metric_value):
        return best_value, None

    improved = metric_value > best_value if higher_is_better else metric_value < best_value
    if not improved:
        return best_value, None

    ckpt_path = os.path.join(model_dir, checkpoint_name)
    policy.save(ckpt_path)
    print(
        f"Saved {checkpoint_name}: {ckpt_path} "
        f"(update {update}, {metric_name}={metric_value:.6f})"
    )
    return metric_value, ckpt_path


def _cuda_device_count() -> int:
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _normalize_device_string(device: str, default_cuda_index: int = 0) -> str:
    device_str = str(device or "").strip().lower()
    if not device_str:
        return "cpu"
    if device_str == "cuda":
        return f"cuda:{int(default_cuda_index)}"
    return device_str


def _extract_cuda_index(device: str) -> Optional[int]:
    device_str = str(device or "").strip().lower()
    if not device_str.startswith("cuda"):
        return None
    if device_str == "cuda":
        return 0
    if ":" not in device_str:
        return 0
    try:
        return int(device_str.split(":", 1)[1])
    except ValueError:
        return 0


def _pick_visible_cuda_index(
    cuda_count: int,
    preferred_indices: List[int],
    blocked_indices: Optional[List[int]] = None,
) -> int:
    blocked = set()
    for idx in blocked_indices or []:
        blocked.add(int(idx))

    for idx in preferred_indices:
        idx = int(idx)
        if 0 <= idx < cuda_count and idx not in blocked:
            return idx

    for idx in range(cuda_count):
        if idx not in blocked:
            return idx
    return 0


def _preferred_train_cuda_index(cuda_count: int) -> int:
    return _pick_visible_cuda_index(cuda_count, preferred_indices=[0])


def _preferred_planner_cuda_index(
    cuda_count: int,
    train_idx: Optional[int],
) -> int:
    blocked: List[int] = []
    if train_idx is not None and cuda_count > 1:
        blocked.append(int(train_idx))
    return _pick_visible_cuda_index(
        cuda_count,
        preferred_indices=[0],
        blocked_indices=blocked,
    )


def _resolve_iibtd_runtime_backend(config: Config) -> str:
    backend = str(config.planner.iibtd_backend).strip().lower()
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        return "gpu" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if backend in {"du_iibtd_res_sr", "du_iibtd_res_sr_learn_nu"}:
        device = str(config.planner.iibtd_device).strip().lower()
        if device == "cpu":
            return "cpu"
        return "gpu" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if backend == "auto":
        if torch is not None and torch.cuda.is_available():
            return "gpu"
        return "cpu"
    return "cpu"


def _resolve_iibtd_runtime_device(config: Config) -> str:
    device = str(config.planner.iibtd_device).strip()
    if not device or device.lower() == "auto":
        return _normalize_device_string(config.mappo.device)
    return _normalize_device_string(device)


def _running_from_interactive_main() -> bool:
    main_file = getattr(__main__, "__file__", None)
    if main_file is None:
        return True
    main_file = str(main_file).strip()
    return (not main_file) or main_file.startswith("<")


def _apply_runtime_backend_policy(config: Config) -> Optional[str]:
    """
    Auto-assign runtime devices:
    - Prefer MAPPO on cuda:1 and II-BTD on cuda:2 when those devices exist.
    - If fewer GPUs are available, fall back to visible devices while still trying
      to keep MAPPO and II-BTD on different GPUs.
    - Spawn-based subproc rollout can safely initialize CUDA inside workers.
    """
    iibtd_backend = _resolve_iibtd_runtime_backend(config)
    cuda_count = _cuda_device_count()
    if iibtd_backend != "gpu" or cuda_count <= 0:
        return None

    messages: List[str] = []

    train_device_arg = str(config.mappo.device).strip().lower()
    planner_device_arg = str(config.planner.iibtd_device).strip().lower()
    preferred_train_idx = _preferred_train_cuda_index(cuda_count)

    if not train_device_arg or train_device_arg == "cuda":
        config.mappo.device = _normalize_device_string(
            config.mappo.device,
            default_cuda_index=preferred_train_idx,
        )
    train_device = _normalize_device_string(
        config.mappo.device,
        default_cuda_index=preferred_train_idx,
    )

    if not planner_device_arg or planner_device_arg in {"auto", "cuda"}:
        train_idx = _extract_cuda_index(train_device)
        planner_default_idx = _preferred_planner_cuda_index(cuda_count, train_idx)
        config.planner.iibtd_device = _normalize_device_string(
            "cuda",
            default_cuda_index=planner_default_idx,
        )
        messages.append(
            f"defaulted II-BTD device {config.planner.iibtd_device} "
            f"(visible CUDA devices: {cuda_count})"
        )

    iibtd_device = _resolve_iibtd_runtime_device(config)
    if not train_device.startswith("cuda") or not iibtd_device.startswith("cuda"):
        return "; ".join(messages) if messages else None

    if train_device != iibtd_device:
        messages.insert(
            0,
            f"MAPPO uses {train_device} and II-BTD uses {iibtd_device}"
        )
        return "; ".join(messages)

    messages.insert(
        0,
        f"MAPPO training and II-BTD both use {train_device}"
    )
    return "; ".join(messages)


def train(config: Config, cli_args: Optional[argparse.Namespace] = None):
    mc = config.mappo
    valid_vec_backends = {"sync", "subproc"}
    if mc.vec_backend not in valid_vec_backends:
        raise ValueError(
            f"Unsupported vec_backend={mc.vec_backend!r}; "
            f"expected one of {sorted(valid_vec_backends)}"
        )
    if mc.vec_backend == "subproc" and _running_from_interactive_main():
        print(
            "[Train] vec_backend='subproc' is not compatible with the current "
            "interactive entrypoint; falling back to 'sync'."
        )
        mc.vec_backend = "sync"
    runtime_policy_message = _apply_runtime_backend_policy(config)
    if runtime_policy_message is not None:
        print(f"[Runtime Policy] {runtime_policy_message}")

    set_seeds(mc.seed)
    os.makedirs(mc.model_dir, exist_ok=True)
    scene_indices = configure_scene_suite_config(config)
    if int(mc.num_envs) != len(scene_indices):
        raise ValueError(
            f"num_envs must equal the scene suite length so one worker maps to one scene; "
            f"got num_envs={int(mc.num_envs)}, scene_count={len(scene_indices)}."
        )
    shared_env_data = make_shared_env_data_suite(config)
    eval_config = deepcopy(config)
    eval_config.mappo.seed = int(config.mappo.seed + _EVAL_RESET_SEED_OFFSET)
    # Train and eval share the same DPM100PSD scene suite; only reset seeds differ.
    eval_shared_env_data = shared_env_data
    run_metadata = build_run_metadata(config, cli_args)
    run_metadata.setdefault("config", {}).setdefault("scene", {})[
        "radioseer_scene_indices"
    ] = list(scene_indices)
    logger = MetricsLogger(mc.log_dir, metadata=run_metadata)
    print_config_summary(config)
    print(
        f"[Scene Suite] DPM100PSD manifest rows: {format_scene_suite(scene_indices)}; "
        f"workers use env_idx % {len(scene_indices)}."
    )

    print("\n*** Using direct simulation data pipeline ***\n")
    # 创建虚拟环境
    eval_env_factory = make_env_factory(
        eval_config,
        shared_data=eval_shared_env_data,
    )
    env_factory = make_env_factory(
        config,
        shared_data=shared_env_data,
        minimal_info=True,
    )
    if mc.vec_backend == "subproc":
        vec_env = SubprocVecUAVUGVEnvironment(
            num_envs=mc.num_envs,
            config=config,
            shared_data=shared_env_data,
            minimal_info=True,
        )
    else:
        vec_env = VecUAVUGVEnvironment(
            num_envs=mc.num_envs,
            config=config,
            env_factory=env_factory,
            minimal_info=True,
        )
    eval_envs = [
        eval_env_factory(scene_pos)
        for scene_pos in range(len(shared_env_data))
    ] if mc.eval_interval > 0 else []
    if mc.eval_interval > 0:
        if int(mc.eval_seed_stride) == 0:
            print(
                f"[Eval] Reusing training DPM100PSD scene suite "
                f"{format_scene_suite(scene_indices)}; reset seeds start at "
                f"{eval_config.mappo.seed}."
            )
        else:
            print(
                f"[Eval] Reusing training DPM100PSD scene suite "
                f"{format_scene_suite(scene_indices)}; reset seed base {eval_config.mappo.seed} "
                f"with stride {int(mc.eval_seed_stride)}"
            )

    try:
        # 获取观测空间和动作空间维度
        obs_dims = vec_env.obs_dims
        action_dims = vec_env.action_dims
        print(f"Observation dims: {obs_dims}")
        print(f"Action dims: {action_dims}")

        # 初始化MAPPO 和 policy
        policy = MAPPOPolicy(obs_dims, action_dims, mc)
        mappo = MAPPO(policy, mc)
        print(
            "[Runtime Devices] "
            f"MAPPO train/update={policy.device} | "
            f"policy action inference={policy.rollout_device} | "
            f"II-BTD reconstruct during rollout={_resolve_iibtd_runtime_device(config)}"
        )

        # 初始化buffer
        buffer = RolloutBuffer(
            rollout_length=mc.rollout_length,
            num_envs=mc.num_envs,
            obs_dims=obs_dims,
            action_dims=action_dims,
            gamma=mc.gamma,
            gae_lambda=mc.gae_lambda,
        )

        transitions_per_update = mc.rollout_length * mc.num_envs
        if mc.num_minibatches <= 0:
            raise ValueError(f"num_minibatches must be positive, got {mc.num_minibatches}")
        if mc.num_minibatches > transitions_per_update:
            print(
                f"[Train] num_minibatches={mc.num_minibatches} is larger than "
                f"rollout batch size {transitions_per_update}; clamping."
            )
            mc.num_minibatches = transitions_per_update

        if mc.total_timesteps < transitions_per_update:
            raise ValueError(
                f"total_timesteps={mc.total_timesteps} is smaller than one full update "
                f"({transitions_per_update} = rollout_length*num_envs). "
                "Increase total_timesteps or reduce rollout_length/num_envs."
            )

        total_updates, remainder = divmod(mc.total_timesteps, transitions_per_update)
        if remainder != 0:
            print(
                f"[Train] Warning: total_timesteps={mc.total_timesteps} is not divisible by "
                f"rollout batch size {transitions_per_update}. "
                f"Ignoring last {remainder} transitions."
            )
        total_transitions = total_updates * transitions_per_update
        global_step = 0
        best_eval_nmse = np.inf
        best_eval_nmse_update: Optional[int] = None
        best_eval_nmse_path: Optional[str] = None
        best_eval_return = -np.inf
        best_eval_return_update: Optional[int] = None
        best_eval_return_path: Optional[str] = None
        last_eval_results: Optional[Dict[str, float]] = None
        last_eval_update: Optional[int] = None
        print(f"\nStarting training for {total_updates} updates ({mc.total_timesteps:,} total timesteps)")
        print(f"  Rollout: {mc.rollout_length} steps × {mc.num_envs} envs = {transitions_per_update} transitions/update\n")

        # 训练前 reset环境
        obs = vec_env.reset()
        # 初始化episode信息收集器
        accumulators, step_counts = make_episode_trackers(mc.num_envs, EPISODE_SUM_KEYS)

        for update in range(1, total_updates + 1):
            buffer.reset()
            episodes_finished = 0

            # 收集 rollout数据
            for _ in range(mc.rollout_length):
                # 获取当前policy下的动作 以及 critic 的value
                action_data = policy.get_actions(
                    uav_obs=obs["uav_obs"],
                    critic_state=obs["critic_state"],
                    uav_action_mask=obs["uav_action_mask"],
                )

                # 运行环境交互，获取 观测 奖励 等信息
                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(
                    uav_actions=action_data["uav_action"],
                )
                dones = make_done_array(terminateds, truncateds)
                timeout_values = compute_timeout_bootstrap_values(
                    policy=policy,
                    infos=infos,
                    terminateds=terminateds,
                    truncateds=truncateds,
                )

                # 将交互信息存储到buffer中
                buffer.add(
                    uav_obs=obs["uav_obs"],
                    critic_state=obs["critic_state"],
                    uav_action=action_data["uav_action"],
                    uav_log_prob=action_data["uav_log_prob"],
                    uav_action_mask=obs["uav_action_mask"],
                    reward=rewards["team_reward"],
                    value=action_data["value"],
                    done=dones,
                    terminated=terminateds.astype(np.float32),
                    truncated=truncateds.astype(np.float32),
                    timeout_value=timeout_values,
                )

                obs = next_obs
                global_step = global_step + mc.num_envs
                episodes_finished = episodes_finished + update_episode_trackers(
                    logger=logger,
                    infos=infos,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    accumulators=accumulators,
                    step_counts=step_counts,
                    keys=EPISODE_SUM_KEYS,
                )

            # rollout结束后 计算critic value
            last_value = policy.get_value(obs["critic_state"])
            # 计算advantage 和 return
            buffer.compute_returns_and_advantages(last_value)
            # 利用前述数据训练mappo 更新策略
            update_metrics = mappo.update(buffer)
            buffer.release_cached_tensors()

            if mc.log_interval > 0 and update % mc.log_interval == 0:
                update_metrics["global_step"] = global_step
                logger.log_update(update, update_metrics)

            if mc.eval_interval > 0 and update % mc.eval_interval == 0:
                eval_seed_base = int(
                    eval_config.mappo.seed + update * max(int(mc.eval_seed_stride), 0)
                )
                eval_results = evaluate_scene_suite(
                    envs=eval_envs,
                    evaluate_fn=evaluate_policy,
                    policy=policy,
                    num_episodes=mc.eval_episodes,
                    max_steps=mc.episode_max_steps,
                    seed_base=eval_seed_base,
                )
                logger.log_eval(update, eval_results)
                last_eval_results = eval_results
                last_eval_update = int(update)

                eval_mean_nmse = float(eval_results.get("eval_mean_nmse", np.nan))
                best_eval_nmse, saved_path = save_best_eval_checkpoint(
                    policy=policy,
                    model_dir=mc.model_dir,
                    checkpoint_name="best_nmse.pt",
                    metric_name="eval_mean_nmse",
                    metric_value=eval_mean_nmse,
                    best_value=best_eval_nmse,
                    update=update,
                    higher_is_better=False,
                )
                if saved_path is not None:
                    best_eval_nmse_update = update
                    best_eval_nmse_path = saved_path

                eval_mean_return = float(eval_results.get("eval_mean_return", np.nan))
                best_eval_return, saved_path = save_best_eval_checkpoint(
                    policy=policy,
                    model_dir=mc.model_dir,
                    checkpoint_name="best_return.pt",
                    metric_name="eval_mean_return",
                    metric_value=eval_mean_return,
                    best_value=best_eval_return,
                    update=update,
                    higher_is_better=True,
                )
                if saved_path is not None:
                    best_eval_return_update = update
                    best_eval_return_path = saved_path

            if mc.save_interval > 0 and update % mc.save_interval == 0:
                ckpt_path = os.path.join(mc.model_dir, f"checkpoint_{update}.pt")
                policy.save(ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            print_progress(
                update=update,
                total_updates=total_updates,
                global_step=global_step,
                total_transitions=total_transitions,
                update_metrics=update_metrics,
                episodes_finished=episodes_finished,
            )

        final_path = os.path.join(mc.model_dir, "final_model.pt")
        policy.save(final_path)

        should_run_final_eval = bool(mc.eval_interval > 0)
        if should_run_final_eval:
            final_eval_seed_base = int(
                eval_config.mappo.seed + total_updates * max(int(mc.eval_seed_stride), 0)
            )
            if last_eval_results is not None and last_eval_update == int(total_updates):
                final_eval_results = last_eval_results
                print(
                    f"\n[Final Eval] Reusing evaluation already run at update {total_updates} "
                    f"on training DPM100PSD scene suite {format_scene_suite(scene_indices)} "
                    f"with reset seed base {final_eval_seed_base}"
                )
            else:
                print(
                    f"\n[Final Eval] Running final evaluation at update {total_updates} "
                    f"on training DPM100PSD scene suite {format_scene_suite(scene_indices)} "
                    f"with reset seed base {final_eval_seed_base}"
                )
                final_eval_results = evaluate_scene_suite(
                    envs=eval_envs,
                    evaluate_fn=evaluate_policy,
                    policy=policy,
                    num_episodes=mc.eval_episodes,
                    max_steps=mc.episode_max_steps,
                    seed_base=final_eval_seed_base,
                )
            logger.log_final_eval(total_updates, final_eval_results)

            eval_mean_nmse = float(final_eval_results.get("eval_mean_nmse", np.nan))
            best_eval_nmse, saved_path = save_best_eval_checkpoint(
                policy=policy,
                model_dir=mc.model_dir,
                checkpoint_name="best_nmse.pt",
                metric_name="eval_mean_nmse",
                metric_value=eval_mean_nmse,
                best_value=best_eval_nmse,
                update=total_updates,
                higher_is_better=False,
            )
            if saved_path is not None:
                best_eval_nmse_update = total_updates
                best_eval_nmse_path = saved_path

            eval_mean_return = float(final_eval_results.get("eval_mean_return", np.nan))
            best_eval_return, saved_path = save_best_eval_checkpoint(
                policy=policy,
                model_dir=mc.model_dir,
                checkpoint_name="best_return.pt",
                metric_name="eval_mean_return",
                metric_value=eval_mean_return,
                best_value=best_eval_return,
                update=total_updates,
                higher_is_better=True,
            )
            if saved_path is not None:
                best_eval_return_update = total_updates
                best_eval_return_path = saved_path
        else:
            print("\n[Final Eval] Skipping final evaluation because eval_interval <= 0.")

        should_render_figures = bool(mc.log_interval > 0 or mc.eval_interval > 0)
        export_dir, metrics_path = export_timestamped_figures(
            logger=logger,
            source_log_dir=mc.log_dir,
            target_nmse=float(config.reward.accuracy_target_nmse),
            run_metadata=run_metadata,
            render_figures=should_render_figures,
        )
        print(f"\nTraining complete. Final model saved to {final_path}")
        if best_eval_nmse_path is not None and best_eval_nmse_update is not None:
            print(
                f"Best NMSE checkpoint saved to {best_eval_nmse_path} "
                f"(update {best_eval_nmse_update}, eval_mean_nmse={best_eval_nmse:.6f})"
            )
        if best_eval_return_path is not None and best_eval_return_update is not None:
            print(
                f"Best return checkpoint saved to {best_eval_return_path} "
                f"(update {best_eval_return_update}, eval_mean_return={best_eval_return:.6f})"
            )
        print(f"Metrics saved to {metrics_path}")
        if run_metadata:
            print(f"Config saved to {os.path.join(export_dir, 'config.json')}")
            if "cli_args" in run_metadata:
                print(f"CLI args saved to {os.path.join(export_dir, 'cli_args.json')}")
        if should_render_figures:
            print(f"Auto figures saved to {os.path.join(export_dir, 'figures')}")
        else:
            print("Auto figure export skipped because log_interval and eval_interval are both <= 0.")
    finally:
        for eval_env in eval_envs:
            if hasattr(eval_env, "close"):
                eval_env.close()
        if hasattr(vec_env, "close"):
            vec_env.close()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    train(config, cli_args=args)

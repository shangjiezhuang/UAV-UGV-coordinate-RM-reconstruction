"""Algorithm-neutral quant learned-policy training runner."""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from du_iibtd_based_fading_delta.runtime_paths import configure_matplotlib_cache_dir
from du_iibtd_based_fading_delta.channel_loss import parse_length_loss_cap

configure_matplotlib_cache_dir()

import numpy as np

from du_iibtd_based_fading_delta.shared.joint_buffer import RolloutBuffer
from du_iibtd_based_fading_delta.shared.quant.config import Config, DU_IIBTD_BACKENDS
from du_iibtd_based_fading_delta.shared.quant.environment import (
    SubprocVecUAVUGVEnvironment,
    UAVUGVEnvironment,
    VecUAVUGVEnvironment,
)
from du_iibtd_based_fading_delta.MAPPO_quant.mappo import MAPPO
from du_iibtd_based_fading_delta.MAPPO_quant.networks import MAPPOPolicy
from du_iibtd_based_fading_delta.IPPO_quant.buffer import RolloutBuffer as IPPORolloutBuffer
from du_iibtd_based_fading_delta.IPPO_quant.ippo import IPPO
from du_iibtd_based_fading_delta.IPPO_quant.networks import IPPOPolicy
from du_iibtd_based_fading_delta.HAPPO.buffer import RolloutBuffer as HAPPORolloutBuffer
from du_iibtd_based_fading_delta.HAPPO.happo import HAPPO
from du_iibtd_based_fading_delta.HAPPO.networks import HAPPOPolicy
from du_iibtd_based_fading_delta.MAPPO_CF.buffer import RolloutBuffer as MAPPOCFRolloutBuffer
from du_iibtd_based_fading_delta.MAPPO_CF.mappo_cf import MAPPOCF
from du_iibtd_based_fading_delta.MAPPO_CF.networks import MAPPOCFPolicy
from du_iibtd_based_fading_delta.fair_training import (
    LEARNED_TRAINERS,
    is_independent_ppo,
    is_uav_only_ppo,
    normalize_trainer,
    training_reward_for_trainer,
    ugv_control_mode_for_trainer,
)
from du_iibtd_based_fading_delta.uav_ppo import UAVPPO, UAVPPOPolicy
from du_iibtd_based_fading_delta.checkpoint_selection import (
    save_best_eval_checkpoint,
    update_pareto_eval_checkpoints,
)
from du_iibtd_based_fading_delta.training_common import (
    _apply_runtime_backend_policy,
    _resolve_iibtd_runtime_backend,
    _resolve_iibtd_runtime_device,
    _running_from_interactive_main,
    compute_ippo_timeout_bootstrap_values,
    compute_timeout_bootstrap_values,
    make_done_array,
    make_episode_trackers,
    parse_bool_arg as _parse_bool_arg,
    print_progress,
    reset_episode_tracker_entry,
    update_episode_trackers,
)
from du_iibtd_based_fading_delta.shared.quant.utils import (
    MetricsLogger,
    evaluate_policy,
    json_default,
    print_config_summary,
    sanitize_json_payload,
    set_seeds,
)
from du_iibtd_based_fading_delta.scene_suite import (
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
    "r_queue",
    "r_progress",
    "r_uav_progress",
    "r_ugv_progress",
    "r_novel_info",
    "r_full_repeat",
    "full_repeat_sample",
    "ensemble_triggered",
    "team_reward",
    "uav_reward",
    "ugv_reward",
    "data_produced_bits",
    "data_delivered_bits",
    "novel_data_delivered_bits",
    "completed_packet_bits",
    "novel_completed_packet_bits",
    "link_transmitted_bits",
    "novel_link_transmitted_bits",
    "dropped_bits",
    "channel_outage",
    "ugv_comm_target_active",
    "uav_move_dist",
    "ugv_move_dist",
    "uav_move_steps",
    "ugv_move_steps",
    "newly_sampled_freqs",
    "newly_visited_spatial",
]

_EVAL_RESET_SEED_OFFSET = 100_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UAV-UGV cooperative PPO training")
    defaults = Config()

    parser.add_argument(
        "--trainer",
        choices=LEARNED_TRAINERS,
        default="mappo",
        help=(
            "MAPPO/HAPPO/MAPPO-CF/IPPO or single-UAV PPO with a "
            "fixed/legacy-heuristic/A* UGV; "
            "all trainers use the same physical environment."
        ),
    )

    parser.add_argument("--num_envs", type=int, default=defaults.training.num_envs)
    parser.add_argument("--total_timesteps", type=int, default=defaults.training.total_timesteps)
    parser.add_argument("--episode_max_steps", type=int, default=defaults.training.episode_max_steps)
    parser.add_argument("--num_minibatches", type=int, default=defaults.training.num_minibatches)
    parser.add_argument("--num_epochs", type=int, default=defaults.training.num_epochs)
    parser.add_argument("--lr_actor", type=float, default=defaults.training.lr_actor)
    parser.add_argument("--lr_critic", type=float, default=defaults.training.lr_critic)
    parser.add_argument("--seed", type=int, default=defaults.training.seed)
    parser.add_argument("--device", type=str, default=defaults.training.device)
    parser.add_argument("--vec_backend", type=str, default=defaults.training.vec_backend)
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
        help="Comma-separated RadioSeer scene specs used as the shared scene suite.",
    )
    parser.add_argument("--grid_spacing", "--grid-spacing", type=float, default=defaults.scene.grid_spacing,
                        help="Physical side length of one grid cell in meters.")
    parser.add_argument("--uav_height", type=float, default=defaults.scene.uav_height)
    parser.add_argument("--ugv_height", type=float, default=defaults.scene.ugv_height)
    parser.add_argument("--building_height_m", type=float, default=defaults.scene.building_height_m, help="Building height in fixed mode only.")
    parser.add_argument("--building_height_mode", choices=("fixed", "uniform_per_building", "uniform_integer_per_building"), default=defaults.scene.building_height_mode)
    parser.add_argument("--building_height_min_m", type=float, default=defaults.scene.building_height_min_m)
    parser.add_argument("--building_height_max_m", type=float, default=defaults.scene.building_height_max_m)
    parser.add_argument("--building_height_seed", type=int, default=defaults.scene.building_height_seed)
    parser.add_argument("--building_split_min_area_m2", type=float, default=defaults.scene.building_split_min_area_m2, help="Split each connected footprint at or above this area into two connected blocks; 0 disables splitting.")
    parser.add_argument(
        "--uav_max_energy",
        type=float,
        default=defaults.uav.max_energy,
        help="UAV episode energy budget in joules; reserve minimum hover+sensing energy to the horizon.",
    )
    parser.add_argument(
        "--outage_snr_mode", choices=("received", "nominal"),
        default=defaults.comm.outage_snr_mode,
        help="Transmission outage gate; nominal is for historical reproduction. Planning always uses nominal SNR.",
    )
    parser.add_argument(
        "--snr_outage_threshold_db",
        type=float,
        default=defaults.comm.snr_outage_threshold_db,
        help="Set link capacity to zero below this SNR threshold (dB); default gate uses received SNR including shadowing.",
    )
    parser.add_argument("--los_excess_db", type=float, default=defaults.comm.los_excess_db,
                        help="Mean LoS path loss above FSPL (dB); urban reference: 1.")
    parser.add_argument("--nlos_length_loss_db_per_m", type=float, default=defaults.comm.nlos_length_loss_db_per_m,
                        help="NLoS loss added per horizontal blocked metre; 0 disables the correction.")
    parser.add_argument("--nlos_length_loss_cap_db", type=parse_length_loss_cap, default=defaults.comm.nlos_length_loss_cap_db,
                        help="Maximum added obstruction-length loss in dB.")
    parser.add_argument("--nlos_excess_db", type=float, default=defaults.comm.nlos_excess_db,
                        help="Mean NLoS path loss above FSPL (dB); urban reference: 20.")
    parser.add_argument("--total_bandwidth", type=float, default=defaults.uav.total_bandwidth, help="Shared RF bandwidth in Hz.")
    parser.add_argument("--source_measurement_bits", type=int, default=defaults.comm.source_measurement_bits, help="Reference bits per uncompressed PSD value.")
    parser.add_argument(
        "--data_per_sample",
        type=float,
        default=defaults.comm.data_per_sample,
        help=(
            "Uncompressed payload per observed frequency band/window (bits), at "
            "source_measurement_bits. Quantized payload scales by quant_bits/source_measurement_bits."
        ),
    )
    parser.add_argument(
        "--quantization_scheme",
        choices=("log_first", "raw"),
        default=defaults.uav.quantization_scheme,
        help=(
            "Numerical measurement quantization domain: log_first quantizes "
            "log-transformed power; raw quantizes linear-domain power directly."
        ),
    )

    parser.add_argument("--alpha_nmse", type=float, default=defaults.reward.alpha_nmse)
    parser.add_argument("--nmse_signed_clip", type=float, default=defaults.reward.nmse_signed_clip)
    parser.add_argument("--gamma_queue", type=float, default=defaults.reward.gamma_queue)
    parser.add_argument("--lambda_uav_progress", type=float, default=defaults.reward.lambda_uav_progress)
    parser.add_argument("--lambda_uav_backtrack", type=float, default=defaults.reward.lambda_uav_backtrack)
    parser.add_argument("--lambda_ugv_progress", type=float, default=defaults.reward.lambda_ugv_progress)
    parser.add_argument("--lambda_ugv_backtrack", type=float, default=defaults.reward.lambda_ugv_backtrack)
    parser.add_argument("--lambda_novel_info", type=float, default=defaults.reward.lambda_novel_info)
    parser.add_argument("--lambda_full_repeat", type=float, default=defaults.reward.lambda_full_repeat)
    parser.add_argument("--bootstrap_progress_scale", type=float, default=defaults.reward.bootstrap_progress_scale)

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
        "--ensemble_quality_weighted",
        type=_parse_bool_arg,
        default=defaults.planner.ensemble_quality_weighted,
        help="Whether to weight ensemble members by observed-entry NMSE.",
    )
    parser.add_argument(
        "--target_arrival_radius_steps",
        type=float,
        default=defaults.planner.target_arrival_radius_steps,
        help="Arrival radius as a multiple of the UAV step size; sampled non-building positions inside this radius complete the current target.",
    )
    parser.add_argument(
        "--target_suppression_radius_steps",
        type=float,
        default=defaults.planner.target_suppression_radius_steps,
        help="Temporarily suppress completed/stuck target neighborhoods by this many UAV step sizes until the next ensemble refresh.",
    )
    parser.add_argument(
        "--target_stuck_no_sample_steps",
        type=int,
        default=defaults.planner.target_stuck_no_sample_steps,
        help="Retarget after this many consecutive active-plan steps without a valid UAV sample.",
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
        "--hybrid_uncertainty_window_updates",
        type=int,
        default=defaults.planner.hybrid_uncertainty_window_updates,
        help="Hybrid mode: number of valid updates in the signed-improvement rolling sum (2).",
    )
    parser.add_argument(
        "--hybrid_uncertainty_improvement_threshold",
        type=float,
        default=defaults.planner.hybrid_uncertainty_improvement_threshold,
        help="Hybrid mode: switch and refit when the rolling sum of signed relative improvements is below this threshold.",
    )
    parser.add_argument(
        "--hybrid_global_hold_intervals",
        type=int,
        default=defaults.planner.hybrid_global_hold_intervals,
        help="Hybrid mode: hold global submode for this many ensemble intervals.",
    )
    parser.add_argument(
        "--hybrid_local_reentry_min_targets",
        type=int,
        default=defaults.planner.hybrid_local_reentry_min_targets,
        help="Hybrid mode: local candidate threshold for returning from global; <=0 uses auto.",
    )
    parser.add_argument(
        "--ugv_comm_target_mode",
        choices=("path_corridor",),
        default=defaults.planner.ugv_comm_target_mode,
        help="Frozen continuous UGV communication-target rule.",
    )
    parser.add_argument(
        "--ugv_comm_backlog_threshold",
        type=float,
        default=defaults.planner.ugv_comm_backlog_threshold,
        help="Switch predictive support to urgent recovery at this normalized bit backlog.",
    )
    parser.add_argument(
        "--ugv_recovery_exit_backlog_threshold",
        type=float,
        default=defaults.planner.ugv_recovery_exit_backlog_threshold,
        help="Return from Support recovery below this normalized bit backlog.",
    )
    parser.add_argument(
        "--ugv_recovery_poor_service_steps",
        type=int,
        default=defaults.planner.ugv_recovery_poor_service_steps,
        help="Consecutive service deficits required to trigger high-backlog recovery.",
    )
    parser.add_argument(
        "--ugv_recovery_min_hold_steps",
        type=int,
        default=defaults.planner.ugv_recovery_min_hold_steps,
        help="Minimum number of environment steps to hold Support recovery.",
    )
    parser.add_argument(
        "--ugv_recovery_good_link_steps",
        type=int,
        default=defaults.planner.ugv_recovery_good_link_steps,
        help="Consecutive healthy-link steps required to return to two-path support.",
    )
    parser.add_argument(
        "--ugv_recovery_service_margin",
        type=float,
        default=defaults.planner.ugv_recovery_service_margin,
        help="Required service-to-arrival bit ratio before leaving recovery.",
    )
    parser.add_argument(
        "--ugv_comm_local_path_horizon",
        type=int,
        default=defaults.planner.ugv_comm_local_path_horizon,
        help="Normal A*-guided communication path horizon in UGV road steps.",
    )
    parser.add_argument(
        "--ugv_comm_expanded_path_horizon",
        type=int,
        default=defaults.planner.ugv_comm_expanded_path_horizon,
        help="Expanded path horizon used when the local corridor has no service point.",
    )
    parser.add_argument(
        "--ugv_comm_corridor_width",
        type=int,
        default=defaults.planner.ugv_comm_corridor_width,
        help="Side-road width around the A*-guided communication path.",
    )
    parser.add_argument(
        "--ugv_service_action_mask",
        type=_parse_bool_arg,
        default=defaults.planner.ugv_service_action_mask,
        help="Restrict UGV actions to progress toward the frozen communication target.",
    )
    parser.add_argument(
        "--obs_remaining_time",
        type=_parse_bool_arg,
        default=defaults.obs.include_remaining_time,
        help="Ablation: append normalized remaining time to both actors and critic.",
    )
    parser.add_argument(
        "--obs_quant_context",
        type=_parse_bool_arg,
        default=defaults.obs.include_quant_context,
        help="Ablation: append current quantization and packet-size context.",
    )
    parser.add_argument(
        "--prefill_percent",
        type=float,
        default=defaults.planner.prefill_percent,
        help="Prefill observation windows as a percentage of the observation-action budget.",
    )
    parser.add_argument(
        "--prefill_budget_basis",
        type=int,
        default=defaults.planner.prefill_budget_basis,
        help="Prefill observation-window budget. Use <= 0 to fall back to episode_max_steps.",
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
        help="DU-IIBTD runtime penalty parameter mu.",
    )
    parser.add_argument(
        "--iibtd_backend",
        type=str,
        default=defaults.planner.iibtd_backend,
        choices=tuple(sorted(DU_IIBTD_BACKENDS)),
        help="DU-IIBTD reconstruction backend: du_iibtd or du_iibtd_res_sr.",
    )
    parser.add_argument(
        "--iibtd_device",
        type=str,
        default=defaults.planner.iibtd_device,
        help="Device used by the DU-IIBTD adapter. Use auto to inherit runtime selection.",
    )
    parser.add_argument(
        "--du_iibtd_checkpoints",
        nargs="+",
        default=None,
        help=(
            "Optional DU-IIBTD ensemble checkpoints. If omitted, defaults follow "
            "--iibtd_backend."
        ),
    )

    parser.add_argument("--log_dir", type=str, default=defaults.training.log_dir)
    parser.add_argument("--model_dir", type=str, default=defaults.training.model_dir)
    parser.add_argument("--log_interval", type=int, default=defaults.training.log_interval)
    parser.add_argument("--eval_interval", type=int, default=defaults.training.eval_interval)
    parser.add_argument("--eval_episodes", type=int, default=defaults.training.eval_episodes)
    parser.add_argument("--eval_seed_stride", type=int, default=defaults.training.eval_seed_stride)
    parser.add_argument("--save_interval", type=int, default=defaults.training.save_interval)
    parser.add_argument(
        "--export_figures",
        type=_parse_bool_arg,
        default=True,
        help="Render post-training figures/GIFs; metrics and configs are always exported.",
    )

    parser.add_argument("--tx_power_choices_dbm", type=float, nargs="+", default=defaults.comm.tx_power_choices_dbm,
                        help="UAV RF output power choices; use one value for a fixed-power ablation.")
    parser.add_argument("--tx_power_dbm", type=float, default=None, help="Initial power, default middle choice.")
    return parser.parse_args(argv)


def build_config(args) -> Config:
    config = Config()
    config.comm.tx_power_choices_dbm = list(getattr(args, "tx_power_choices_dbm", config.comm.tx_power_choices_dbm))
    initial_power = getattr(args, "tx_power_dbm", None)
    config.comm.tx_power_dbm = (float(initial_power) if initial_power is not None else config.comm.tx_power_choices_dbm[len(config.comm.tx_power_choices_dbm) // 2])
    trainer_name = normalize_trainer(getattr(args, "trainer", "mappo"))

    config.training.num_envs = args.num_envs
    config.training.total_timesteps = args.total_timesteps
    config.training.episode_max_steps = int(args.episode_max_steps)
    config.training.num_minibatches = args.num_minibatches
    config.training.num_epochs = args.num_epochs
    config.training.lr_actor = args.lr_actor
    config.training.lr_critic = args.lr_critic
    config.training.seed = args.seed
    config.training.device = args.device
    config.training.vec_backend = str(args.vec_backend).strip().lower()
    config.training.log_dir = args.log_dir
    config.training.model_dir = args.model_dir
    config.training.log_interval = args.log_interval
    config.training.eval_interval = args.eval_interval
    config.training.eval_episodes = max(1, int(args.eval_episodes))
    config.training.eval_seed_stride = max(0, int(args.eval_seed_stride))
    config.training.save_interval = args.save_interval
    apply_scene_cli_overrides(config, args)
    config.scene.grid_spacing = float(args.grid_spacing)
    config.scene.uav_height = float(args.uav_height)
    config.scene.ugv_height = float(args.ugv_height)
    config.scene.building_height_m = float(args.building_height_m)
    config.scene.building_height_mode = args.building_height_mode
    config.scene.building_height_min_m = float(args.building_height_min_m)
    config.scene.building_height_max_m = float(args.building_height_max_m)
    config.scene.building_height_seed = int(args.building_height_seed)
    config.scene.building_split_min_area_m2 = float(args.building_split_min_area_m2)
    config.uav.max_energy = float(args.uav_max_energy)
    config.comm.los_excess_db = float(args.los_excess_db)
    config.comm.nlos_excess_db = float(args.nlos_excess_db)
    config.comm.nlos_length_loss_db_per_m = float(args.nlos_length_loss_db_per_m)
    config.comm.nlos_length_loss_cap_db = args.nlos_length_loss_cap_db
    config.comm.outage_snr_mode = args.outage_snr_mode
    config.comm.snr_outage_threshold_db = float(args.snr_outage_threshold_db)
    config.uav.total_bandwidth = float(args.total_bandwidth)
    config.comm.source_measurement_bits = int(args.source_measurement_bits)
    config.comm.data_per_sample = float(args.data_per_sample)
    config.uav.quantization_scheme = str(args.quantization_scheme)

    config.reward.alpha_nmse = args.alpha_nmse
    config.reward.nmse_signed_clip = args.nmse_signed_clip
    config.reward.gamma_queue = args.gamma_queue
    config.reward.lambda_uav_progress = args.lambda_uav_progress
    config.reward.lambda_uav_backtrack = args.lambda_uav_backtrack
    config.reward.lambda_ugv_progress = args.lambda_ugv_progress
    config.reward.lambda_ugv_backtrack = args.lambda_ugv_backtrack
    config.reward.lambda_novel_info = args.lambda_novel_info
    config.reward.lambda_full_repeat = args.lambda_full_repeat
    config.reward.bootstrap_progress_scale = args.bootstrap_progress_scale
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
    config.planner.incremental_outer_iters = max(1, int(args.incremental_outer_iters))
    config.planner.incremental_max_svt_iters = max(1, int(args.incremental_max_svt_iters))
    config.planner.ensemble_quality_weighted = bool(args.ensemble_quality_weighted)
    config.planner.target_arrival_radius_steps = max(0.0, float(args.target_arrival_radius_steps))
    config.planner.target_suppression_radius_steps = max(
        0.0,
        float(args.target_suppression_radius_steps),
    )
    config.planner.target_stuck_no_sample_steps = max(1, int(args.target_stuck_no_sample_steps))
    config.planner.target_mode = str(args.planner_target_mode).strip().lower() or config.planner.target_mode
    config.planner.initial_observation_mode = (
        str(args.initial_observation_mode).strip().lower() or config.planner.initial_observation_mode
    )
    config.planner.local_planner_radius = max(1, int(args.local_planner_radius))
    config.planner.hybrid_uncertainty_window_updates = max(1, int(args.hybrid_uncertainty_window_updates))
    config.planner.hybrid_uncertainty_improvement_threshold = max(
        0.0,
        float(args.hybrid_uncertainty_improvement_threshold),
    )
    config.planner.hybrid_global_hold_intervals = max(1, int(args.hybrid_global_hold_intervals))
    config.planner.hybrid_local_reentry_min_targets = max(0, int(args.hybrid_local_reentry_min_targets))
    config.planner.ugv_comm_target_mode = str(args.ugv_comm_target_mode).strip().lower()
    config.planner.ugv_comm_backlog_threshold = float(
        args.ugv_comm_backlog_threshold
    )
    config.planner.ugv_recovery_exit_backlog_threshold = float(
        args.ugv_recovery_exit_backlog_threshold
    )
    config.planner.ugv_recovery_poor_service_steps = int(
        args.ugv_recovery_poor_service_steps
    )
    config.planner.ugv_recovery_min_hold_steps = int(
        args.ugv_recovery_min_hold_steps
    )
    config.planner.ugv_recovery_good_link_steps = int(
        args.ugv_recovery_good_link_steps
    )
    config.planner.ugv_recovery_service_margin = float(
        args.ugv_recovery_service_margin
    )
    config.planner.ugv_comm_local_path_horizon = int(
        args.ugv_comm_local_path_horizon
    )
    config.planner.ugv_comm_expanded_path_horizon = int(
        args.ugv_comm_expanded_path_horizon
    )
    config.planner.ugv_comm_corridor_width = int(args.ugv_comm_corridor_width)
    config.planner.ugv_service_action_mask = bool(args.ugv_service_action_mask)
    config.planner.ugv_control_mode = ugv_control_mode_for_trainer(trainer_name)
    config.obs.include_remaining_time = bool(args.obs_remaining_time)
    config.obs.include_quant_context = bool(args.obs_quant_context)
    config.planner.prefill_percent = float(args.prefill_percent)
    config.planner.prefill_budget_basis = max(0, int(args.prefill_budget_basis))
    config.planner.init_building_clearance = max(0, int(args.init_building_clearance))
    config.planner.bootstrap_building_clearance = max(0, int(args.bootstrap_building_clearance))
    config.planner.flush_reconstruction_on_episode_end = bool(
        args.flush_reconstruction_on_episode_end
    )
    config.planner.iibtd_mu = float(args.iibtd_mu)
    config.planner.iibtd_backend = str(args.iibtd_backend).strip().lower() or config.planner.iibtd_backend
    config.planner.iibtd_device = str(args.iibtd_device).strip() or config.planner.iibtd_device
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
    from du_iibtd_based_fading_delta.shared.quant.sim_models import GridScene, IIBTD_opt, SimDataGen

    next_auto_idx = {"value": 0}
    if shared_data is None:
        shared_data_seed = int(config.training.seed if data_seed is None else data_seed)
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
            seed=env_config.training.seed + idx,
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
    from du_iibtd_based_fading_delta.shared.quant.sim_models import SimDataGen

    shared_data_seed = int(config.training.seed if data_seed is None else data_seed)
    return build_shared_data_suite(
        config,
        SimDataGen,
        shared_data_seed,
    )


def build_run_metadata(config: Config, cli_args: Optional[argparse.Namespace]) -> Dict[str, object]:
    """Build a JSON-serializable snapshot of the run settings."""
    metadata: Dict[str, object] = {
        "config": sanitize_json_payload(asdict(config)),
        "run_info": {
            "argv": sanitize_json_payload(list(sys.argv)),
            "cwd": os.getcwd(),
        },
    }
    if cli_args is not None:
        metadata["cli_args"] = sanitize_json_payload(vars(cli_args).copy())
    return sanitize_json_payload(metadata)


def write_json(path: str, payload: Dict[str, object]) -> str:
    """Write a JSON payload with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_json_payload(payload), f, indent=2, default=json_default)
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
        from du_iibtd_based_fading_delta.shared.quant.visualize import plot_all
        plot_all(log_dir=target_dir, target_nmse=target_nmse)
    return target_dir, dst_metrics


def train(config: Config, cli_args: Optional[argparse.Namespace] = None):
    mc = config.training
    trainer_name = normalize_trainer(getattr(cli_args, "trainer", "mappo"))
    expected_ugv_mode = ugv_control_mode_for_trainer(trainer_name)
    if config.planner.ugv_control_mode != expected_ugv_mode:
        raise ValueError(
            f"trainer={trainer_name!r} requires ugv_control_mode={expected_ugv_mode!r}, "
            f"got {config.planner.ugv_control_mode!r}"
        )
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
    if len(scene_indices) > 1 and int(mc.num_envs) != len(scene_indices):
        raise ValueError(
            f"num_envs must equal the scene suite length for multi-scene training; "
            f"got num_envs={int(mc.num_envs)}, scene_count={len(scene_indices)}."
        )
    shared_env_data = make_shared_env_data_suite(config)
    eval_config = deepcopy(config)
    eval_config.training.seed = int(config.training.seed + _EVAL_RESET_SEED_OFFSET)
    # Train and eval share the same DPM100PSD scene suite; reset seeds differ
    # so starts, prefill, and bootstrap targets can differ.
    run_metadata = build_run_metadata(config, cli_args)
    run_metadata.setdefault("run_info", {})["trainer"] = trainer_name
    run_metadata.setdefault("config", {}).setdefault("scene", {})[
        "radioseer_scene_indices"
    ] = list(scene_indices)
    logger = MetricsLogger(mc.log_dir, metadata=run_metadata)
    print_config_summary(config)
    print(
        f"[Scene Suite] RadioSeer scene specs: {format_scene_suite(scene_indices)}; "
        f"workers use env_idx % {len(scene_indices)}."
    )

    print("\n*** Using direct simulation data pipeline ***\n")
    # 创建虚拟环境
    eval_env_factory = make_env_factory(
        eval_config,
        shared_data=shared_env_data,
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
                f"[Eval] Reusing training RadioSeer scene suite "
                f"{format_scene_suite(scene_indices)}; reset seeds start at "
                f"{eval_config.training.seed} so starts/prefill can differ."
            )
        else:
            print(
                f"[Eval] Reusing training RadioSeer scene suite "
                f"{format_scene_suite(scene_indices)}; reset seed base "
                f"{eval_config.training.seed} with stride {int(mc.eval_seed_stride)}."
            )

    try:
        # 获取观测空间和动作空间维度
        obs_dims = vec_env.obs_dims
        action_dims = vec_env.action_dims
        print(f"Observation dims: {obs_dims}")
        print(f"Action dims: {action_dims}")

        if trainer_name == "mappo":
            policy = MAPPOPolicy(obs_dims, action_dims, mc)
            algorithm = MAPPO(policy, mc)
            buffer = RolloutBuffer(
                rollout_length=mc.episode_max_steps,
                num_envs=mc.num_envs,
                obs_dims=obs_dims,
                action_dims=action_dims,
                gamma=mc.gamma,
                gae_lambda=mc.gae_lambda,
            )
        elif trainer_name == "happo":
            policy = HAPPOPolicy(obs_dims, action_dims, mc)
            algorithm = HAPPO(policy, mc)
            buffer = HAPPORolloutBuffer(
                rollout_length=mc.episode_max_steps,
                num_envs=mc.num_envs,
                obs_dims=obs_dims,
                action_dims=action_dims,
                gamma=mc.gamma,
                gae_lambda=mc.gae_lambda,
            )
        elif trainer_name == "mappo_cf":
            policy = MAPPOCFPolicy(obs_dims, action_dims, mc)
            algorithm = MAPPOCF(policy, mc)
            buffer = MAPPOCFRolloutBuffer(
                rollout_length=mc.episode_max_steps,
                num_envs=mc.num_envs,
                obs_dims=obs_dims,
                action_dims=action_dims,
                gamma=mc.gamma,
                gae_lambda=mc.gae_lambda,
            )
        elif is_independent_ppo(trainer_name):
            policy = IPPOPolicy(obs_dims, action_dims, mc)
            algorithm = IPPO(policy, mc)
            buffer = IPPORolloutBuffer(
                rollout_length=mc.episode_max_steps,
                num_envs=mc.num_envs,
                obs_dims=obs_dims,
                action_dims=action_dims,
                gamma=mc.gamma,
                gae_lambda=mc.gae_lambda,
            )
        elif is_uav_only_ppo(trainer_name):
            policy = UAVPPOPolicy(obs_dims, action_dims, mc)
            algorithm = UAVPPO(policy, mc)
            buffer = RolloutBuffer(
                rollout_length=mc.episode_max_steps,
                num_envs=mc.num_envs,
                obs_dims=obs_dims,
                action_dims=action_dims,
                gamma=mc.gamma,
                gae_lambda=mc.gae_lambda,
            )
        else:  # normalize_trainer makes this unreachable.
            raise AssertionError(f"Unhandled trainer {trainer_name!r}")
        print(
            "[Runtime Devices] "
            f"{trainer_name.upper()} train/update={policy.device} | "
            f"policy action inference={policy.rollout_device} | "
            f"DU-IIBTD reconstruct during rollout={_resolve_iibtd_runtime_device(config)}"
        )

        transitions_per_update = mc.episode_max_steps * mc.num_envs
        if mc.num_minibatches <= 0:
            raise ValueError(f"num_minibatches must be positive, got {mc.num_minibatches}")
        if mc.num_minibatches > transitions_per_update:
            raise ValueError(
                f"num_minibatches={mc.num_minibatches} exceeds rollout batch size "
                f"{transitions_per_update}."
            )

        if mc.total_timesteps < transitions_per_update:
            raise ValueError(
                f"total_timesteps={mc.total_timesteps} is smaller than one full update "
                f"({transitions_per_update} = episode_max_steps*num_envs). "
                "Increase total_timesteps or reduce episode_max_steps/num_envs."
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
        pareto_eval_frontier: List[Dict[str, object]] = []
        last_eval_results: Optional[Dict[str, float]] = None
        last_eval_update: Optional[int] = None

        print(f"\nStarting training for {total_updates} updates ({mc.total_timesteps:,} total timesteps)")
        print(f"  Rollout: {mc.episode_max_steps} steps × {mc.num_envs} envs = {transitions_per_update} transitions/update\n")

        # 训练前 reset环境
        obs = vec_env.reset()
        # 初始化episode信息收集器
        accumulators, step_counts = make_episode_trackers(mc.num_envs, EPISODE_SUM_KEYS)

        for update in range(1, total_updates + 1):
            rollout_started = time.perf_counter()
            buffer.reset()
            episodes_finished = 0

            # 收集 rollout数据
            for _ in range(mc.episode_max_steps):
                # 获取当前policy下的动作 以及 critic 的value
                action_data = policy.get_actions(
                    uav_obs=obs["uav_obs"],
                    ugv_obs=obs["ugv_obs"],
                    critic_state=obs["critic_state"],
                    uav_action_mask=obs["uav_action_mask"],
                    ugv_action_mask=obs["ugv_action_mask"],
                )

                # 运行环境交互，获取 观测 奖励 等信息
                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(
                    uav_actions=action_data["uav_action"],
                    ugv_actions=action_data["ugv_action"],
                )
                dones = make_done_array(terminateds, truncateds)
                if not is_independent_ppo(trainer_name):
                    timeout_values = compute_timeout_bootstrap_values(
                        policy=policy,
                        infos=infos,
                        terminateds=terminateds,
                        truncateds=truncateds,
                    )
                else:
                    timeout_values = compute_ippo_timeout_bootstrap_values(
                        policy=policy,
                        infos=infos,
                        terminateds=terminateds,
                        truncateds=truncateds,
                    )

                # 将交互信息存储到buffer中
                if not is_independent_ppo(trainer_name):
                    buffer.add(
                        uav_obs=obs["uav_obs"],
                        ugv_obs=obs["ugv_obs"],
                        critic_state=obs["critic_state"],
                        uav_action=action_data["uav_action"],
                        ugv_action=action_data["ugv_action"],
                        uav_log_prob=action_data["uav_log_prob"],
                        ugv_log_prob=action_data["ugv_log_prob"],
                        uav_action_mask=obs["uav_action_mask"],
                        ugv_action_mask=obs["ugv_action_mask"],
                        reward=training_reward_for_trainer(
                            trainer_name,
                            rewards["team_reward"],
                            infos,
                        ),
                        value=action_data["value"],
                        done=dones,
                        terminated=terminateds.astype(np.float32),
                        truncated=truncateds.astype(np.float32),
                        timeout_value=timeout_values,
                    )
                else:
                    buffer.add(
                        uav_obs=obs["uav_obs"],
                        ugv_obs=obs["ugv_obs"],
                        uav_action=action_data["uav_action"],
                        ugv_action=action_data["ugv_action"],
                        uav_log_prob=action_data["uav_log_prob"],
                        ugv_log_prob=action_data["ugv_log_prob"],
                        uav_action_mask=obs["uav_action_mask"],
                        ugv_action_mask=obs["ugv_action_mask"],
                        # Fair MAPPO/IPPO comparison: both optimize the same
                        # cooperative team objective; only critic visibility differs.
                        uav_reward=rewards["team_reward"],
                        ugv_reward=rewards["team_reward"],
                        uav_value=action_data["uav_value"],
                        ugv_value=action_data["ugv_value"],
                        done=dones,
                        terminated=terminateds.astype(np.float32),
                        truncated=truncateds.astype(np.float32),
                        uav_timeout_value=timeout_values["uav_timeout_value"],
                        ugv_timeout_value=timeout_values["ugv_timeout_value"],
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

            rollout_seconds = time.perf_counter() - rollout_started
            optimization_started = time.perf_counter()
            # rollout结束后 计算critic value
            if not is_independent_ppo(trainer_name):
                last_value = policy.get_value(obs["critic_state"])
                buffer.compute_returns_and_advantages(last_value)
            else:
                last_values = policy.get_values(
                    uav_obs=obs["uav_obs"],
                    ugv_obs=obs["ugv_obs"],
                )
                buffer.compute_returns_and_advantages(
                    last_uav_value=last_values["uav_value"],
                    last_ugv_value=last_values["ugv_value"],
                )
            update_metrics = algorithm.update(buffer)
            buffer.release_cached_tensors()
            update_metrics["rollout_seconds"] = rollout_seconds
            update_metrics["optimization_seconds"] = time.perf_counter() - optimization_started

            if mc.log_interval > 0 and update % mc.log_interval == 0:
                update_metrics["global_step"] = global_step
                logger.log_update(update, update_metrics)

            if mc.eval_interval > 0 and update % mc.eval_interval == 0:
                eval_seed_base = int(
                    eval_config.training.seed + update * max(int(mc.eval_seed_stride), 0)
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
                progress_metrics_path = logger.save(
                    os.path.join(mc.log_dir, "metrics_in_progress.json")
                )
                print(f"Saved in-progress metrics: {progress_metrics_path}")
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
                    update=int(update),
                    higher_is_better=False,
                )
                if saved_path is not None:
                    best_eval_nmse_update = int(update)
                    best_eval_nmse_path = saved_path

                eval_mean_return = float(eval_results.get("eval_mean_return", np.nan))
                best_eval_return, saved_path = save_best_eval_checkpoint(
                    policy=policy,
                    model_dir=mc.model_dir,
                    checkpoint_name="best_return.pt",
                    metric_name="eval_mean_return",
                    metric_value=eval_mean_return,
                    best_value=best_eval_return,
                    update=int(update),
                    higher_is_better=True,
                )
                if saved_path is not None:
                    best_eval_return_update = int(update)
                    best_eval_return_path = saved_path

                pareto_eval_frontier, _ = update_pareto_eval_checkpoints(
                    policy=policy,
                    model_dir=mc.model_dir,
                    frontier=pareto_eval_frontier,
                    update=int(update),
                    eval_results=eval_results,
                )

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

        if mc.eval_interval > 0:
            final_eval_seed_base = int(
                eval_config.training.seed + total_updates * max(int(mc.eval_seed_stride), 0)
            )
            if last_eval_results is not None and last_eval_update == int(total_updates):
                final_eval_results = last_eval_results
                print(
                    f"\n[Final Eval] Reusing evaluation already run at update {total_updates} "
                    f"on training RadioSeer scene suite {format_scene_suite(scene_indices)} "
                    f"with reset seed base {final_eval_seed_base}"
                )
            else:
                print(
                    f"\n[Final Eval] Running final evaluation at update {total_updates} "
                    f"on training RadioSeer scene suite {format_scene_suite(scene_indices)} "
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
        else:
            print("\n[Final Eval] Skipping final evaluation because eval_interval <= 0.")

        should_render_figures = bool(
            getattr(cli_args, "export_figures", True)
            and (mc.log_interval > 0 or mc.eval_interval > 0)
        )
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
        if pareto_eval_frontier:
            print(
                f"Pareto archive contains {len(pareto_eval_frontier)} feasible "
                f"NMSE/link-bits checkpoint(s) in {mc.model_dir}."
            )
        print(f"Metrics saved to {metrics_path}")
        if run_metadata:
            print(f"Config saved to {os.path.join(export_dir, 'config.json')}")
            if "cli_args" in run_metadata:
                print(f"CLI args saved to {os.path.join(export_dir, 'cli_args.json')}")
        if should_render_figures:
            print(f"Auto figures saved to {os.path.join(export_dir, 'figures')}")
        else:
            print("Auto figure export skipped; metrics/config export is complete.")
    finally:
        for eval_env in eval_envs:
            if hasattr(eval_env, "close"):
                eval_env.close()
        if hasattr(vec_env, "close"):
            vec_env.close()


def main(argv=None, forced_trainer: Optional[str] = None) -> None:
    args = parse_args(argv)
    if forced_trainer is not None:
        args.trainer = normalize_trainer(forced_trainer)
    config = build_config(args)
    train(config, cli_args=args)


if __name__ == "__main__":
    main()

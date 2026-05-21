"""Run the total-random UAV-UGV quantized-sampling baseline."""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_MEMBER_DIR = os.path.dirname(_THIS_DIR)
_CODE_DIR = os.path.dirname(_SHARED_MEMBER_DIR)
for _path in (_CODE_DIR, _SHARED_MEMBER_DIR, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

def _configure_matplotlib_cache_dir() -> None:
    """Avoid matplotlib cache writes to an unwritable HOME during export."""
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

from config import Config
from environment import UAVUGVEnvironment
from random_policy import RandomPolicy
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

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in the main runtime env
    torch = None


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
    parser = argparse.ArgumentParser(description="Total-random UAV-UGV quantized-sampling baseline")
    defaults = Config()

    parser.add_argument("--episode_max_steps", type=int, default=defaults.run.episode_max_steps)
    parser.add_argument("--seed", type=int, default=defaults.run.seed)
    parser.add_argument("--device", type=str, default=defaults.run.device)
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
    parser.add_argument(
        "--quant_bits",
        type=int,
        nargs="+",
        default=list(defaults.uav.quant_bits),
        help="Discrete UAV sensing quantization bit-depth choices.",
    )
    parser.add_argument(
        "--default_quant_bits",
        type=int,
        default=defaults.uav.default_quant_bits,
        help="Initial/default UAV sensing quantization bit depth.",
    )

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
    parser.add_argument("--lambda_ugv_progress", type=float, default=defaults.reward.lambda_ugv_progress)
    parser.add_argument("--lambda_ugv_backtrack", type=float, default=defaults.reward.lambda_ugv_backtrack)
    parser.add_argument("--ugv_progress_uav_weight", type=float, default=defaults.reward.ugv_progress_uav_weight)
    parser.add_argument("--ugv_progress_target_weight", type=float, default=defaults.reward.ugv_progress_target_weight)
    parser.add_argument("--bootstrap_progress_scale", type=float, default=defaults.reward.bootstrap_progress_scale)
    parser.add_argument("--lambda_spatial_revisit", type=float, default=defaults.reward.lambda_spatial_revisit)
    parser.add_argument("--local_goal_arrival_bonus", type=float, default=defaults.reward.local_goal_arrival_bonus)
    parser.add_argument("--terminal_failure_penalty", type=float, default=defaults.reward.terminal_failure_penalty)

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
        "--hybrid_nmse_stall_steps",
        type=int,
        default=defaults.planner.hybrid_nmse_stall_steps,
        help="Hybrid mode: switch local->global after this many low-improvement map updates.",
    )
    parser.add_argument(
        "--hybrid_nmse_stall_threshold",
        type=float,
        default=defaults.planner.hybrid_nmse_stall_threshold,
        help="Hybrid mode: treat planner NMSE improvement below this threshold as stalled.",
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
            "Optional DU-IIBTD adapter checkpoint list used when "
            "--iibtd_backend is du_iibtd_res_sr or du_iibtd_res_sr_learn_nu."
        ),
    )

    parser.add_argument("--log_dir", type=str, default=defaults.run.log_dir)
    parser.add_argument("--eval_episodes", type=int, default=defaults.run.eval_episodes)

    return parser.parse_args()


def build_config(args) -> Config:
    config = Config()

    config.run.episode_max_steps = args.episode_max_steps
    config.run.seed = args.seed
    config.run.device = args.device
    config.run.log_dir = args.log_dir
    config.run.eval_episodes = args.eval_episodes

    apply_scene_cli_overrides(config, args)
    config.scene.uav_height = float(args.uav_height)
    config.scene.ugv_height = float(args.ugv_height)
    config.scene.building_height_m = float(args.building_height_m)
    config.uav.quant_bits = [int(bits) for bits in args.quant_bits]
    config.uav.default_quant_bits = int(args.default_quant_bits)

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
    config.reward.lambda_ugv_progress = args.lambda_ugv_progress
    config.reward.lambda_ugv_backtrack = args.lambda_ugv_backtrack
    config.reward.ugv_progress_uav_weight = args.ugv_progress_uav_weight
    config.reward.ugv_progress_target_weight = args.ugv_progress_target_weight
    config.reward.bootstrap_progress_scale = args.bootstrap_progress_scale
    config.reward.lambda_spatial_revisit = args.lambda_spatial_revisit
    config.reward.local_goal_arrival_bonus = args.local_goal_arrival_bonus
    config.reward.terminal_failure_penalty = args.terminal_failure_penalty

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
    config.planner.target_mode = str(args.planner_target_mode).strip().lower() or config.planner.target_mode
    config.planner.initial_observation_mode = (
        str(args.initial_observation_mode).strip().lower() or config.planner.initial_observation_mode
    )
    config.planner.local_planner_radius = max(1, int(args.local_planner_radius))
    config.planner.hybrid_nmse_stall_steps = max(1, int(args.hybrid_nmse_stall_steps))
    config.planner.hybrid_nmse_stall_threshold = max(0.0, float(args.hybrid_nmse_stall_threshold))
    config.planner.hybrid_global_hold_intervals = max(1, int(args.hybrid_global_hold_intervals))
    config.planner.hybrid_local_reentry_min_targets = max(0, int(args.hybrid_local_reentry_min_targets))
    config.planner.prefill_percent = float(args.prefill_percent)
    config.planner.prefill_budget_basis = max(0, int(args.prefill_budget_basis))
    config.planner.init_building_clearance = max(0, int(args.init_building_clearance))
    config.planner.bootstrap_building_clearance = max(0, int(args.bootstrap_building_clearance))
    config.planner.flush_reconstruction_on_episode_end = bool(args.flush_reconstruction_on_episode_end)
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
        shared_data_seed = int(config.run.seed if data_seed is None else data_seed)
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
            seed=env_config.run.seed + idx,
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

    shared_data_seed = int(config.run.seed if data_seed is None else data_seed)
    return build_shared_data_suite(
        config,
        SimDataGen,
        shared_data_seed,
    )


def build_run_metadata(config: Config, cli_args: Optional[argparse.Namespace]) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "config": asdict(config),
        "run_info": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "control_policy": "uniform_random_legal_actions_with_quantized_sensing",
        },
    }
    if cli_args is not None:
        metadata["cli_args"] = vars(cli_args).copy()
    return metadata


def write_json(path: str, payload: Dict[str, object]) -> str:
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
    if _resolve_iibtd_runtime_backend(config) == "cpu":
        return "cpu"
    device = str(config.planner.iibtd_device).strip()
    if not device or device.lower() == "auto":
        return _normalize_device_string(config.run.device)
    return _normalize_device_string(device)


def _apply_runtime_backend_policy(config: Config) -> Optional[str]:
    iibtd_backend = _resolve_iibtd_runtime_backend(config)
    cuda_count = _cuda_device_count()
    if iibtd_backend != "gpu" or cuda_count <= 0:
        return None

    planner_device_arg = str(config.planner.iibtd_device).strip().lower()
    if not planner_device_arg or planner_device_arg in {"auto", "cuda"}:
        config.planner.iibtd_device = _normalize_device_string("cuda", default_cuda_index=0)
    return f"II-BTD reconstruct uses {_resolve_iibtd_runtime_device(config)}"


def run_random_baseline(config: Config, cli_args: Optional[argparse.Namespace] = None):
    rc = config.run
    runtime_policy_message = _apply_runtime_backend_policy(config)
    if runtime_policy_message is not None:
        print(f"[Runtime Policy] {runtime_policy_message}")

    set_seeds(rc.seed)
    scene_indices = configure_scene_suite_config(config)
    shared_env_data = make_shared_env_data_suite(config)
    run_metadata = build_run_metadata(config, cli_args)
    run_metadata.setdefault("config", {}).setdefault("scene", {})[
        "radioseer_scene_indices"
    ] = list(scene_indices)
    logger = MetricsLogger(rc.log_dir, metadata=run_metadata)
    print_config_summary(config)
    print(f"[Scene Suite] DPM100PSD manifest rows: {format_scene_suite(scene_indices)}")

    print("\n*** Using direct simulation data pipeline ***\n")
    eval_env_factory = make_env_factory(config, shared_data=shared_env_data)
    eval_envs = [
        eval_env_factory(scene_pos)
        for scene_pos in range(len(shared_env_data))
    ]

    try:
        obs_dims = eval_envs[0].get_obs_dims()
        action_dims = eval_envs[0].get_action_dims()
        print(f"Observation dims: {obs_dims}")
        print(f"Action dims: {action_dims}")
        print(
            "[Policy] UAV and UGV sample uniformly from legal action_mask entries; "
            f"II-BTD reconstruct device={_resolve_iibtd_runtime_device(config)}"
        )

        print(
            f"\nStarting direct random-policy evaluation: "
            f"{rc.eval_episodes} episodes x up to {rc.episode_max_steps} steps"
        )

        eval_policy = RandomPolicy(action_dims=action_dims, seed=rc.seed)
        final_eval_results = evaluate_scene_suite(
            envs=eval_envs,
            evaluate_fn=evaluate_policy,
            policy=eval_policy,
            num_episodes=rc.eval_episodes,
            max_steps=rc.episode_max_steps,
            seed_base=rc.seed,
        )
        logger.log_final_eval(0, final_eval_results)

        export_dir, metrics_path = export_timestamped_figures(
            logger=logger,
            source_log_dir=rc.log_dir,
            target_nmse=float(config.reward.accuracy_target_nmse),
            run_metadata=run_metadata,
            render_figures=True,
        )
        print("\nRandom baseline complete.")
        print(f"Metrics saved to {metrics_path}")
        if run_metadata:
            print(f"Config saved to {os.path.join(export_dir, 'config.json')}")
            if "cli_args" in run_metadata:
                print(f"CLI args saved to {os.path.join(export_dir, 'cli_args.json')}")
        print(f"Auto figures saved to {os.path.join(export_dir, 'figures')}")
    finally:
        for eval_env in eval_envs:
            if hasattr(eval_env, "close"):
                eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    run_random_baseline(config, cli_args=args)

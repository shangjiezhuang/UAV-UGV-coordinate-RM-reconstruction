"""Fail fast when a formal run is not comparable to the clean A* protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENES = [
    8513,
    1807,
    1579,
    1371,
    10001,
    "FARMOmniDPM100PSD_251:905",
    "FARMOmniDPM100PSD_251:43",
    "FARMOmniDPM100PSD_251:705",
]
UGV_MODES = {
    "mappo": "policy",
    "ippo": "policy",
    "happo": "policy",
    "mappo_cf": "policy",
    "ppo_fixed": "fixed",
    "ppo_heuristic": "legacy_heuristic",
    "ppo_astar": "astar_target",
    "ppo_astar_support": "astar_support",
    "ppo_astar_2path_support": "astar_2path_support",
    "ppo_astar_2path_support_recovery": "astar_2path_support_recovery",
}


def get(data, dotted):
    value = data
    for key in dotted.split("."):
        value = value[key]
    return value


def equal(actual, expected):
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--variant", choices=("quant", "noquant"), required=True)
    parser.add_argument("--trainer", choices=tuple(UGV_MODES), required=True)
    parser.add_argument("--ugv-weight", type=float, required=True)
    parser.add_argument("--full-repeat-weight", type=float, required=True)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "scene.grid_size": [100, 100],
        "scene.grid_spacing": 2,
        "scene.building_height_m": 25.0,
        "scene.radioseer_scene_indices": SCENES,
        "uav.num_directions": 5,
        "uav.max_energy": 9000.0,
        "uav.bandwidth_ratios": [0.2, 0.3, 0.5, 0.6],
        "uav.small_scale_fading_std": 0.01,
        "uav.receiver_noise_std": 0.01,
        "ugv.num_directions": 5,
        "comm.data_per_sample": 8_000_000.0,
        "comm.source_measurement_bits": 16,
        "comm.snr_outage_threshold_db": -5.0,
        "reward.alpha_nmse": 20.0,
        "reward.nmse_signed_clip": 0.25,
        "reward.gamma_queue": 1.5,
        "reward.lambda_uav_progress": 2.0,
        "reward.lambda_uav_backtrack": 2.0,
        "reward.lambda_ugv_progress": args.ugv_weight,
        "reward.lambda_ugv_backtrack": args.ugv_weight,
        "reward.lambda_novel_info": 0.0,
        "reward.lambda_full_repeat": args.full_repeat_weight,
        "obs.include_remaining_time": False,
        "obs.num_planner_features": 5,
        "planner.target_mode": "hybrid",
        "planner.initial_observation_mode": "prefill",
        "planner.local_planner_radius": 15,
        "planner.hybrid_uncertainty_stall_steps": 2,
        "planner.hybrid_uncertainty_improvement_threshold": 0.08,
        "planner.hybrid_global_hold_intervals": 5,
        "planner.hybrid_local_reentry_min_targets": 2,
        "planner.ugv_comm_backlog_threshold": 0.5,
        "planner.ugv_recovery_exit_backlog_threshold": 0.2,
        "planner.ugv_recovery_poor_service_steps": 2,
        "planner.ugv_recovery_min_hold_steps": 2,
        "planner.ugv_recovery_good_link_steps": 2,
        "planner.ugv_recovery_service_margin": 1.0,
        "planner.ugv_service_action_mask": False,
        "planner.ugv_control_mode": UGV_MODES[args.trainer],
        "planner.prefill_percent": 5.0,
        "planner.prefill_budget_basis": 200,
        "planner.ensemble_refresh_interval": 3,
        "planner.ensemble_full_refresh_interval": 0,
        "planner.nmse_refresh_delta": 0.1,
        "planner.incremental_outer_iters": 2,
        "planner.incremental_max_svt_iters": 20,
        "mappo.num_envs": 8,
        "mappo.total_timesteps": 192000,
        "mappo.episode_max_steps": 160,
        "mappo.num_minibatches": 4,
        "mappo.num_epochs": 6,
        "mappo.lr_actor": 0.0001,
        "mappo.lr_critic": 0.0001,
        "mappo.seed": 42,
        "mappo.device": "cuda:0",
        "mappo.vec_backend": "subproc",
        "mappo.eval_seed_stride": 10000,
    }
    if args.variant == "quant":
        expected.update(
            {
                "obs.include_quant_context": False,
                "uav.quant_bits": [10, 8, 6],
                "uav.quantization_scheme": "log_first",
            }
        )

    errors = []
    for field, wanted in expected.items():
        try:
            actual = get(config, field)
        except KeyError:
            errors.append(f"missing {field}")
            continue
        if not equal(actual, wanted):
            errors.append(f"{field}: expected {wanted!r}, got {actual!r}")

    direction_count = int(get(config, "uav.num_directions"))
    bandwidth_count = len(get(config, "uav.bandwidth_ratios"))
    if args.variant == "quant":
        action_dim = direction_count * bandwidth_count * len(get(config, "uav.quant_bits"))
        expected_action_dim = 60
    else:
        action_dim = direction_count * bandwidth_count
        expected_action_dim = 20
    if action_dim != expected_action_dim:
        errors.append(f"UAV action dimension: expected {expected_action_dim}, got {action_dim}")

    log_text = Path(args.train_log).read_text(encoding="utf-8", errors="replace")
    obs_contract = "Observation dims: {'uav_obs': 14, 'ugv_obs': 14, 'critic_state': 19}"
    action_contract = f"'uav_action': {expected_action_dim}"
    if obs_contract not in log_text:
        errors.append("training log does not confirm observation dims 14/14/19")
    if action_contract not in log_text:
        errors.append(f"training log does not confirm UAV action dim {expected_action_dim}")

    if errors:
        raise SystemExit("[CONTRACT FAIL]\n- " + "\n- ".join(errors))
    print(
        f"[CONTRACT PASS] trainer={args.trainer} variant={args.variant} "
        f"obs=14/14/19 uav_action={expected_action_dim} ugv_action=5"
    )


if __name__ == "__main__":
    main()

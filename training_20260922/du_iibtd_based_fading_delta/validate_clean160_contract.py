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
    parser.add_argument("--grid-spacing", type=float, default=2.0,
                        help="Expected meters per cell for the formal training configuration.")
    parser.add_argument("--energy-profile", choices=("current", "legacy-9000"), default="current",
                        help="Current: 8500 J; legacy-9000 explicitly audits earlier 9000 J experiments.")
    parser.add_argument(
        "--bandwidth-profile", choices=("current", "legacy-four"), default="current",
        help="Use legacy-four only when auditing results produced before 2026-09-14.",
    )
    parser.add_argument("--power-profile", choices=("current", "legacy-fixed", "legacy-202730"), default="current",
                        help="Current: 9/12/15 dBm; legacy-202730: former power actions; legacy-fixed: old 1 dBm without TX energy.")
    parser.add_argument("--height-profile", choices=("current", "legacy-fixed", "legacy-continuous", "legacy-integer"), default="current",
                        help="Current: UAV 50 m, integer buildings 47-53 m; legacy profiles retain UAV 30 m and old geometry.")
    parser.add_argument("--outage-profile", choices=("current", "legacy-nominal"), default="current",
                        help="legacy-nominal explicitly audits results predating the received-SNR outage gate.")
    parser.add_argument("--resource-profile", choices=("current", "legacy-100mhz-ref16"), default="current")
    parser.add_argument("--loss-profile", choices=("current", "legacy-1-20", "legacy-0-15"), default="current",
                        help="Current: 1.6/23 dB plus 0.22 dB/m (uncapped); legacy profiles disable the length term.")
    args = parser.parse_args()
    expected_bandwidth_ratios = (
        [0.3, 0.5, 0.6] if args.bandwidth_profile == "current"
        else [0.2, 0.3, 0.5, 0.6]
    )

    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "scene.grid_size": [100, 100],
        "scene.grid_spacing": float(args.grid_spacing),
        "scene.uav_height": 50.0 if args.height_profile == "current" else 30.0,
        "scene.ugv_height": 0.0,
        "scene.building_height_m": 47.0 if args.height_profile == "current" else 25.0,
        "scene.radioseer_scene_indices": SCENES,
        "uav.num_directions": 5,
        "uav.max_energy": 8500.0 if args.energy_profile == "current" else 9000.0,
        "uav.step_size": 4.0,
        "uav.flight_power": 12.0,
        "uav.hover_power": 8.0,
        "uav.sensing_power": 5.0,
        "uav.step_duration": 1.0,
        "uav.bandwidth_ratios": expected_bandwidth_ratios,
        "uav.small_scale_fading_std": 0.01,
        "uav.receiver_noise_std": 0.01,
        "ugv.num_directions": 5,
        "uav.total_bandwidth": 50_000_000.0 if args.resource_profile == "current" else 100_000_000.0,
        "comm.data_per_sample": 8_000_000.0,
        "comm.source_measurement_bits": 32 if args.resource_profile == "current" else 16,
        "comm.snr_outage_threshold_db": -5.0,
        "comm.carrier_freq": 3.5e9,
        "comm.noise_figure_db": 8.0,
        "comm.shadow_std_los_db": 2.0,
        "comm.shadow_std_nlos_db": 6.0,
        "comm.los_excess_db": 1.6 if args.loss_profile == "current" else (0.0 if args.loss_profile == "legacy-0-15" else 1.0),
        "comm.nlos_excess_db": 23.0 if args.loss_profile == "current" else (15.0 if args.loss_profile == "legacy-0-15" else 20.0),
        "comm.nlos_length_loss_db_per_m": 0.22 if args.loss_profile == "current" else 0.0,
        "comm.nlos_length_loss_cap_db": None if args.loss_profile == "current" else 0.0,
        "uav.queue_capacity_bits": 512_000_000.0,
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
        "planner.hybrid_uncertainty_window_updates": 2,
        "planner.hybrid_uncertainty_improvement_threshold": 0.03,
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
        "planner.hybrid_switch_metric": "sum_relative_frobenius",
        "planner.reconstruction_refresh_mode": "local_to_global",
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

    if args.power_profile == "current":
        expected.update({"comm.tx_power_choices_dbm": [9.0, 12.0, 15.0],
                         "comm.tx_power_dbm": 12.0, "comm.tx_energy_enabled": True})
    elif args.power_profile == "legacy-202730":
        expected.update({"comm.tx_power_choices_dbm": [20.0, 27.0, 30.0],
                         "comm.tx_power_dbm": 27.0, "comm.tx_energy_enabled": True})
    else:
        expected["comm.tx_power_dbm"] = 1.0
        expected["comm.tx_power_choices_dbm"] = [1.0]
        expected["comm.tx_energy_enabled"] = False
        config["comm"].setdefault("tx_power_choices_dbm", [config["comm"]["tx_power_dbm"]])
        config["comm"].setdefault("tx_energy_enabled", False)
    if args.height_profile == "current":
        expected["scene.building_split_min_area_m2"] = 1200.0
    if args.height_profile in ("current", "legacy-continuous", "legacy-integer"):
        expected.update({"scene.building_height_mode": ("uniform_per_building" if args.height_profile == "legacy-continuous" else "uniform_integer_per_building"),
                         "scene.building_height_min_m": 47.0 if args.height_profile == "current" else 25.0,
                         "scene.building_height_max_m": 53.0 if args.height_profile == "current" else 29.0,
                         "scene.building_height_seed": 42})
    else:
        expected["scene.building_height_mode"] = "fixed"
        config["scene"].setdefault("building_height_mode", "fixed")
    expected["comm.outage_snr_mode"] = "received" if args.outage_profile == "current" else "nominal"
    if args.outage_profile == "legacy-nominal":
        config["comm"].setdefault("outage_snr_mode", "nominal")
    if args.loss_profile == "legacy-0-15":
        config["comm"].setdefault("los_excess_db", 0.0)
    if args.loss_profile != "current":
        config["comm"].setdefault("nlos_length_loss_db_per_m", 0.0)
        config["comm"].setdefault("nlos_length_loss_cap_db", 0.0)
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
        expected_action_dim = 5 * len(expected_bandwidth_ratios) * 3
    else:
        action_dim = direction_count * bandwidth_count
        expected_action_dim = 5 * len(expected_bandwidth_ratios)
    action_dim *= len(config["comm"].get("tx_power_choices_dbm", [config["comm"]["tx_power_dbm"]]))
    expected_action_dim *= 1 if args.power_profile == "legacy-fixed" else 3
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

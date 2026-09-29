"""Evaluate non-learning controllers in the shared based environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.evaluation_common import (  # noqa: E402
    apply_saved_config,
    compact_eval_result,
    json_default,
    scene_artifacts_payload,
)
from du_iibtd_based_fading_delta.fair_baseline_policies import (  # noqa: E402
    BASELINE_CONTROLLERS,
    make_baseline_policy,
    normalize_baseline_controller,
    ugv_control_mode_for_baseline,
)
from du_iibtd_based_fading_delta.scene_suite import aggregate_eval_results  # noqa: E402
from du_iibtd_based_fading_delta.energy_tradeoff import summarize_energy_episode_records  # noqa: E402
from du_iibtd_based_fading_delta.ugv_control import uses_service_action_mask  # noqa: E402
from du_iibtd_based_fading_delta.Greedy_heuristic_quant.greedy_policy import GreedyPolicyConfig


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("quant", "noquant"), default="quant")
    parser.add_argument("--controller", choices=BASELINE_CONTROLLERS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iibtd-device", default="cuda:0")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=200_042)
    parser.add_argument("--scene-seed-stride", type=int, default=10_000)
    parser.add_argument("--local-planner-radius", type=int, default=15)
    parser.add_argument("--grid-spacing", "--grid_spacing", type=float, default=None,
                        help="Override saved meters per cell explicitly; otherwise preserve the saved scene scale.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.num_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("num-episodes and max-steps must be positive")

    if args.variant == "quant":
        from du_iibtd_based_fading_delta.shared.quant import runner as train
        from du_iibtd_based_fading_delta.shared.quant.config import Config
        from du_iibtd_based_fading_delta.shared.quant.utils import evaluate_policy
    else:
        from du_iibtd_based_fading_delta.shared.noquant import runner as train
        from du_iibtd_based_fading_delta.shared.noquant.config import Config
        from du_iibtd_based_fading_delta.shared.noquant.utils import evaluate_policy

    controller = normalize_baseline_controller(args.controller)

    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = Config()
    apply_saved_config(config, raw_config)
    if args.grid_spacing is not None:
        config.scene.grid_spacing = float(args.grid_spacing)
    config.training.device = str(args.device)
    config.planner.iibtd_device = str(args.iibtd_device)
    config.planner.local_planner_radius = int(args.local_planner_radius)
    config.training.episode_max_steps = int(args.max_steps)
    config.planner.ugv_control_mode = ugv_control_mode_for_baseline(controller)
    if controller == "total_random":
        config.planner.ugv_service_action_mask = False
    config.__post_init__()

    train.set_seeds(int(config.training.seed))
    shared_data = train.make_shared_env_data_suite(config)
    env_factory = train.make_env_factory(config, shared_data=shared_data)
    envs = [env_factory(index) for index in range(len(shared_data))]
    scene_results = []
    try:
        for scene_position, env in enumerate(envs):
            reset_seed = int(args.seed_base) + scene_position * int(
                args.scene_seed_stride
            )
            policy = make_baseline_policy(controller, seed=reset_seed)
            result = evaluate_policy(
                env=env,
                policy=policy,
                num_episodes=int(args.num_episodes),
                max_steps=int(args.max_steps),
                seed_base=reset_seed,
            )
            scene_results.append(result)
            print(
                f"[Baseline] controller={controller} scene={scene_position + 1}/"
                f"{len(envs)} sample={result['eval_scene_sample_index']} "
                f"nmse={float(result['eval_mean_nmse']):.6f} "
                f"link_Gbit={float(result['eval_mean_link_transmitted_bits']) * 1e-9:.3f}",
                flush=True,
            )
    finally:
        for env in envs:
            env.close()

    aggregate = aggregate_eval_results(scene_results, seed_base=int(args.seed_base))
    energy_tradeoff = summarize_energy_episode_records(
        aggregate["eval_episode_records"]
    )
    payload = {
        "evaluation_config": {
            "hybrid_switch_metric": config.planner.hybrid_switch_metric,
            "hybrid_uncertainty_window_updates": config.planner.hybrid_uncertainty_window_updates,
            "hybrid_uncertainty_improvement_threshold": config.planner.hybrid_uncertainty_improvement_threshold,
            "reconstruction_refresh_mode": config.planner.reconstruction_refresh_mode,
            "hybrid_global_hold_steps": config.planner.hybrid_global_hold_intervals * config.planner.ensemble_refresh_interval,
            "variant": str(args.variant),
            "controller": str(controller),
            "grid_spacing_m": float(config.scene.grid_spacing),
            "greedy_uncertainty_scope": "local_manhattan" if controller.startswith("greedy") else None,
            "greedy_uncertainty_radius_cells": int(GreedyPolicyConfig().uncertainty_radius) if controller.startswith("greedy") else None,
            "source_config": str(args.config),
            "device": str(args.device),
            "iibtd_device": str(args.iibtd_device),
            "num_episodes_per_scene": int(args.num_episodes),
            "max_steps": int(args.max_steps),
            "local_planner_radius": int(config.planner.local_planner_radius),
            "uav_max_energy": float(config.uav.max_energy),
            "outage_snr_mode": str(config.comm.outage_snr_mode),
            "snr_outage_threshold_db": float(config.comm.snr_outage_threshold_db),
            "source_measurement_bits": int(config.comm.source_measurement_bits),
            "data_per_sample_bits": float(config.comm.data_per_sample),
            "queue_capacity_bits": float(config.uav.queue_capacity_bits),
            "prefill_percent": float(config.planner.prefill_percent),
            "prefill_budget_basis": int(config.planner.prefill_budget_basis),
            "physical_link_reward_penalty": False,
            "ugv_control_mode": str(config.planner.ugv_control_mode),
            "ugv_service_action_mask_effective": bool(
                uses_service_action_mask(config)
            ),
            "seed_base": int(args.seed_base),
            "scene_seed_stride": int(args.scene_seed_stride),
        },
        "aggregate": aggregate,
        "energy_nmse_link_pairs": energy_tradeoff,
        "per_scene": [compact_eval_result(result) for result in scene_results],
        "scene_artifacts": scene_artifacts_payload(scene_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"[Baseline] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

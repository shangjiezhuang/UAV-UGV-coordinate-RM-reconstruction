"""Evaluate any saved learned-policy checkpoint under current based semantics.

This entrypoint is intentionally evaluation-only.  It reloads the saved run
configuration, then applies explicit outage/prefill overrides so an old policy
can be replayed against the repaired communication and reconstruction pipeline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.scene_suite import aggregate_eval_results
from du_iibtd_based_fading_delta.energy_tradeoff import summarize_energy_episode_records
from du_iibtd_based_fading_delta.evaluation_common import (
    apply_saved_config as _apply_saved_config,
    compact_eval_result,
    json_default as _json_default,
    scene_artifacts_payload,
)
from du_iibtd_based_fading_delta.fair_training import (
    LEARNED_TRAINERS,
    is_uav_only_ppo,
    ugv_control_mode_for_trainer,
)
from du_iibtd_based_fading_delta.uav_ppo import UAVPPOPolicy
from du_iibtd_based_fading_delta.HAPPO.networks import HAPPOPolicy
from du_iibtd_based_fading_delta.MAPPO_CF.networks import MAPPOCFPolicy
from du_iibtd_based_fading_delta.fair_baseline_policies import GuidanceGreedyUGVOverridePolicy
from du_iibtd_based_fading_delta.training_common import parse_bool_arg as _parse_bool_arg
from du_iibtd_based_fading_delta.ugv_control import uses_service_action_mask


RUNTIME_VARIANT_PACKAGES = {
    "noquant": "du_iibtd_based_fading_delta.shared.noquant",
    "quant": "du_iibtd_based_fading_delta.shared.quant",
}
MAPPO_NETWORK_PACKAGES = {
    "noquant": "du_iibtd_based_fading_delta.MAPPO_noquant.networks",
    "quant": "du_iibtd_based_fading_delta.MAPPO_quant.networks",
}
# Compatibility name used by external evaluation scripts.
VARIANT_PACKAGES = RUNTIME_VARIANT_PACKAGES

IPPO_NETWORK_PACKAGES = {
    "noquant": "du_iibtd_based_fading_delta.IPPO_noquant.networks",
    "quant": "du_iibtd_based_fading_delta.IPPO_quant.networks",
}


def _optional_config_bool(section: Any, name: str) -> bool:
    """Read a variant-specific boolean config field without inventing state."""
    return bool(getattr(section, name, False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a based learned-policy checkpoint in the current environment.",
    )
    parser.add_argument("--variant", choices=tuple(VARIANT_PACKAGES), required=True)
    parser.add_argument(
        "--trainer",
        choices=LEARNED_TRAINERS,
        default="mappo",
        help="Checkpoint network type; both execute in the same based environment.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iibtd_device", default="cuda:0")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--scene_limit", type=int, default=0)
    parser.add_argument("--seed_base", type=int, default=100_042)
    parser.add_argument("--scene_seed_stride", type=int, default=10_000)
    parser.add_argument("--snr_outage_threshold_db", type=float, default=-5.0)
    parser.add_argument("--prefill_percent", type=float, default=5.0)
    parser.add_argument(
        "--ugv_control",
        choices=("policy", "guidance_greedy"),
        default="policy",
        help="Use the learned UGV actor or an evaluation-only greedy guidance follower.",
    )
    parser.add_argument(
        "--uav_max_energy",
        type=float,
        default=None,
        help="Optional UAV episode energy-budget override in joules.",
    )
    parser.add_argument(
        "--ugv_comm_target_mode",
        choices=("path_corridor",),
        default=None,
        help="Optional frozen communication-target rule override.",
    )
    parser.add_argument(
        "--ugv_comm_backlog_threshold",
        type=float,
        default=None,
        help="Optional normalized bit-backlog threshold override.",
    )
    parser.add_argument(
        "--ugv_comm_local_path_horizon",
        type=int,
        default=None,
        help="Optional normal communication-path horizon override.",
    )
    parser.add_argument(
        "--ugv_comm_expanded_path_horizon",
        type=int,
        default=None,
        help="Optional expanded communication-path horizon override.",
    )
    parser.add_argument(
        "--ugv_comm_corridor_width",
        type=int,
        default=None,
        help="Optional communication-path side-corridor width override.",
    )
    parser.add_argument(
        "--ugv_service_action_mask",
        type=_parse_bool_arg,
        default=None,
        help="Optional service-aware UGV action-mask override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.scene_limit < 0:
        raise ValueError("scene_limit must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    runtime_package = RUNTIME_VARIANT_PACKAGES[args.variant]
    config_module = importlib.import_module(f"{runtime_package}.config")
    networks_module = None
    if args.trainer == "mappo":
        networks_module = importlib.import_module(
            MAPPO_NETWORK_PACKAGES[args.variant]
        )
    elif args.trainer == "ippo":
        networks_module = importlib.import_module(IPPO_NETWORK_PACKAGES[args.variant])
    train_module = importlib.import_module(f"{runtime_package}.runner")
    utils_module = importlib.import_module(f"{runtime_package}.utils")

    with args.config.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Expected a JSON object in {args.config}")

    config = config_module.Config()
    _apply_saved_config(config, raw_config)
    config.training.device = str(args.device)
    config.planner.iibtd_device = str(args.iibtd_device)
    config.comm.snr_outage_threshold_db = float(args.snr_outage_threshold_db)
    config.planner.prefill_percent = float(args.prefill_percent)
    config.planner.ugv_control_mode = ugv_control_mode_for_trainer(args.trainer)
    if args.max_steps is not None:
        config.training.episode_max_steps = int(args.max_steps)
    if args.uav_max_energy is not None:
        config.uav.max_energy = float(args.uav_max_energy)
    if args.ugv_comm_target_mode is not None:
        config.planner.ugv_comm_target_mode = str(args.ugv_comm_target_mode)
    if args.ugv_comm_backlog_threshold is not None:
        config.planner.ugv_comm_backlog_threshold = float(
            args.ugv_comm_backlog_threshold
        )
    if args.ugv_comm_local_path_horizon is not None:
        config.planner.ugv_comm_local_path_horizon = int(
            args.ugv_comm_local_path_horizon
        )
    if args.ugv_comm_expanded_path_horizon is not None:
        config.planner.ugv_comm_expanded_path_horizon = int(
            args.ugv_comm_expanded_path_horizon
        )
    if args.ugv_comm_corridor_width is not None:
        config.planner.ugv_comm_corridor_width = int(args.ugv_comm_corridor_width)
    if args.ugv_service_action_mask is not None:
        config.planner.ugv_service_action_mask = bool(args.ugv_service_action_mask)
    if args.scene_limit > 0:
        config.scene.radioseer_scene_indices = list(
            config.scene.radioseer_scene_indices[: args.scene_limit]
        )
    config.__post_init__()

    max_steps = int(config.training.episode_max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    utils_module.set_seeds(int(config.training.seed))
    shared_data = train_module.make_shared_env_data_suite(config)
    env_factory = train_module.make_env_factory(config, shared_data=shared_data)
    envs = [env_factory(scene_pos) for scene_pos in range(len(shared_data))]
    scene_results = []
    try:
        if args.trainer == "mappo":
            policy_type = networks_module.MAPPOPolicy
        elif args.trainer == "happo":
            policy_type = HAPPOPolicy
        elif args.trainer == "mappo_cf":
            policy_type = MAPPOCFPolicy
        elif args.trainer == "ippo":
            policy_type = networks_module.IPPOPolicy
        elif is_uav_only_ppo(args.trainer):
            policy_type = UAVPPOPolicy
        else:
            raise AssertionError(f"Unhandled trainer {args.trainer!r}")
        policy = policy_type(
            envs[0].get_obs_dims(),
            envs[0].get_action_dims(),
            config.training,
        )
        policy.load(str(args.checkpoint))

        for scene_pos, env in enumerate(envs):
            scene_seed_base = int(args.seed_base) + scene_pos * int(
                args.scene_seed_stride
            )
            print(
                f"[Eval] variant={args.variant} scene={scene_pos + 1}/{len(envs)} "
                f"sample={env.config.scene.radioseer_sample_index} seed={scene_seed_base}",
                flush=True,
            )
            if args.ugv_control != "policy" and is_uav_only_ppo(args.trainer):
                raise ValueError(
                    "--ugv_control guidance_greedy cannot override the fixed/heuristic/A* "
                    "UGV controller selected by --trainer"
                )
            eval_policy = (
                policy
                if args.ugv_control == "policy"
                else GuidanceGreedyUGVOverridePolicy(policy, env)
            )
            result = utils_module.evaluate_policy(
                env=env,
                policy=eval_policy,
                num_episodes=int(args.num_episodes),
                max_steps=max_steps,
                seed_base=scene_seed_base,
            )
            scene_results.append(result)
            print(
                "[Eval] result "
                f"nmse={float(result['eval_mean_nmse']):.6f} "
                f"outage={float(result['eval_mean_outage_ratio']):.3f} "
                f"used_Gbit={float(result['eval_mean_data_delivered_bits']) * 1e-9:.3f} "
                f"completed_Gbit={float(result['eval_mean_completed_packet_bits']) * 1e-9:.3f} "
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
    source_max_packet_bits = float(
        config.uav.sensing_units_for_ratio(max(config.uav.bandwidth_ratios))
    ) * float(config.comm.data_per_sample)
    payload = {
        "evaluation_config": {
            "variant": str(args.variant),
            "trainer": str(args.trainer),
            "checkpoint": str(args.checkpoint),
            "source_config": str(args.config),
            "device": str(args.device),
            "iibtd_device": str(args.iibtd_device),
            "num_episodes_per_scene": int(args.num_episodes),
            "max_steps": max_steps,
            "uav_max_energy": float(config.uav.max_energy),
            "snr_outage_threshold_db": float(config.comm.snr_outage_threshold_db),
            "prefill_percent": float(config.planner.prefill_percent),
            "prefill_budget_basis": int(config.planner.prefill_budget_basis),
            "obs_remaining_time": _optional_config_bool(
                config.obs,
                "include_remaining_time",
            ),
            "obs_quant_context": _optional_config_bool(
                config.obs,
                "include_quant_context",
            ),
            "alpha_nmse": float(config.reward.alpha_nmse),
            "physical_link_reward_penalty": False,
            "gamma_queue": float(config.reward.gamma_queue),
            "lambda_uav_progress": float(config.reward.lambda_uav_progress),
            "lambda_uav_backtrack": float(config.reward.lambda_uav_backtrack),
            "lambda_ugv_progress": float(config.reward.lambda_ugv_progress),
            "lambda_ugv_backtrack": float(config.reward.lambda_ugv_backtrack),
            "lambda_novel_info": float(config.reward.lambda_novel_info),
            "lambda_full_repeat": float(config.reward.lambda_full_repeat),
            "source_measurement_bits": int(config.comm.source_measurement_bits),
            "data_per_sample_bits": float(config.comm.data_per_sample),
            "queue_capacity_bits": float(config.uav.queue_capacity_bits),
            "source_max_packet_bits": source_max_packet_bits,
            "ugv_control": str(args.ugv_control),
            "ugv_comm_target_mode": str(config.planner.ugv_comm_target_mode),
            "ugv_comm_backlog_threshold": float(
                config.planner.ugv_comm_backlog_threshold
            ),
            "ugv_comm_local_path_horizon": int(
                config.planner.ugv_comm_local_path_horizon
            ),
            "ugv_comm_expanded_path_horizon": int(
                config.planner.ugv_comm_expanded_path_horizon
            ),
            "ugv_comm_corridor_width": int(
                config.planner.ugv_comm_corridor_width
            ),
            "ugv_service_action_mask": bool(
                config.planner.ugv_service_action_mask
            ),
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
        "artifact_scene": {
            "uav_trajectory": aggregate.get("eval_uav_trajectory", []),
            "ugv_trajectory": aggregate.get("eval_ugv_trajectory", []),
            "step_details": aggregate.get("eval_step_details", {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[Eval] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

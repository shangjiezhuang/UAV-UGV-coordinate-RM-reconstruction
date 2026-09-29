"""Evaluate one fair method across UAV energy budgets on paired scenes/seeds."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.energy_tradeoff import summarize_energy_episode_records
from du_iibtd_based_fading_delta.evaluation_common import (
    apply_saved_config as _apply_saved_config,
    compact_eval_result,
    json_default as _json_default,
)
from du_iibtd_based_fading_delta.shared.evaluate_checkpoint import (
    IPPO_NETWORK_PACKAGES,
    MAPPO_NETWORK_PACKAGES,
    RUNTIME_VARIANT_PACKAGES,
)
from du_iibtd_based_fading_delta.fair_baseline_policies import (
    make_baseline_policy,
    ugv_control_mode_for_baseline,
)
from du_iibtd_based_fading_delta.fair_training import (
    LEARNED_TRAINERS,
    is_uav_only_ppo,
    ugv_control_mode_for_trainer,
)
from du_iibtd_based_fading_delta.scene_suite import aggregate_eval_results
from du_iibtd_based_fading_delta.uav_ppo import UAVPPOPolicy
from du_iibtd_based_fading_delta.ugv_control import uses_service_action_mask


NONLEARNING_METHODS = (
    "total_random",
    "greedy_guidance",
    "greedy_legacy",
    "greedy_astar_2path_support",
)
METHODS = (*LEARNED_TRAINERS, *NONLEARNING_METHODS)


def _parse_energy_values(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("energies must be positive comma-separated joules")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("energies must not contain duplicates")
    return values


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=tuple(RUNTIME_VARIANT_PACKAGES),
        required=True,
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--energies", type=_parse_energy_values, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iibtd-device", default="cuda:0")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=300_042)
    parser.add_argument("--scene-seed-stride", type=int, default=10_000)
    parser.add_argument("--local-planner-radius", type=int, default=15)
    return parser.parse_args(argv)


def _make_learned_policy(
    method: str,
    variant: str,
    obs_dims: dict,
    action_dims: dict,
    config: Any,
    checkpoint: Path,
):
    if method == "mappo":
        policy_type = importlib.import_module(
            MAPPO_NETWORK_PACKAGES[variant]
        ).MAPPOPolicy
    elif method == "ippo":
        policy_type = importlib.import_module(
            IPPO_NETWORK_PACKAGES[variant]
        ).IPPOPolicy
    elif is_uav_only_ppo(method):
        policy_type = UAVPPOPolicy
    else:
        raise AssertionError(f"Unhandled learned method {method!r}")
    policy = policy_type(obs_dims, action_dims, config.training)
    policy.load(str(checkpoint))
    return policy


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("num-episodes and max-steps must be positive")
    if args.scene_limit < 0:
        raise ValueError("scene-limit must be non-negative")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    learned = args.method in LEARNED_TRAINERS
    if learned and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise FileNotFoundError(
            "a valid --checkpoint is required for learned methods"
        )
    if not learned and args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid for learned methods")

    runtime_package = RUNTIME_VARIANT_PACKAGES[args.variant]
    config_module = importlib.import_module(f"{runtime_package}.config")
    train_module = importlib.import_module(f"{runtime_package}.runner")
    utils_module = importlib.import_module(f"{runtime_package}.utils")
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Expected a JSON object in {args.config}")
    base_config = config_module.Config()
    _apply_saved_config(base_config, raw_config)
    base_config.training.device = str(args.device)
    base_config.planner.iibtd_device = str(args.iibtd_device)
    base_config.planner.local_planner_radius = int(args.local_planner_radius)
    base_config.training.episode_max_steps = int(args.max_steps)
    if args.scene_limit > 0:
        base_config.scene.radioseer_scene_indices = list(
            base_config.scene.radioseer_scene_indices[: args.scene_limit]
        )
    base_config.__post_init__()
    utils_module.set_seeds(int(base_config.training.seed))
    shared_data = train_module.make_shared_env_data_suite(base_config)

    raw_episode_records = []
    energy_results = []
    for energy_j in args.energies:
        config = deepcopy(base_config)
        config.uav.max_energy = float(energy_j)
        if learned:
            config.planner.ugv_control_mode = ugv_control_mode_for_trainer(args.method)
        else:
            config.planner.ugv_control_mode = ugv_control_mode_for_baseline(
                args.method
            )
        if args.method == "total_random":
            config.planner.ugv_service_action_mask = False
        config.__post_init__()

        env_factory = train_module.make_env_factory(config, shared_data=shared_data)
        envs = [env_factory(scene_pos) for scene_pos in range(len(shared_data))]
        scene_results = []
        try:
            learned_policy = None
            if learned:
                learned_policy = _make_learned_policy(
                    method=args.method,
                    variant=args.variant,
                    obs_dims=envs[0].get_obs_dims(),
                    action_dims=envs[0].get_action_dims(),
                    config=config,
                    checkpoint=args.checkpoint,
                )
            for scene_pos, env in enumerate(envs):
                reset_seed = int(args.seed_base) + scene_pos * int(
                    args.scene_seed_stride
                )
                if learned:
                    policy = learned_policy
                else:
                    policy = make_baseline_policy(args.method, seed=reset_seed)
                result = utils_module.evaluate_policy(
                    env=env,
                    policy=policy,
                    num_episodes=int(args.num_episodes),
                    max_steps=int(args.max_steps),
                    seed_base=reset_seed,
                )
                scene_results.append(result)
                print(
                    f"[Energy Eval] method={args.method} energy={energy_j:g}J "
                    f"scene={scene_pos + 1}/{len(envs)} "
                    f"nmse={float(result['eval_mean_nmse']):.6f} "
                    f"link_Gbit={float(result['eval_mean_link_transmitted_bits']) * 1e-9:.3f}",
                    flush=True,
                )
        finally:
            for env in envs:
                env.close()

        aggregate = aggregate_eval_results(
            scene_results,
            seed_base=int(args.seed_base),
        )
        records = [dict(record) for record in aggregate["eval_episode_records"]]
        for record in records:
            record["method"] = str(args.method)
            record["variant"] = str(args.variant)
        raw_episode_records.extend(records)
        energy_results.append(
            {
                "energy_budget_j": float(energy_j),
                "aggregate": compact_eval_result(aggregate),
                "per_scene": [compact_eval_result(result) for result in scene_results],
            }
        )

    payload = {
        "evaluation_protocol": {
            "variant": str(args.variant),
            "method": str(args.method),
            "source_config": str(args.config),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "energies_j": [float(value) for value in args.energies],
            "num_episodes_per_scene": int(args.num_episodes),
            "max_steps": int(args.max_steps),
            "seed_base": int(args.seed_base),
            "scene_seed_stride": int(args.scene_seed_stride),
            "deterministic_learned_policy": bool(learned),
            "physical_transmission_metric": "link_transmitted_bits",
            "local_planner_radius": int(args.local_planner_radius),
            "ugv_control_mode": str(config.planner.ugv_control_mode),
            "ugv_service_action_mask_effective": bool(
                uses_service_action_mask(config)
            ),
            "source_measurement_bits": int(config.comm.source_measurement_bits),
            "outage_snr_mode": str(config.comm.outage_snr_mode),
            "snr_outage_threshold_db": float(config.comm.snr_outage_threshold_db),
            "data_per_sample_bits": float(config.comm.data_per_sample),
            "queue_capacity_bits": float(config.uav.queue_capacity_bits),
        },
        "energy_nmse_link_pairs": summarize_energy_episode_records(
            raw_episode_records
        ),
        "episode_records": raw_episode_records,
        "by_energy": energy_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"[Energy Eval] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Build a strict NMSE/link-bit frontier from frozen-protocol replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.scene_suite import (
    dominates_nmse_link,
    is_strict_pareto_feasible,
)


def _load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _metric(aggregate: Mapping[str, Any], key: str) -> float:
    try:
        return float(aggregate[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _config_float(config: Mapping[str, Any], key: str) -> float:
    try:
        return float(config[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def summarize(
    runs: Sequence[tuple[str, Path]],
    *,
    expected_prefill_band_units: float,
    expected_scene_count: int,
    expected_episodes_per_scene: int,
    expected_energy_j: float = 9000.0,
    expected_horizon: int = 200,
    expected_prefill_percent: float = 5.0,
    expected_outage_threshold_db: float = -5.0,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    expected_total_episodes = expected_scene_count * expected_episodes_per_scene
    for run_name, run_dir in runs:
        replay_paths = sorted(run_dir.glob("replay_*_seed*_n*.json"))
        if not replay_paths:
            raise FileNotFoundError(f"No replay JSON files in {run_dir}")
        for replay_path in replay_paths:
            payload = _load(replay_path)
            aggregate = payload.get("aggregate", {})
            evaluation_config = payload.get("evaluation_config", {})
            if not isinstance(aggregate, Mapping):
                raise ValueError(f"Missing aggregate object in {replay_path}")
            if not isinstance(evaluation_config, Mapping):
                evaluation_config = {}

            nmse = _metric(aggregate, "eval_mean_nmse")
            link_bits = _metric(aggregate, "eval_mean_link_transmitted_bits")
            completion = _metric(
                aggregate,
                "eval_mean_service_completion_ratio",
            )
            min_scene_completion = _metric(
                aggregate,
                "eval_min_scene_service_completion_ratio",
            )
            energy_failure = _metric(aggregate, "eval_energy_failure_rate")
            full_horizon = _metric(aggregate, "eval_full_horizon_rate")
            accounting = _metric(
                aggregate,
                "eval_max_data_accounting_violation_bits",
            )
            prefill = _metric(
                aggregate,
                "eval_mean_prefill_observed_band_units",
            )
            scene_count = _metric(aggregate, "eval_scene_count")
            total_episodes = _metric(aggregate, "eval_num_total_episodes")
            configured_energy = _config_float(evaluation_config, "uav_max_energy")
            configured_horizon = _config_float(evaluation_config, "max_steps")
            configured_prefill_percent = _config_float(
                evaluation_config,
                "prefill_percent",
            )
            configured_prefill_budget_basis = _config_float(
                evaluation_config,
                "prefill_budget_basis",
            )
            configured_outage_threshold = _config_float(
                evaluation_config,
                "snr_outage_threshold_db",
            )
            configured_episodes = _config_float(
                evaluation_config,
                "num_episodes_per_scene",
            )
            device = str(evaluation_config.get("device", ""))
            iibtd_device = str(evaluation_config.get("iibtd_device", ""))
            comm_target_mode = str(
                evaluation_config.get("ugv_comm_target_mode", "")
            )
            comm_backlog_threshold = _config_float(
                evaluation_config,
                "ugv_comm_backlog_threshold",
            )
            comm_local_horizon = _config_float(
                evaluation_config,
                "ugv_comm_local_path_horizon",
            )
            comm_expanded_horizon = _config_float(
                evaluation_config,
                "ugv_comm_expanded_path_horizon",
            )
            comm_corridor_width = _config_float(
                evaluation_config,
                "ugv_comm_corridor_width",
            )
            service_action_mask = _config_float(
                evaluation_config,
                "ugv_service_action_mask",
            )
            physical_link_reward_penalty = bool(
                evaluation_config.get("physical_link_reward_penalty", False)
            )
            gamma_queue = _config_float(evaluation_config, "gamma_queue")
            protocol_match = bool(
                np.isfinite(configured_energy)
                and abs(configured_energy - expected_energy_j) <= 1e-9
                and np.isfinite(configured_horizon)
                and int(round(configured_horizon)) == expected_horizon
                and np.isfinite(configured_prefill_percent)
                and abs(configured_prefill_percent - expected_prefill_percent) <= 1e-12
                and np.isfinite(configured_prefill_budget_basis)
                and int(round(configured_prefill_budget_basis)) == expected_horizon
                and np.isfinite(configured_outage_threshold)
                and abs(configured_outage_threshold - expected_outage_threshold_db)
                <= 1e-12
                and np.isfinite(configured_episodes)
                and int(round(configured_episodes)) == expected_episodes_per_scene
                and device.startswith("cuda")
                and iibtd_device.startswith("cuda")
                and comm_target_mode == "path_corridor"
                and np.isfinite(comm_backlog_threshold)
                and abs(comm_backlog_threshold - 0.5) <= 1e-12
                and np.isfinite(comm_local_horizon)
                and int(round(comm_local_horizon)) == 10
                and np.isfinite(comm_expanded_horizon)
                and int(round(comm_expanded_horizon)) == 20
                and np.isfinite(comm_corridor_width)
                and int(round(comm_corridor_width)) == 1
                and np.isfinite(service_action_mask)
                and int(round(service_action_mask)) == 1
            )
            protocol_fields_finite = all(
                np.isfinite(value)
                for value in (
                    prefill,
                    scene_count,
                    total_episodes,
                )
            )
            feasible = bool(
                is_strict_pareto_feasible(
                    nmse=nmse,
                    link_bits=link_bits,
                    energy_failure_rate=energy_failure,
                    full_horizon_rate=full_horizon,
                    max_data_accounting_violation_bits=accounting,
                )
                and protocol_fields_finite
                and abs(prefill - expected_prefill_band_units) <= 1e-9
                and int(round(scene_count)) == expected_scene_count
                and int(round(total_episodes)) == expected_total_episodes
                and protocol_match
            )
            points.append(
                {
                    "run": run_name,
                    "run_dir": str(run_dir),
                    "replay": str(replay_path),
                    "checkpoint": str(evaluation_config.get("checkpoint", "")),
                    "physical_link_reward_penalty": physical_link_reward_penalty,
                    "gamma_queue": gamma_queue,
                    "obs_remaining_time": evaluation_config.get(
                        "obs_remaining_time"
                    ),
                    "obs_quant_context": evaluation_config.get(
                        "obs_quant_context"
                    ),
                    "nmse": nmse,
                    "link_bits": link_bits,
                    "service_completion_ratio": completion,
                    "min_scene_service_completion_ratio": min_scene_completion,
                    "outage_ratio": _metric(
                        aggregate,
                        "eval_mean_outage_ratio",
                    ),
                    "energy_failure_rate": energy_failure,
                    "full_horizon_rate": full_horizon,
                    "max_data_accounting_violation_bits": accounting,
                    "prefill_observed_band_units": prefill,
                    "scene_count": scene_count,
                    "total_episodes": total_episodes,
                    "protocol_match": protocol_match,
                    "feasible": feasible,
                }
            )

    feasible_points = [point for point in points if bool(point["feasible"])]
    frontier = [
        point
        for point in feasible_points
        if not any(
            dominates_nmse_link(
                float(other["nmse"]),
                float(other["link_bits"]),
                float(point["nmse"]),
                float(point["link_bits"]),
            )
            for other in feasible_points
            if other is not point
        )
    ]
    frontier.sort(key=lambda point: (float(point["nmse"]), float(point["link_bits"])))
    return {
        "objectives": ["min_eval_mean_nmse", "min_eval_mean_link_transmitted_bits"],
        "constraints": {
            "eval_energy_failure_rate": 0.0,
            "eval_full_horizon_rate": 1.0,
            "max_data_accounting_violation_bits": 1e-6,
            "expected_prefill_observed_band_units": expected_prefill_band_units,
            "expected_scene_count": expected_scene_count,
            "expected_episodes_per_scene": expected_episodes_per_scene,
            "expected_uav_energy_j": expected_energy_j,
            "expected_horizon": expected_horizon,
            "expected_prefill_percent": expected_prefill_percent,
            "required_prefill_budget_basis": expected_horizon,
            "expected_snr_outage_threshold_db": expected_outage_threshold_db,
            "required_ugv_comm_target_mode": "path_corridor",
            "required_ugv_comm_backlog_threshold": 0.5,
            "required_ugv_comm_local_path_horizon": 10,
            "required_ugv_comm_expanded_path_horizon": 20,
            "required_ugv_comm_corridor_width": 1,
            "required_ugv_service_action_mask": True,
            "required_eval_backend": "cuda",
        },
        "points": points,
        "frontier": frontier,
    }


def _markdown(
    summary: Mapping[str, Any],
    title: str = "Long observation-ablation replay summary",
) -> str:
    frontier_replays = {str(point["replay"]) for point in summary["frontier"]}
    lines = [
        f"# {title}",
        "",
        "| run | checkpoint | direct link reward | gamma | NMSE | link Gbit | completion | min scene | outage | feasible | frontier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in summary["points"]:
        checkpoint_name = Path(str(point["checkpoint"])).name
        lines.append(
            f"| {point['run']} | {checkpoint_name} | "
            f"{'yes' if point['physical_link_reward_penalty'] else 'no'} | "
            f"{float(point['gamma_queue']):.4g} | "
            f"{float(point['nmse']):.6f} | "
            f"{float(point['link_bits']) * 1e-9:.3f} | "
            f"{float(point['service_completion_ratio']):.1%} | "
            f"{float(point['min_scene_service_completion_ratio']):.1%} | "
            f"{float(point['outage_ratio']):.1%} | "
            f"{'yes' if point['feasible'] else 'no'} | "
            f"{'yes' if str(point['replay']) in frontier_replays else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Only points with measured finite constraint fields, the frozen prefill count, the complete scene/episode suite, and all hard gates enter the frontier.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-prefill-band-units", type=float, default=80.0)
    parser.add_argument("--expected-scene-count", type=int, default=8)
    parser.add_argument("--expected-episodes-per-scene", type=int, default=3)
    parser.add_argument("--expected-energy-j", type=float, default=9000.0)
    parser.add_argument("--expected-horizon", type=int, default=200)
    parser.add_argument("--expected-prefill-percent", type=float, default=5.0)
    parser.add_argument("--expected-outage-threshold-db", type=float, default=-5.0)
    parser.add_argument(
        "--summary-stem",
        default="observation_replay_summary",
        help="Output filename stem for the JSON and Markdown summaries.",
    )
    parser.add_argument(
        "--title",
        default="Long observation-ablation replay summary",
        help="Markdown report title.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = summarize(
        args.run,
        expected_prefill_band_units=float(args.expected_prefill_band_units),
        expected_scene_count=int(args.expected_scene_count),
        expected_episodes_per_scene=int(args.expected_episodes_per_scene),
        expected_energy_j=float(args.expected_energy_j),
        expected_horizon=int(args.expected_horizon),
        expected_prefill_percent=float(args.expected_prefill_percent),
        expected_outage_threshold_db=float(args.expected_outage_threshold_db),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_stem = str(args.summary_stem).strip()
    if not summary_stem or Path(summary_stem).name != summary_stem:
        raise ValueError("--summary-stem must be a non-empty filename stem")
    json_path = args.output_dir / f"{summary_stem}.json"
    markdown_path = args.output_dir / f"{summary_stem}.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    markdown = _markdown(summary, title=str(args.title))
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

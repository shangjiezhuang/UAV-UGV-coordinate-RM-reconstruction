"""Summarize one MAPPO run under the current NMSE/link-bit protocol.

This tool deliberately separates engineering smoke tests, 32k mechanism
screens, and 176k candidate runs. It uses the logged mean per-episode service
completion when available and falls back to completed/produced bits for older
logs, so packets left behind cannot manufacture a low-bit Pareto point.
"""

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


def _latest_metrics(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "logs").glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.json under {run_dir / 'logs'}")
    return candidates[-1]


def _series(training: Mapping[str, Any], key: str, count: int, default: float) -> list[float]:
    values = training.get(key)
    if values is None:
        return [float(default)] * count
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count:
        raise ValueError(f"{key} has {array.size} values, expected {count}")
    return [float(value) for value in array]


def _evidence_level(total_timesteps: int) -> str:
    if total_timesteps < 32_000:
        return "engineering_smoke_only"
    if total_timesteps < 176_000:
        return "mechanism_screen_only"
    return "candidate_training_requires_multi_seed_confirmation"


def summarize_run(metrics_path: Path) -> dict[str, Any]:
    payload = _load(metrics_path)
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError(f"Missing training metrics in {metrics_path}")

    updates = [int(value) for value in np.asarray(training.get("eval_update", []), dtype=int)]
    count = len(updates)
    if count == 0:
        raise ValueError(f"No eval_update entries in {metrics_path}")

    nmse = _series(training, "eval_mean_nmse", count, np.nan)
    link_bits = _series(training, "eval_mean_link_transmitted_bits", count, np.nan)
    produced_bits = _series(training, "eval_mean_data_produced_bits", count, np.nan)
    completed_bits = _series(training, "eval_mean_completed_packet_bits", count, np.nan)
    logged_completion = _series(
        training,
        "eval_mean_service_completion_ratio",
        count,
        np.nan,
    )
    min_scene_completion = _series(
        training,
        "eval_min_scene_service_completion_ratio",
        count,
        np.nan,
    )
    energy_failure = _series(training, "eval_energy_failure_rate", count, np.nan)
    full_horizon = _series(training, "eval_full_horizon_rate", count, np.nan)
    accounting = _series(
        training,
        "eval_max_data_accounting_violation_bits",
        count,
        np.nan,
    )
    outage = _series(training, "eval_mean_outage_ratio", count, np.nan)

    points: list[dict[str, Any]] = []
    for idx, update in enumerate(updates):
        produced = produced_bits[idx]
        if np.isfinite(logged_completion[idx]):
            completion = logged_completion[idx]
            completion_source = "logged_mean_episode_ratio"
        else:
            completion = (
                completed_bits[idx] / produced
                if np.isfinite(produced) and produced > 1e-9
                else 1.0
            )
            completion_source = "recomputed_completed_over_produced"
        feasible = is_strict_pareto_feasible(
            nmse=nmse[idx],
            link_bits=link_bits[idx],
            energy_failure_rate=energy_failure[idx],
            full_horizon_rate=full_horizon[idx],
            max_data_accounting_violation_bits=accounting[idx],
        )
        points.append(
            {
                "update": update,
                "nmse": nmse[idx],
                "link_bits": link_bits[idx],
                "produced_bits": produced,
                "completed_bits": completed_bits[idx],
                "service_completion_ratio": float(completion),
                "min_scene_service_completion_ratio": float(
                    min_scene_completion[idx]
                ),
                "service_completion_source": completion_source,
                "outage_ratio": outage[idx],
                "energy_failure_rate": energy_failure[idx],
                "full_horizon_rate": full_horizon[idx],
                "max_data_accounting_violation_bits": accounting[idx],
                "feasible": feasible,
            }
        )

    feasible_points = [point for point in points if point["feasible"]]
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

    config = payload.get("config", {})
    mappo = config.get("mappo", {}) if isinstance(config, Mapping) else {}
    planned_timesteps = (
        int(mappo.get("total_timesteps", 0))
        if isinstance(mappo, Mapping)
        else 0
    )
    logged_global_steps = np.asarray(training.get("global_step", []), dtype=float).reshape(-1)
    finite_global_steps = logged_global_steps[np.isfinite(logged_global_steps)]
    completed_timesteps = (
        int(np.max(finite_global_steps))
        if finite_global_steps.size > 0
        else planned_timesteps
    )
    return {
        "metrics_path": str(metrics_path),
        "planned_timesteps": planned_timesteps,
        "completed_timesteps": completed_timesteps,
        # Backward-compatible field now means actual completed transitions.
        "total_timesteps": completed_timesteps,
        "evidence_level": _evidence_level(completed_timesteps),
        "objectives": ["min_final_nmse", "min_link_transmitted_bits"],
        "constraints": {
            "energy_failure_rate": 0.0,
            "full_horizon_rate": 1.0,
            "max_data_accounting_violation_bits": 1e-6,
        },
        "points": points,
        "frontier": frontier,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pareto run summary",
        "",
        f"Evidence level: `{summary['evidence_level']}`; "
        f"completed transitions: {int(summary['completed_timesteps']):,}; "
        f"planned: {int(summary['planned_timesteps']):,}.",
        "",
        "| update | NMSE | link Gbit | produced Gbit | completed Gbit | completion | min scene | outage | feasible | frontier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    frontier_updates = {int(point["update"]) for point in summary["frontier"]}
    for point in summary["points"]:
        lines.append(
            f"| {int(point['update'])} | {float(point['nmse']):.6f} | "
            f"{float(point['link_bits']) * 1e-9:.3f} | "
            f"{float(point['produced_bits']) * 1e-9:.3f} | "
            f"{float(point['completed_bits']) * 1e-9:.3f} | "
            f"{float(point['service_completion_ratio']):.1%} | "
            f"{float(point['min_scene_service_completion_ratio']):.1%} | "
            f"{float(point['outage_ratio']):.1%} | "
            f"{'yes' if point['feasible'] else 'no'} | "
            f"{'yes' if int(point['update']) in frontier_updates else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Only feasible, nondominated points are in the frontier. A mechanism screen is not a convergence or final-ranking claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics_path = args.metrics if args.metrics is not None else _latest_metrics(args.run_dir)
    summary = summarize_run(metrics_path)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.run_dir / "pareto_screen_summary.json"
    markdown_path = args.run_dir / "pareto_screen_summary.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    markdown = _markdown(summary)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

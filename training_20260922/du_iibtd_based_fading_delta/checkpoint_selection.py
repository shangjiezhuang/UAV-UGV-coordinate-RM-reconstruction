"""Shared evaluation-based checkpoint selection for all learned trainers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from du_iibtd_based_fading_delta.evaluation_common import json_default
from du_iibtd_based_fading_delta.scene_suite import (
    dominates_nmse_link,
    is_strict_pareto_feasible,
)


def save_best_eval_checkpoint(
    policy: Any,
    model_dir: str,
    checkpoint_name: str,
    metric_name: str,
    metric_value: float,
    best_value: float,
    update: int,
    higher_is_better: bool,
) -> Tuple[float, Optional[str]]:
    """Persist a best-so-far checkpoint when an evaluation metric improves."""
    if not np.isfinite(metric_value):
        return best_value, None
    improved = metric_value > best_value if higher_is_better else metric_value < best_value
    if not improved:
        return best_value, None

    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    policy.save(checkpoint_path)
    print(
        f"Saved {checkpoint_name}: {checkpoint_path} "
        f"(update {update}, {metric_name}={metric_value:.6f})"
    )
    return metric_value, checkpoint_path


def update_pareto_eval_checkpoints(
    policy: Any,
    model_dir: str,
    frontier: List[Dict[str, object]],
    update: int,
    eval_results: Dict[str, float],
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Archive feasible, non-dominated NMSE/physical-link checkpoints."""
    nmse = float(eval_results.get("eval_mean_nmse", np.nan))
    link_bits = float(eval_results.get("eval_mean_link_transmitted_bits", np.nan))
    failure_rate = float(eval_results.get("eval_energy_failure_rate", np.nan))
    full_horizon_rate = float(eval_results.get("eval_full_horizon_rate", np.nan))
    service_completion_ratio = float(
        eval_results.get("eval_mean_service_completion_ratio", np.nan)
    )
    min_scene_service_completion_ratio = float(
        eval_results.get("eval_min_scene_service_completion_ratio", np.nan)
    )
    accounting_violation_bits = float(
        eval_results.get("eval_max_data_accounting_violation_bits", np.nan)
    )
    if not is_strict_pareto_feasible(
        nmse=nmse,
        link_bits=link_bits,
        energy_failure_rate=failure_rate,
        full_horizon_rate=full_horizon_rate,
        max_data_accounting_violation_bits=accounting_violation_bits,
    ):
        return frontier, None

    tolerance = 1e-12
    for point in frontier:
        point_nmse = float(point["eval_mean_nmse"])
        point_bits = float(point["eval_mean_link_transmitted_bits"])
        if dominates_nmse_link(point_nmse, point_bits, nmse, link_bits) or (
            abs(point_nmse - nmse) <= tolerance
            and abs(point_bits - link_bits) <= tolerance
        ):
            return frontier, None

    retained = [
        point
        for point in frontier
        if not dominates_nmse_link(
            nmse,
            link_bits,
            float(point["eval_mean_nmse"]),
            float(point["eval_mean_link_transmitted_bits"]),
        )
    ]
    checkpoint_name = f"pareto_update_{int(update):04d}.pt"
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    policy.save(checkpoint_path)
    retained.append(
        {
            "update": int(update),
            "checkpoint": checkpoint_path,
            "eval_mean_nmse": nmse,
            "eval_mean_link_transmitted_bits": link_bits,
            "eval_mean_service_completion_ratio": service_completion_ratio,
            "eval_min_scene_service_completion_ratio": min_scene_service_completion_ratio,
            "eval_mean_prefill_observed_band_units": float(
                eval_results.get("eval_mean_prefill_observed_band_units", np.nan)
            ),
            "eval_mean_prefill_equivalent_data_bits": float(
                eval_results.get("eval_mean_prefill_equivalent_data_bits", np.nan)
            ),
            "eval_max_data_accounting_violation_bits": accounting_violation_bits,
        }
    )
    retained.sort(key=lambda point: float(point["eval_mean_nmse"]))
    frontier_path = os.path.join(model_dir, "pareto_frontier.json")
    os.makedirs(os.path.dirname(frontier_path), exist_ok=True)
    with open(frontier_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "objectives": [
                    "min_eval_mean_nmse",
                    "min_eval_mean_link_transmitted_bits",
                ],
                "constraints": [
                    "eval_energy_failure_rate=0",
                    "eval_full_horizon_rate=1",
                    "eval_max_data_accounting_violation_bits<=1e-6",
                ],
                "points": retained,
            },
            handle,
            indent=2,
            default=json_default,
        )
    print(
        f"Saved Pareto checkpoint: {checkpoint_path} "
        f"(update {update}, nmse={nmse:.6f}, link_bits={link_bits:.3f})"
    )
    return retained, checkpoint_path


__all__ = ["save_best_eval_checkpoint", "update_pareto_eval_checkpoints"]

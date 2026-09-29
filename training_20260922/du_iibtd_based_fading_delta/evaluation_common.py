"""Variant-independent evaluation and reproducibility helpers."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import numpy as np


HEAVY_EVAL_KEYS = frozenset(
    {
        "eval_uav_trajectory",
        "eval_ugv_trajectory",
        "eval_step_details",
        "eval_episode_records",
        "eval_episode_artifacts",
    }
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def apply_saved_config(config: Any, raw_config: Mapping[str, Any]) -> None:
    """Apply known nested dataclass fields while ignoring obsolete saved keys."""
    if not isinstance(raw_config, Mapping):
        raise ValueError("saved config must be a JSON object")
    planner_values = raw_config.get("planner", {})
    if isinstance(planner_values, Mapping) and planner_values:
        obsolete = {"nmse_refresh_delta", "uncertainty_refresh_metric", "uncertainty_refresh_ratio",
                    "uncertainty_refresh_patience", "ensemble_full_refresh_interval", "hybrid_uncertainty_stall_steps"}
        if obsolete.intersection(planner_values) or (
            planner_values.get("hybrid_switch_metric") != "sum_relative_frobenius"
            or planner_values.get("reconstruction_refresh_mode") != "local_to_global"
        ):
            raise ValueError("Legacy reconstruction/switch protocol: use its frozen source or explicitly migrate the configuration.")
    for section_name, section_values in raw_config.items():
        target = getattr(config, str(section_name), None)
        if target is None or not isinstance(section_values, Mapping):
            continue
        for key, value in section_values.items():
            if hasattr(target, str(key)):
                setattr(target, str(key), value)

    scene_values = raw_config.get("scene", {})
    if isinstance(scene_values, Mapping) and "building_height_m" in scene_values and "building_height_mode" not in scene_values:
        config.scene.building_height_mode = "fixed"
        warnings.warn("Legacy building-height config: preserving fixed building heights; current main experiments use uniform_integer_per_building.", UserWarning)

    if isinstance(scene_values, Mapping) and scene_values and "building_split_min_area_m2" not in scene_values:
        config.scene.building_split_min_area_m2 = 0.0

    comm_values = raw_config.get("comm", {})
    if isinstance(comm_values, Mapping) and comm_values and "nlos_length_loss_db_per_m" not in comm_values:
        config.comm.nlos_length_loss_db_per_m = 0.0
        config.comm.nlos_length_loss_cap_db = 0.0
        warnings.warn("Legacy length-loss config: disabling the new obstruction-length correction to reproduce the saved run.", UserWarning)
    if isinstance(comm_values, Mapping) and comm_values and "los_excess_db" not in comm_values:
        config.comm.los_excess_db = 0.0
        config.comm.nlos_excess_db = float(comm_values.get("nlos_excess_db", 15.0))
        warnings.warn("Legacy excess-loss config: preserving zero LoS loss and the saved NLoS loss; current urban experiments use 1/20 dB.", UserWarning)
    if isinstance(comm_values, Mapping) and comm_values and "outage_snr_mode" not in comm_values:
        config.comm.outage_snr_mode = "nominal"
        warnings.warn("Legacy outage config: preserving the nominal SNR gate; current main experiments use received SNR including shadowing.", UserWarning)
    if isinstance(comm_values, Mapping) and "tx_power_dbm" in comm_values and "tx_power_choices_dbm" not in comm_values:
        config.comm.tx_power_choices_dbm = [float(comm_values["tx_power_dbm"])]
        config.comm.tx_energy_enabled = False
        warnings.warn("Legacy fixed-power config: preserving one power choice and zero TX energy; this is not the current main-experiment protocol.", UserWarning)


def compact_eval_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove trajectory/detail/raw-episode payloads from a scene summary."""
    return {key: value for key, value in result.items() if key not in HEAVY_EVAL_KEYS}


def add_cumulative_step_metrics(
    step_details: Mapping[str, Any],
) -> Dict[str, Any]:
    """Add directly plottable cumulative test metrics to one episode trace."""
    details = dict(step_details)

    def series(key: str) -> np.ndarray:
        values = np.asarray(details.get(key, []), dtype=float)
        if values.ndim != 1:
            raise ValueError(f"step detail {key!r} must be one-dimensional")
        return values

    produced = series("data_produced_bits")
    delivered = series("data_delivered_bits")
    completed = series("completed_packet_bits")
    transmitted = series("link_transmitted_bits")
    outage = series("channel_outage")
    reward_components = [
        series(key)
        for key in (
            "r_nmse",
            "r_queue",
            "r_progress",
            "r_novel_info",
            "r_full_repeat",
        )
    ]
    expected_length = len(produced)
    named_series = {
        "data_delivered_bits": delivered,
        "completed_packet_bits": completed,
        "link_transmitted_bits": transmitted,
        "channel_outage": outage,
        "reward components": reward_components[0],
    }
    for name, values in named_series.items():
        if len(values) != expected_length:
            raise ValueError(
                f"step detail {name!r} has length {len(values)}, "
                f"expected {expected_length}"
            )
    for values in reward_components[1:]:
        if len(values) != expected_length:
            raise ValueError("reward component step series lengths do not match")

    team_reward = np.sum(np.stack(reward_components, axis=0), axis=0)
    cumulative_produced = np.cumsum(produced)
    cumulative_completed = np.cumsum(completed)
    cumulative_outage = np.divide(
        np.cumsum(outage),
        np.arange(1, expected_length + 1, dtype=float),
    )
    cumulative_completion = np.divide(
        cumulative_completed,
        cumulative_produced,
        out=np.ones(expected_length, dtype=float),
        where=cumulative_produced > 1e-9,
    )
    details.update(
        {
            "team_reward": team_reward.tolist(),
            "cumulative_team_reward": np.cumsum(team_reward).tolist(),
            "cumulative_data_produced_bits": cumulative_produced.tolist(),
            "cumulative_data_delivered_bits": np.cumsum(delivered).tolist(),
            "cumulative_completed_packet_bits": cumulative_completed.tolist(),
            "cumulative_link_transmitted_bits": np.cumsum(
                transmitted
            ).tolist(),
            "cumulative_outage_ratio": cumulative_outage.tolist(),
            "cumulative_service_completion_ratio": (
                cumulative_completion.tolist()
            ),
        }
    )
    return details


def scene_artifacts_payload(
    scene_results: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Return all scene/episode step traces without duplicating summary data."""
    artifacts: list[Dict[str, Any]] = []
    for scene_pos, result in enumerate(scene_results):
        artifacts.append(
            {
                "scene_pos": int(scene_pos),
                "scene_source": str(result.get("eval_scene_source", "")),
                "scene_sample_index": int(
                    result.get("eval_scene_sample_index", -1)
                ),
                "scene_sample_tag": str(
                    result.get("eval_scene_sample_tag", "")
                ),
                "reset_seed_base": int(
                    result.get("eval_reset_seed_base", -1)
                ),
                "episodes": list(result.get("eval_episode_artifacts", [])),
            }
        )
    return artifacts


def zero_machine_roundoff(value: float, *scale_values: float) -> float:
    """Map only floating-point cancellation noise to an exact diagnostic zero."""
    result = float(value)
    scale = max(1.0, *(abs(float(item)) for item in scale_values))
    tolerance = 8.0 * np.finfo(np.float64).eps * scale
    return 0.0 if abs(result) <= tolerance else result


def eval_scene_metadata(env: Any, seed_base: Optional[int]) -> Dict[str, Any]:
    sim_data = getattr(env, "sim_data", None)
    raw_data = getattr(sim_data, "_data", {})
    scene_data = raw_data.get("config", {}) if isinstance(raw_data, dict) else {}

    def int_or_default(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    actual_scene_source = str(
        scene_data.get("scene_source", env.config.scene.scene_source)
    )
    actual_sample_index = int_or_default(
        scene_data.get("sample_index", env.config.scene.radioseer_sample_index)
    )
    actual_dataset_root = str(
        scene_data.get("dataset_root", env.config.scene.radioseer_root)
    )
    i_mask = getattr(env, "I_mask", getattr(sim_data, "I_mask", None))
    non_building_mask = getattr(sim_data, "non_building_mask", i_mask)
    i_mask_arr = (
        np.asarray(i_mask, dtype=bool)
        if i_mask is not None
        else np.asarray([], dtype=bool)
    )
    non_building_mask_arr = (
        np.asarray(non_building_mask, dtype=bool)
        if non_building_mask is not None
        else np.asarray([], dtype=bool)
    )
    mask_is_non_building = bool(
        i_mask_arr.size > 0
        and non_building_mask_arr.shape == i_mask_arr.shape
        and np.array_equal(i_mask_arr, non_building_mask_arr)
    )
    return {
        "eval_scene_source": actual_scene_source,
        "eval_scene_sample_index": actual_sample_index,
        "eval_scene_sample_tag": str(scene_data.get("sample_tag", "")),
        "eval_scene_dataset_root": actual_dataset_root,
        "eval_reset_seed_base": int_or_default(seed_base),
        "eval_nmse_mask": (
            "non_building/I_mask" if mask_is_non_building else "custom_or_full_map"
        ),
        "eval_nmse_mask_is_non_building": int(mask_is_non_building),
        "eval_nmse_mask_cells": int(np.sum(i_mask_arr)),
    }


def resolve_eval_max_steps(env: Any, max_steps: Optional[int]) -> int:
    if max_steps is not None:
        return int(max_steps)
    env_config = getattr(env, "config", None)
    for section_name in ("mappo", "ippo", "run"):
        section = getattr(env_config, section_name, None)
        if section is not None and hasattr(section, "episode_max_steps"):
            return int(section.episode_max_steps)
    raise AttributeError("Unable to infer episode_max_steps from env.config")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "apply_saved_config",
    "add_cumulative_step_metrics",
    "compact_eval_result",
    "eval_scene_metadata",
    "json_default",
    "resolve_eval_max_steps",
    "scene_artifacts_payload",
    "set_seeds",
    "zero_machine_roundoff",
]

"""Shared RadioSeerDPM100PSD scene suite for du_iibtd_based_fading_delta experiments."""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DPM100PSD_ROOT = "RadioSeerDPM100PSD"
DEFAULT_FARM_ROOT = "FARMOmniDPM100PSD_251"

# DPM100PSD manifest rows aligned with the UAVTest-style scenes plus scene218.
DEFAULT_DPM100PSD_SCENE_INDICES = (8513, 1807, 1579, 1371, 10001)
DEFAULT_RADIOSEER_SCENE_SPECS = (
    8513,
    1807,
    1579,
    1371,
    10001,
    f"{DEFAULT_FARM_ROOT}:905",  # FARM scene4 tx80 PSD001
    f"{DEFAULT_FARM_ROOT}:43",   # FARM scene7 tx45 PSD003
    f"{DEFAULT_FARM_ROOT}:705",  # FARM scene9 tx13 PSD001
)

# Keep train/eval on the same scene suite while separating deterministic resets.
EVAL_SCENE_SEED_STRIDE = 10_000


def _is_int_text(value: str) -> bool:
    text = str(value).strip()
    if text.startswith(("+", "-")):
        text = text[1:]
    return bool(text) and text.isdigit()


def _split_scene_spec_string(value: str) -> List[str]:
    text = str(value).strip()
    if not text:
        return []
    for sep in (";", " "):
        text = text.replace(sep, ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _scene_spec_from_value(value: Any, default_root: str) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        root = value.get("root", value.get("dataset_root", default_root))
        sample_index = value.get(
            "index",
            value.get("sample_index", value.get("radioseer_sample_index")),
        )
        if sample_index is None:
            raise ValueError(f"scene spec mapping must include index/sample_index: {value!r}")
    elif isinstance(value, Number):
        root = default_root
        sample_index = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("scene spec text must not be empty")
        if ":" in text and not _is_int_text(text):
            root_text, sample_text = text.rsplit(":", 1)
            root = root_text.strip() or default_root
            sample_index = sample_text.strip()
        else:
            root = default_root
            sample_index = text

    return {"root": str(root or default_root), "index": int(sample_index)}


def parse_scene_specs(
    value: Any,
    default: Sequence[Any] = DEFAULT_RADIOSEER_SCENE_SPECS,
    root: str = DEFAULT_DPM100PSD_ROOT,
) -> List[Dict[str, Any]]:
    """Normalize CLI/config scene specs into non-empty root/index records."""
    default_root = str(root or DEFAULT_DPM100PSD_ROOT)
    if value is None:
        items = list(default)
    elif isinstance(value, str):
        items = _split_scene_spec_string(value) or list(default)
    elif isinstance(value, (Number, Mapping)):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise TypeError(f"Unsupported scene index value: {value!r}")

    specs = [_scene_spec_from_value(item, default_root) for item in items]
    if not specs:
        raise ValueError("scene index suite must not be empty")
    return specs


def format_scene_spec(spec: Any, default_root: str = DEFAULT_DPM100PSD_ROOT) -> Any:
    scene_spec = _scene_spec_from_value(spec, default_root)
    root = str(scene_spec["root"])
    index = int(scene_spec["index"])
    if root == str(default_root or DEFAULT_DPM100PSD_ROOT):
        return index
    return f"{root}:{index}"


def parse_scene_indices(
    value: Any,
    default: Sequence[Any] = DEFAULT_RADIOSEER_SCENE_SPECS,
    root: str = DEFAULT_DPM100PSD_ROOT,
) -> List[Any]:
    """Normalize CLI/config scene values into display-safe index/spec labels."""
    return [
        format_scene_spec(spec, default_root=root)
        for spec in parse_scene_specs(value, default=default, root=root)
    ]


def get_scene_specs(config: Any) -> List[Dict[str, Any]]:
    scene_cfg = getattr(config, "scene", None)
    value = getattr(scene_cfg, "radioseer_scene_indices", None)
    root = getattr(scene_cfg, "radioseer_root", DEFAULT_DPM100PSD_ROOT)
    return parse_scene_specs(value, root=str(root or DEFAULT_DPM100PSD_ROOT))


def get_scene_indices(config: Any) -> List[Any]:
    scene_cfg = getattr(config, "scene", None)
    value = getattr(scene_cfg, "radioseer_scene_indices", None)
    root = getattr(scene_cfg, "radioseer_root", DEFAULT_DPM100PSD_ROOT)
    return parse_scene_indices(value, root=str(root or DEFAULT_DPM100PSD_ROOT))


def configure_scene_config(
    config: Any,
    sample_index: Any,
    root: Optional[str] = None,
) -> Any:
    scene_cfg = getattr(config, "scene")
    default_root = str(
        root
        if root is not None
        else (getattr(scene_cfg, "radioseer_root", "") or DEFAULT_DPM100PSD_ROOT)
    )
    scene_spec = _scene_spec_from_value(sample_index, default_root)
    selected_root = str(scene_spec["root"])
    selected_index = int(scene_spec["index"])
    scene_cfg.radioseer_root = selected_root
    scene_cfg.radioseer_sample_index = selected_index
    if hasattr(scene_cfg, "radioseer_scene_indices"):
        scene_cfg.radioseer_scene_indices = [
            format_scene_spec(
                {"root": selected_root, "index": selected_index},
                default_root=DEFAULT_DPM100PSD_ROOT,
            )
        ]
    return config


def configure_scene_suite_config(
    config: Any,
    scene_indices: Optional[Sequence[Any]] = None,
    root: Optional[str] = None,
) -> List[Any]:
    scene_cfg = getattr(config, "scene")
    default_root = str(
        root
        if root is not None
        else (getattr(scene_cfg, "radioseer_root", "") or DEFAULT_DPM100PSD_ROOT)
    )
    value = scene_indices if scene_indices is not None else getattr(
        scene_cfg, "radioseer_scene_indices", None
    )
    specs = parse_scene_specs(value, root=default_root)
    labels = [
        format_scene_spec(spec, default_root=DEFAULT_DPM100PSD_ROOT)
        for spec in specs
    ]
    first_spec = specs[0]
    scene_cfg.radioseer_root = str(first_spec["root"])
    scene_cfg.radioseer_sample_index = int(first_spec["index"])
    setattr(scene_cfg, "radioseer_scene_indices", labels)
    return labels


def apply_scene_cli_overrides(config: Any, args: Any) -> None:
    """Apply unambiguous scene CLI overrides to a config object."""
    scene_cfg = getattr(config, "scene")
    if hasattr(args, "radioseer_root"):
        scene_cfg.radioseer_root = str(args.radioseer_root).strip() or scene_cfg.radioseer_root

    sample_index = getattr(args, "radioseer_sample_index", None)
    scene_indices = getattr(args, "radioseer_scene_indices", None)
    if sample_index is not None and scene_indices is not None:
        raise ValueError(
            "Use either --radioseer_sample_index for one scene or "
            "--radioseer_scene_indices for a scene suite, not both."
        )
    if scene_indices is not None:
        specs = parse_scene_specs(
            str(scene_indices),
            root=str(getattr(scene_cfg, "radioseer_root", DEFAULT_DPM100PSD_ROOT)),
        )
        scene_cfg.radioseer_scene_indices = [
            format_scene_spec(spec, default_root=DEFAULT_DPM100PSD_ROOT)
            for spec in specs
        ]
        scene_cfg.radioseer_root = str(specs[0]["root"])
        scene_cfg.radioseer_sample_index = int(specs[0]["index"])
    elif sample_index is not None:
        scene_cfg.radioseer_sample_index = int(sample_index)
        scene_cfg.radioseer_scene_indices = [
            format_scene_spec(
                {
                    "root": getattr(scene_cfg, "radioseer_root", DEFAULT_DPM100PSD_ROOT),
                    "index": int(sample_index),
                },
                default_root=DEFAULT_DPM100PSD_ROOT,
            )
        ]


def clone_config_for_scene(
    config: Any,
    sample_index: Any,
    root: Optional[str] = None,
) -> Any:
    return configure_scene_config(deepcopy(config), sample_index=sample_index, root=root)


def select_shared_data(shared_data: Any, env_idx: int) -> Dict:
    if isinstance(shared_data, (list, tuple)):
        if not shared_data:
            raise ValueError("shared_data scene suite must not be empty")
        return shared_data[int(env_idx) % len(shared_data)]
    return shared_data


def scene_config_from_shared_data(config: Any, shared_data: Mapping[str, Any]) -> Any:
    scene_meta = shared_data.get("config", {}) if isinstance(shared_data, Mapping) else {}
    sample_index = scene_meta.get(
        "sample_index",
        getattr(getattr(config, "scene", None), "radioseer_sample_index", 0),
    )
    dataset_root = scene_meta.get(
        "dataset_root",
        getattr(getattr(config, "scene", None), "radioseer_root", DEFAULT_DPM100PSD_ROOT),
    )
    scene_config = clone_config_for_scene(config, sample_index=int(sample_index), root=str(dataset_root))
    grid_size = grid_size_from_shared_data(shared_data)
    if grid_size is not None:
        scene_config.scene.grid_size = grid_size
    return scene_config


def _normalize_grid_size(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, Number):
        side = int(value)
        return (side, side)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
        if len(values) >= 2:
            return (values[0], values[1])
    return None


def grid_size_from_shared_data(shared_data: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    if not isinstance(shared_data, Mapping):
        return None

    h_tensor = shared_data.get("H")
    if h_tensor is not None:
        h_shape = np.asarray(h_tensor).shape
        if len(h_shape) >= 2:
            return (int(h_shape[0]), int(h_shape[1]))

    metadata = shared_data.get("radioseer_metadata", {})
    if isinstance(metadata, Mapping):
        grid_size = _normalize_grid_size(metadata.get("crop_size"))
        if grid_size is not None:
            return grid_size

    scene_meta = shared_data.get("config", {})
    if isinstance(scene_meta, Mapping):
        grid_size = _normalize_grid_size(scene_meta.get("grid_size"))
        if grid_size is not None:
            return grid_size
    return None


def _stamp_shared_grid_size(shared_data: Mapping[str, Any], grid_size: Tuple[int, int]) -> None:
    if not isinstance(shared_data, dict):
        return
    scene_meta = shared_data.setdefault("config", {})
    if isinstance(scene_meta, dict):
        scene_meta["grid_size"] = [int(grid_size[0]), int(grid_size[1])]


def sync_scene_grid_size_from_shared_data(config: Any, shared_data: Mapping[str, Any]) -> Tuple[int, int]:
    grid_size = grid_size_from_shared_data(shared_data)
    if grid_size is None:
        raise ValueError("shared scene data must expose H or metadata.crop_size to resolve grid_size")
    getattr(config, "scene").grid_size = grid_size
    _stamp_shared_grid_size(shared_data, grid_size)
    return grid_size


def sync_scene_grid_size_from_shared_suite(
    config: Any,
    shared_data_suite: Sequence[Mapping[str, Any]],
) -> Tuple[int, int]:
    if not shared_data_suite:
        raise ValueError("shared_data_suite must not be empty")

    grid_sizes: List[Tuple[int, int]] = []
    for shared_data in shared_data_suite:
        grid_size = grid_size_from_shared_data(shared_data)
        if grid_size is None:
            raise ValueError("shared scene data must expose H or metadata.crop_size to resolve grid_size")
        _stamp_shared_grid_size(shared_data, grid_size)
        grid_sizes.append(grid_size)

    unique_grid_sizes = sorted(set(grid_sizes))
    if len(unique_grid_sizes) != 1:
        raise ValueError(
            "All scenes in one training/evaluation suite must share the same grid_size; "
            f"got {unique_grid_sizes}."
        )
    getattr(config, "scene").grid_size = unique_grid_sizes[0]
    return unique_grid_sizes[0]


def build_shared_data_suite(
    config: Any,
    sim_data_cls: Any,
    data_seed: int,
    scene_indices: Optional[Sequence[Any]] = None,
) -> List[Dict]:
    scene_cfg = getattr(config, "scene", None)
    default_root = str(getattr(scene_cfg, "radioseer_root", DEFAULT_DPM100PSD_ROOT))
    value = scene_indices if scene_indices is not None else getattr(
        scene_cfg, "radioseer_scene_indices", None
    )
    specs = parse_scene_specs(value, root=default_root)
    shared_data = []
    for spec in specs:
        scene_config = clone_config_for_scene(
            config,
            sample_index=int(spec["index"]),
            root=str(spec["root"]),
        )
        shared_data.append(sim_data_cls(scene_config, seed=int(data_seed)).export_data())
    sync_scene_grid_size_from_shared_suite(config, shared_data)
    return shared_data


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _metric_float(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key, np.nan)
    return float(value) if _is_numeric(value) else float("nan")


def data_accounting_violation_bits(
    *,
    produced_bits: float,
    link_transmitted_bits: float,
    completed_packet_bits: float,
    delivered_bits: float,
    dropped_bits: float,
    final_queue_bits: float,
) -> float:
    """Return the largest physical data-flow invariant violation in bits."""
    produced = float(produced_bits)
    link = float(link_transmitted_bits)
    completed = float(completed_packet_bits)
    delivered = float(delivered_bits)
    dropped = float(dropped_bits)
    queued = float(final_queue_bits)
    values = (produced, link, completed, delivered, dropped, queued)
    if not all(np.isfinite(value) for value in values):
        return float("inf")

    raw_violation = float(
        max(
            0.0,
            abs(math.fsum((produced, -link, -dropped, -queued))),
            link - produced,
            completed - link,
            delivered - completed,
            -dropped,
            -queued,
        )
    )
    # Episode totals are accumulated at Gbit scale.  A few microbits of
    # cancellation error are therefore possible even when every step obeys
    # exact conservation.  Ignore only machine-scale roundoff; material
    # violations are returned unchanged and remain subject to the 1e-6-bit
    # experiment gate.
    scale = max(1.0, *(abs(value) for value in values))
    roundoff_tolerance = 8.0 * np.finfo(np.float64).eps * scale
    return 0.0 if raw_violation <= roundoff_tolerance else raw_violation


def is_strict_pareto_feasible(
    *,
    nmse: float,
    link_bits: float,
    energy_failure_rate: float,
    full_horizon_rate: float,
    max_data_accounting_violation_bits: float,
    accounting_tolerance_bits: float = 1e-6,
) -> bool:
    """Apply hard physical-validity gates before NMSE/link Pareto ranking."""
    values = (
        nmse,
        link_bits,
        energy_failure_rate,
        full_horizon_rate,
        max_data_accounting_violation_bits,
    )
    if not all(np.isfinite(float(value)) for value in values):
        return False
    return bool(
        float(energy_failure_rate) <= 1e-12
        and float(full_horizon_rate) >= 1.0 - 1e-12
        and float(max_data_accounting_violation_bits)
        <= float(accounting_tolerance_bits)
    )


def dominates_nmse_link(
    left_nmse: float,
    left_link_bits: float,
    right_nmse: float,
    right_link_bits: float,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether the left NMSE/link-bit point strictly dominates the right."""
    no_worse = (
        float(left_nmse) <= float(right_nmse) + float(tolerance)
        and float(left_link_bits) <= float(right_link_bits) + float(tolerance)
    )
    strictly_better = (
        float(left_nmse) < float(right_nmse) - float(tolerance)
        or float(left_link_bits) < float(right_link_bits) - float(tolerance)
    )
    return bool(no_worse and strictly_better)


def _finite_metric_values(
    scene_results: Sequence[Mapping[str, Any]],
    key: str,
) -> List[float]:
    return [
        float(result[key])
        for result in scene_results
        if key in result and _is_numeric(result[key])
    ]


def _pooled_population_std(
    scene_results: Sequence[Mapping[str, Any]],
    mean_key: str,
    std_key: str,
) -> float:
    """Pool per-scene population moments without discarding between-scene spread."""
    total_count = 0
    total_sum = 0.0
    total_second_moment = 0.0
    for result in scene_results:
        count = int(result.get("eval_num_episodes", 0))
        mean = result.get(mean_key)
        std = result.get(std_key)
        if count <= 0 or not _is_numeric(mean) or not _is_numeric(std):
            continue
        mean = float(mean)
        std = float(std)
        total_count += count
        total_sum += count * mean
        total_second_moment += count * (std * std + mean * mean)
    if total_count <= 0:
        return float("nan")
    pooled_mean = total_sum / float(total_count)
    pooled_variance = max(
        total_second_moment / float(total_count) - pooled_mean * pooled_mean,
        0.0,
    )
    return float(np.sqrt(pooled_variance))


def _extreme_scene_result(
    scene_results: Sequence[Mapping[str, Any]],
    key: str,
    choose_max: bool = False,
) -> Optional[Mapping[str, Any]]:
    candidates = [
        (float(result[key]), result)
        for result in scene_results
        if key in result and _is_numeric(result[key])
    ]
    if not candidates:
        return None
    selector = max if choose_max else min
    return selector(candidates, key=lambda item: item[0])[1]


def aggregate_eval_results(
    scene_results: Sequence[Mapping[str, Any]],
    seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    """Average evaluation metrics by scene, keeping artifacts from the last scene."""
    if not scene_results:
        raise ValueError("scene_results must not be empty")

    aggregate: Dict[str, Any] = {}
    keys = set().union(*(result.keys() for result in scene_results))
    for key in sorted(keys):
        values = [result[key] for result in scene_results if key in result]
        numeric_values = [float(value) for value in values if _is_numeric(value)]
        if numeric_values and len(numeric_values) == len(values):
            aggregate[key] = float(np.mean(numeric_values))

    # Generic averaging is correct for scene-balanced mean metrics, but not for
    # extrema or standard deviations. Reconstruct the cross-scene statistics
    # explicitly so one episode per scene does not misleadingly report std=0.
    extrema = {
        "eval_best_nmse": np.min,
        "eval_worst_nmse": np.max,
        "eval_min_steps": np.min,
        "eval_min_uav_energy_remaining": np.min,
        "eval_max_uav_energy_remaining": np.max,
        "eval_min_data_delivered_bits": np.min,
        "eval_max_data_delivered_bits": np.max,
        "eval_min_service_completion_ratio": np.min,
        "eval_min_link_transmitted_bits": np.min,
        "eval_max_link_transmitted_bits": np.max,
        "eval_max_data_accounting_violation_bits": np.max,
    }
    for key, reduction in extrema.items():
        values = _finite_metric_values(scene_results, key)
        if values:
            aggregate[key] = float(reduction(values))

    for mean_key, std_key in (
        ("eval_mean_nmse", "eval_std_nmse"),
        ("eval_mean_return", "eval_std_return"),
    ):
        pooled_std = _pooled_population_std(
            scene_results,
            mean_key=mean_key,
            std_key=std_key,
        )
        if np.isfinite(pooled_std):
            aggregate[std_key] = pooled_std

    # Preserve episode pairing for conditional metrics instead of averaging
    # unrelated scenes.
    best_nmse_scene = _extreme_scene_result(
        scene_results,
        key="eval_best_nmse",
    )
    worst_nmse_scene = _extreme_scene_result(
        scene_results,
        key="eval_worst_nmse",
        choose_max=True,
    )
    min_link_scene = _extreme_scene_result(
        scene_results,
        key="eval_min_link_transmitted_bits",
    )
    for key in (
        "eval_data_delivered_bits_at_best_nmse",
        "eval_link_transmitted_bits_at_best_nmse",
        "eval_service_completion_ratio_at_best_nmse",
        "eval_outage_ratio_at_best_nmse",
    ):
        if best_nmse_scene is not None and key in best_nmse_scene:
            aggregate[key] = best_nmse_scene[key]
    for key in (
        "eval_data_delivered_bits_at_worst_nmse",
        "eval_link_transmitted_bits_at_worst_nmse",
        "eval_service_completion_ratio_at_worst_nmse",
        "eval_outage_ratio_at_worst_nmse",
    ):
        if worst_nmse_scene is not None and key in worst_nmse_scene:
            aggregate[key] = worst_nmse_scene[key]
    if (
        min_link_scene is not None
        and "eval_nmse_at_min_link_transmitted_bits" in min_link_scene
    ):
        aggregate["eval_nmse_at_min_link_transmitted_bits"] = min_link_scene[
            "eval_nmse_at_min_link_transmitted_bits"
        ]

    # Keep visualization artifacts and detailed step traces from one concrete scene.
    last_result = dict(scene_results[-1])
    for key in (
        "eval_uav_trajectory",
        "eval_ugv_trajectory",
        "eval_step_details",
        "eval_visualized_episode_index",
        "eval_visualized_reset_seed",
    ):
        if key in last_result:
            aggregate[key] = last_result[key]
    if "eval_scene_sample_index" in last_result:
        aggregate["eval_artifact_scene_sample_index"] = int(last_result["eval_scene_sample_index"])

    per_scene_metrics = []
    for scene_pos, result in enumerate(scene_results):
        sample_index = int(result.get("eval_scene_sample_index", -1))
        sample_tag = str(result.get("eval_scene_sample_tag", "") or "")
        scene_label = sample_tag if sample_tag else str(sample_index)
        per_scene_metrics.append(
            {
                "scene_pos": int(scene_pos),
                "scene_sample_index": sample_index,
                "scene_sample_tag": sample_tag,
                "scene_label": scene_label,
                "eval_num_episodes": int(result.get("eval_num_episodes", 0)),
                "eval_reset_seed_base": int(result.get("eval_reset_seed_base", -1)),
                "eval_mean_nmse": _metric_float(result, "eval_mean_nmse"),
                "eval_mean_data_delivered_bits": _metric_float(
                    result,
                    "eval_mean_data_delivered_bits",
                ),
                "eval_mean_link_transmitted_bits": _metric_float(
                    result,
                    "eval_mean_link_transmitted_bits",
                ),
                "eval_mean_data_produced_bits": _metric_float(
                    result,
                    "eval_mean_data_produced_bits",
                ),
                "eval_mean_completed_packet_bits": _metric_float(
                    result,
                    "eval_mean_completed_packet_bits",
                ),
                "eval_mean_dropped_bits": _metric_float(
                    result,
                    "eval_mean_dropped_bits",
                ),
                "eval_mean_final_queue_bits": _metric_float(
                    result,
                    "eval_mean_final_queue_bits",
                ),
                "eval_mean_service_completion_ratio": _metric_float(
                    result,
                    "eval_mean_service_completion_ratio",
                ),
                "eval_min_service_completion_ratio": _metric_float(
                    result,
                    "eval_min_service_completion_ratio",
                ),
                "eval_mean_outage_ratio": _metric_float(
                    result,
                    "eval_mean_outage_ratio",
                ),
                "eval_mean_uav_energy_remaining": _metric_float(
                    result,
                    "eval_mean_uav_energy_remaining",
                ),
                "eval_mean_ugv_comm_target_ratio": _metric_float(
                    result,
                    "eval_mean_ugv_comm_target_ratio",
                ),
                "eval_max_data_accounting_violation_bits": _metric_float(
                    result,
                    "eval_max_data_accounting_violation_bits",
                ),
                "eval_mean_prefill_observed_band_units": _metric_float(
                    result,
                    "eval_mean_prefill_observed_band_units",
                ),
                "eval_mean_prefill_equivalent_data_bits": _metric_float(
                    result,
                    "eval_mean_prefill_equivalent_data_bits",
                ),
            }
        )

    scene_indices = [
        int(result.get("eval_scene_sample_index", -1))
        for result in scene_results
    ]
    aggregate["eval_scene_count"] = int(len(scene_results))
    aggregate["eval_scene_suite"] = 1
    aggregate["eval_scene_sample_index"] = -1
    aggregate["eval_scene_sample_indices"] = scene_indices
    aggregate["eval_per_scene_metrics"] = per_scene_metrics
    aggregate["eval_per_scene_labels"] = [
        str(metric["scene_label"]) for metric in per_scene_metrics
    ]
    aggregate["eval_per_scene_mean_nmse"] = [
        float(metric["eval_mean_nmse"]) for metric in per_scene_metrics
    ]
    aggregate["eval_per_scene_mean_data_delivered_bits"] = [
        float(metric["eval_mean_data_delivered_bits"])
        for metric in per_scene_metrics
    ]
    aggregate["eval_per_scene_mean_link_transmitted_bits"] = [
        float(metric["eval_mean_link_transmitted_bits"])
        for metric in per_scene_metrics
    ]
    aggregate["eval_per_scene_mean_outage_ratio"] = [
        float(metric["eval_mean_outage_ratio"])
        for metric in per_scene_metrics
    ]
    aggregate["eval_per_scene_mean_service_completion_ratio"] = [
        float(metric["eval_mean_service_completion_ratio"])
        for metric in per_scene_metrics
    ]
    aggregate["eval_min_scene_service_completion_ratio"] = float(
        np.nanmin(aggregate["eval_per_scene_mean_service_completion_ratio"])
    )
    if seed_base is not None:
        aggregate["eval_reset_seed_base"] = int(seed_base)
    aggregate["eval_num_total_episodes"] = int(
        sum(int(result.get("eval_num_episodes", 0)) for result in scene_results)
    )
    aggregate["eval_num_episodes"] = aggregate["eval_num_total_episodes"]
    aggregate["eval_num_episodes_per_scene"] = [
        int(result.get("eval_num_episodes", 0)) for result in scene_results
    ]
    aggregate["eval_episode_records"] = [
        dict(record)
        for result in scene_results
        for record in result.get("eval_episode_records", [])
        if isinstance(record, Mapping)
    ]
    return aggregate


def evaluate_scene_suite(
    envs: Sequence[Any],
    evaluate_fn: Any,
    seed_base: Optional[int],
    scene_seed_stride: int = EVAL_SCENE_SEED_STRIDE,
    **evaluate_kwargs: Any,
) -> Dict[str, Any]:
    results = []
    for scene_pos, env in enumerate(envs):
        scene_seed_base = (
            None if seed_base is None else int(seed_base) + scene_pos * int(scene_seed_stride)
        )
        results.append(
            evaluate_fn(
                env=env,
                seed_base=scene_seed_base,
                **evaluate_kwargs,
            )
        )
    return aggregate_eval_results(results, seed_base=seed_base)


def format_scene_suite(indices: Sequence[Any]) -> str:
    return ", ".join(
        str(format_scene_spec(index, default_root=DEFAULT_DPM100PSD_ROOT))
        for index in indices
    )

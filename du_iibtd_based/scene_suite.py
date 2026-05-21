"""Shared RadioSeerDPM100PSD scene suite for du_iibtd_based experiments."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_DPM100PSD_ROOT = "RadioSeerDPM100PSD"

# DPM100PSD manifest rows aligned with the UAVTest-style scenes plus scene218.
DEFAULT_DPM100PSD_SCENE_INDICES = (8513, 1807, 1651, 1371, 10001)

# Keep train/eval on the same scene suite while separating deterministic resets.
EVAL_SCENE_SEED_STRIDE = 10_000


def parse_scene_indices(value: Any, default: Sequence[int] = DEFAULT_DPM100PSD_SCENE_INDICES) -> List[int]:
    """Normalize CLI/config scene index values into a non-empty integer list."""
    if value is None:
        indices = list(default)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            indices = list(default)
        else:
            for sep in (";", " "):
                text = text.replace(sep, ",")
            indices = [int(part.strip()) for part in text.split(",") if part.strip()]
    elif isinstance(value, Number):
        indices = [int(value)]
    elif isinstance(value, Iterable):
        indices = [int(item) for item in value]
    else:
        raise TypeError(f"Unsupported scene index value: {value!r}")

    if not indices:
        raise ValueError("scene index suite must not be empty")
    return indices


def get_scene_indices(config: Any) -> List[int]:
    scene_cfg = getattr(config, "scene", None)
    value = getattr(scene_cfg, "radioseer_scene_indices", None)
    return parse_scene_indices(value)


def configure_scene_config(
    config: Any,
    sample_index: int,
    root: Optional[str] = None,
) -> Any:
    scene_cfg = getattr(config, "scene")
    scene_cfg.radioseer_root = str(
        root
        if root is not None
        else (getattr(scene_cfg, "radioseer_root", "") or DEFAULT_DPM100PSD_ROOT)
    )
    scene_cfg.radioseer_sample_index = int(sample_index)
    if hasattr(scene_cfg, "radioseer_scene_indices"):
        scene_cfg.radioseer_scene_indices = [int(sample_index)]
    return config


def configure_scene_suite_config(
    config: Any,
    scene_indices: Optional[Sequence[int]] = None,
    root: Optional[str] = None,
) -> List[int]:
    indices = parse_scene_indices(scene_indices, default=get_scene_indices(config))
    scene_cfg = getattr(config, "scene")
    scene_cfg.radioseer_root = str(
        root
        if root is not None
        else (getattr(scene_cfg, "radioseer_root", "") or DEFAULT_DPM100PSD_ROOT)
    )
    scene_cfg.radioseer_sample_index = int(indices[0])
    setattr(scene_cfg, "radioseer_scene_indices", [int(index) for index in indices])
    return [int(index) for index in indices]


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
        scene_cfg.radioseer_scene_indices = str(scene_indices)
        parsed_indices = parse_scene_indices(scene_cfg.radioseer_scene_indices)
        scene_cfg.radioseer_sample_index = int(parsed_indices[0])
    elif sample_index is not None:
        scene_cfg.radioseer_sample_index = int(sample_index)
        scene_cfg.radioseer_scene_indices = [int(sample_index)]


def clone_config_for_scene(
    config: Any,
    sample_index: int,
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
    return clone_config_for_scene(config, sample_index=int(sample_index), root=str(dataset_root))


def build_shared_data_suite(
    config: Any,
    sim_data_cls: Any,
    data_seed: int,
    scene_indices: Optional[Sequence[int]] = None,
) -> List[Dict]:
    indices = parse_scene_indices(scene_indices, default=get_scene_indices(config))
    shared_data = []
    for sample_index in indices:
        scene_config = clone_config_for_scene(config, sample_index=sample_index)
        shared_data.append(sim_data_cls(scene_config, seed=int(data_seed)).export_data())
    return shared_data


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


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

    # Keep visualization artifacts and detailed step traces from one concrete scene.
    last_result = dict(scene_results[-1])
    for key in ("eval_uav_trajectory", "eval_ugv_trajectory", "eval_step_details"):
        if key in last_result:
            aggregate[key] = last_result[key]
    if "eval_scene_sample_index" in last_result:
        aggregate["eval_artifact_scene_sample_index"] = int(last_result["eval_scene_sample_index"])

    scene_indices = [
        int(result.get("eval_scene_sample_index", -1))
        for result in scene_results
    ]
    aggregate["eval_scene_count"] = int(len(scene_results))
    aggregate["eval_scene_suite"] = 1
    aggregate["eval_scene_sample_index"] = -1
    aggregate["eval_scene_sample_indices"] = scene_indices
    if seed_base is not None:
        aggregate["eval_reset_seed_base"] = int(seed_base)
    aggregate["eval_num_total_episodes"] = int(
        sum(int(result.get("eval_num_episodes", 0)) for result in scene_results)
    )
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


def format_scene_suite(indices: Sequence[int]) -> str:
    return ", ".join(str(int(index)) for index in indices)

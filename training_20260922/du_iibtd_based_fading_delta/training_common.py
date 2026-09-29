"""Variant-independent helpers used by both based training executors."""

from __future__ import annotations

import __main__
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from du_iibtd_based_fading_delta.scene_suite import data_accounting_violation_bits

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None


def parse_bool_arg(value: Any) -> bool:
    """Parse explicit boolean CLI values without Python's truthiness traps."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def make_episode_trackers(
    num_envs: int,
    keys: List[str],
) -> Tuple[List[Dict[str, float]], List[int]]:
    """Create per-environment accumulators for episode-summed metrics."""
    return (
        [{key: 0.0 for key in keys} for _ in range(int(num_envs))],
        [0 for _ in range(int(num_envs))],
    )


def reset_episode_tracker_entry(
    accumulators: List[Dict[str, float]],
    step_counts: List[int],
    env_idx: int,
    keys: List[str],
) -> None:
    for key in keys:
        accumulators[env_idx][key] = 0.0
    step_counts[env_idx] = 0


def update_episode_trackers(
    logger: Any,
    infos: List[dict],
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    accumulators: List[Dict[str, float]],
    step_counts: List[int],
    keys: List[str],
) -> int:
    """Accumulate step metrics and emit complete episode records."""
    episodes_finished = 0
    for env_idx, info in enumerate(infos):
        step_counts[env_idx] += 1
        for key in keys:
            accumulators[env_idx][key] += float(info.get(key, 0.0))

        if not (bool(terminateds[env_idx]) or bool(truncateds[env_idx])):
            continue

        episode_info = dict(info)
        for key in keys:
            episode_info[key] = float(accumulators[env_idx][key])
        episode_info["data_accounting_violation_bits"] = (
            data_accounting_violation_bits(
                produced_bits=episode_info["data_produced_bits"],
                link_transmitted_bits=episode_info["link_transmitted_bits"],
                completed_packet_bits=episode_info["completed_packet_bits"],
                delivered_bits=episode_info["data_delivered_bits"],
                dropped_bits=episode_info["dropped_bits"],
                final_queue_bits=float(info.get("queue_bits_after_tx", 0.0)),
            )
        )
        episode_info["episode_steps"] = int(step_counts[env_idx])
        logger.log_episode(episode_info)
        reset_episode_tracker_entry(accumulators, step_counts, env_idx, keys)
        episodes_finished += 1
    return episodes_finished


def make_done_array(terminateds: np.ndarray, truncateds: np.ndarray) -> np.ndarray:
    return np.logical_or(terminateds, truncateds).astype(np.float32)


def compute_timeout_bootstrap_values(
    policy: Any,
    infos: List[dict],
    terminateds: np.ndarray,
    truncateds: np.ndarray,
) -> np.ndarray:
    """Bootstrap a centralized critic only for time-limit truncations."""
    timeout_values = np.zeros(len(infos), dtype=np.float32)
    timeout_indices: List[int] = []
    timeout_states: List[np.ndarray] = []

    for env_idx, info in enumerate(infos):
        if not bool(truncateds[env_idx]) or bool(terminateds[env_idx]):
            continue
        terminal_obs = info.get("terminal_obs")
        if not isinstance(terminal_obs, dict) or "critic_state" not in terminal_obs:
            continue
        timeout_indices.append(env_idx)
        timeout_states.append(np.asarray(terminal_obs["critic_state"], dtype=np.float32))

    if timeout_indices:
        values = np.asarray(
            policy.get_value(np.stack(timeout_states, axis=0)),
            dtype=np.float32,
        )
        timeout_values[np.asarray(timeout_indices, dtype=int)] = values
    return timeout_values


def compute_ippo_timeout_bootstrap_values(
    policy: Any,
    infos: List[dict],
    terminateds: np.ndarray,
    truncateds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Bootstrap the two independent critics only for time-limit truncations."""
    timeout_values = {
        "uav_timeout_value": np.zeros(len(infos), dtype=np.float32),
        "ugv_timeout_value": np.zeros(len(infos), dtype=np.float32),
    }
    timeout_indices: List[int] = []
    timeout_uav_obs: List[np.ndarray] = []
    timeout_ugv_obs: List[np.ndarray] = []
    for env_idx, info in enumerate(infos):
        if not bool(truncateds[env_idx]) or bool(terminateds[env_idx]):
            continue
        terminal_obs = info.get("terminal_obs")
        if not isinstance(terminal_obs, dict):
            continue
        if "uav_obs" not in terminal_obs or "ugv_obs" not in terminal_obs:
            continue
        timeout_indices.append(env_idx)
        timeout_uav_obs.append(np.asarray(terminal_obs["uav_obs"], dtype=np.float32))
        timeout_ugv_obs.append(np.asarray(terminal_obs["ugv_obs"], dtype=np.float32))

    if timeout_indices:
        values = policy.get_values(
            uav_obs=np.stack(timeout_uav_obs, axis=0),
            ugv_obs=np.stack(timeout_ugv_obs, axis=0),
        )
        indices = np.asarray(timeout_indices, dtype=int)
        timeout_values["uav_timeout_value"][indices] = np.asarray(
            values["uav_value"], dtype=np.float32
        )
        timeout_values["ugv_timeout_value"][indices] = np.asarray(
            values["ugv_value"], dtype=np.float32
        )
    return timeout_values


def print_progress(
    update: int,
    total_updates: int,
    global_step: int,
    total_transitions: int,
    update_metrics: Dict[str, float],
    episodes_finished: int,
) -> None:
    line = (
        f"[Progress] update {update}/{total_updates} | "
        f"steps {global_step}/{total_transitions} | "
        f"uav pi loss {update_metrics['uav_policy_loss']:.3f} | "
        f"ugv pi loss {update_metrics['ugv_policy_loss']:.3f} | "
        f"v {update_metrics['value_loss']:.3f}"
    )
    if episodes_finished > 0:
        line += f" | ep {episodes_finished}"
    print(line)


def _cuda_device_count() -> int:
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _normalize_device_string(device: str, default_cuda_index: int = 0) -> str:
    device_str = str(device or "").strip().lower()
    if not device_str:
        return "cpu"
    if device_str == "cuda":
        return f"cuda:{int(default_cuda_index)}"
    return device_str


def _extract_cuda_index(device: str) -> Optional[int]:
    device_str = str(device or "").strip().lower()
    if not device_str.startswith("cuda"):
        return None
    if device_str == "cuda" or ":" not in device_str:
        return 0
    try:
        index = int(device_str.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device string {device!r}") from exc
    if index < 0:
        raise ValueError(f"CUDA device index must be non-negative, got {device!r}")
    return index


def _pick_visible_cuda_index(
    cuda_count: int,
    preferred_indices: List[int],
    blocked_indices: Optional[List[int]] = None,
) -> int:
    blocked = {int(idx) for idx in (blocked_indices or [])}
    for idx in preferred_indices:
        idx = int(idx)
        if 0 <= idx < cuda_count and idx not in blocked:
            return idx
    for idx in range(cuda_count):
        if idx not in blocked:
            return idx
    return 0


def _preferred_train_cuda_index(cuda_count: int) -> int:
    return _pick_visible_cuda_index(cuda_count, preferred_indices=[0])


def _preferred_planner_cuda_index(
    cuda_count: int,
    train_idx: Optional[int],
) -> int:
    blocked = [int(train_idx)] if train_idx is not None and cuda_count > 1 else []
    return _pick_visible_cuda_index(
        cuda_count,
        preferred_indices=[0],
        blocked_indices=blocked,
    )


def _resolve_iibtd_runtime_backend(config: Any) -> str:
    """Return the effective IIBTD compute class (CPU/GPU), not solver variant."""
    device = str(config.planner.iibtd_device).strip().lower()
    if device == "cpu":
        return "cpu"
    return "gpu" if (torch is not None and torch.cuda.is_available()) else "cpu"


def _resolve_iibtd_runtime_device(config: Any) -> str:
    device = str(config.planner.iibtd_device).strip()
    if not device or device.lower() == "auto":
        return _normalize_device_string(config.training.device)
    return _normalize_device_string(device)


def _running_from_interactive_main() -> bool:
    main_file = getattr(__main__, "__file__", None)
    if main_file is None:
        return True
    main_file = str(main_file).strip()
    return (not main_file) or main_file.startswith("<")


def _apply_runtime_backend_policy(config: Any) -> Optional[str]:
    """Resolve CUDA devices while preserving every explicit device selection.

    Bare ``cuda`` selects cuda:0.  A bare/auto IIBTD device uses another
    visible GPU when possible; otherwise both computations share one GPU.
    """
    cuda_count = _cuda_device_count()
    train_device_arg = str(config.training.device).strip().lower()
    planner_device_arg = str(config.planner.iibtd_device).strip().lower()
    for label, device in (
        ("training", train_device_arg),
        ("DU-IIBTD", planner_device_arg),
    ):
        if not device.startswith("cuda"):
            continue
        index = _extract_cuda_index(device)
        if cuda_count <= 0:
            raise RuntimeError(
                f"{label} device {device!r} was explicitly requested, but CUDA is unavailable"
            )
        if index is not None and index >= cuda_count:
            raise ValueError(
                f"{label} device {device!r} is outside the {cuda_count} visible CUDA device(s)"
            )

    compute_backend = _resolve_iibtd_runtime_backend(config)
    if compute_backend != "gpu" or cuda_count <= 0:
        return None

    messages: List[str] = []
    preferred_train_idx = _preferred_train_cuda_index(cuda_count)

    if not train_device_arg or train_device_arg == "cuda":
        config.training.device = _normalize_device_string(
            config.training.device,
            default_cuda_index=preferred_train_idx,
        )
    train_device = _normalize_device_string(
        config.training.device,
        default_cuda_index=preferred_train_idx,
    )

    if not planner_device_arg or planner_device_arg in {"auto", "cuda"}:
        train_idx = _extract_cuda_index(train_device)
        planner_default_idx = _preferred_planner_cuda_index(cuda_count, train_idx)
        config.planner.iibtd_device = _normalize_device_string(
            "cuda",
            default_cuda_index=planner_default_idx,
        )
        messages.append(
            f"defaulted DU-IIBTD device {config.planner.iibtd_device} "
            f"(visible CUDA devices: {cuda_count})"
        )

    iibtd_device = _resolve_iibtd_runtime_device(config)
    if not train_device.startswith("cuda") or not iibtd_device.startswith("cuda"):
        return "; ".join(messages) if messages else None

    if train_device != iibtd_device:
        messages.insert(
            0,
            f"policy training uses {train_device} and DU-IIBTD uses {iibtd_device}",
        )
    else:
        messages.insert(
            0,
            f"policy training and DU-IIBTD both use {train_device}",
        )
    return "; ".join(messages)


__all__ = [
    "parse_bool_arg",
    "make_episode_trackers",
    "reset_episode_tracker_entry",
    "update_episode_trackers",
    "make_done_array",
    "compute_timeout_bootstrap_values",
    "compute_ippo_timeout_bootstrap_values",
    "print_progress",
    "_apply_runtime_backend_policy",
    "_resolve_iibtd_runtime_backend",
    "_resolve_iibtd_runtime_device",
    "_running_from_interactive_main",
]

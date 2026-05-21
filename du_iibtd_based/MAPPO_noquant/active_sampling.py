"""
Active-sampling helpers adapted from Test/iibtdEnsembleTest.py.

This module keeps environment.py focused on multi-agent orchestration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from du_iibtd_based.du_iibtd_learn_nu import (
    checkpoint_config_float as _checkpoint_config_float,
    default_du_iibtd_checkpoints_for_backend as _default_du_iibtd_checkpoints_for_backend,
    is_du_iibtd_backend as _is_du_iibtd_backend,
    resolve_solver_hyperparams as _checkpoint_solver_hyperparams,
)

try:
    from DU_IIBTD.solver_adapter import DU_IIBTDSolverAdapter
except ModuleNotFoundError:  # pragma: no cover - surfaced when reconstruction starts
    DU_IIBTDSolverAdapter = None

try:
    from DU_IIBTD_res_Sr.solver_adapter import DU_IIBTDSolverAdapter as DU_IIBTDResSrSolverAdapter
except ModuleNotFoundError:  # pragma: no cover - surfaced when reconstruction starts
    DU_IIBTDResSrSolverAdapter = None

try:
    from DU_IIBTD_res_Sr_learn_nu.solver_adapter import (
        DU_IIBTDSolverAdapter as DU_IIBTDResSrLearnNuSolverAdapter,
    )
except ModuleNotFoundError:  # pragma: no cover - surfaced when reconstruction starts
    DU_IIBTDResSrLearnNuSolverAdapter = None


DEFAULT_DU_IIBTD_MIN_SENSORS_FOR_UPDATE = 0
DEFAULT_DU_IIBTD_UPDATE_BATCH_SIZE = 0
DEFAULT_ENSEMBLE_INIT_JITTER_SCALE = 1e-3


def _du_iibtd_adapter_for_backend(backend: str):
    backend = str(backend or "du_iibtd").strip().lower()
    if backend == "du_iibtd":
        return DU_IIBTDSolverAdapter, "DU-IIBTD"
    if backend == "du_iibtd_res_sr":
        return DU_IIBTDResSrSolverAdapter, "DU-IIBTD_res_Sr"
    if backend == "du_iibtd_res_sr_learn_nu":
        return DU_IIBTDResSrLearnNuSolverAdapter, "DU-IIBTD_res_Sr_learn_nu"
    return None, str(backend)


def quantize_to_8bit(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Uniform 8-bit quantization with de-quantized float output."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy(), {"min": 0.0, "max": 0.0, "levels": 256.0}

    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max <= v_min + 1e-12:
        return np.full_like(arr, v_min, dtype=float), {
            "min": v_min,
            "max": v_max,
            "levels": 256.0,
        }

    norm = (arr - v_min) / (v_max - v_min)
    q_uint8 = np.round(norm * 255.0).astype(np.uint8)
    dq = (q_uint8.astype(float) / 255.0) * (v_max - v_min) + v_min
    return dq, {"min": v_min, "max": v_max, "levels": 256.0}


def build_observe_mask(
    num_bands: int,
    center_freq: int,
    band_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build contiguous observation mask around a center frequency."""
    width = int(np.clip(band_width, 1, num_bands))
    center = int(np.clip(center_freq, 0, num_bands - 1))

    f_start = center - (width // 2)
    f_end = f_start + width
    if f_start < 0:
        f_start = 0
        f_end = width
    elif f_end > num_bands:
        f_end = num_bands
        f_start = num_bands - width

    omega = np.zeros(num_bands, dtype=np.int32)
    omega[f_start:f_end] = 1
    observed_bands = np.where(omega > 0)[0]
    return omega, observed_bands


def _copy_model_state(dst_model, src_model) -> None:
    """Copy warm-start state between solvers with the same dimensions."""
    if hasattr(dst_model, "load_state_from"):
        dst_model.load_state_from(src_model)
    else:
        for name in ("Theta", "Phi", "Sr", "H_hat"):
            if hasattr(src_model, name):
                setattr(dst_model, name, np.array(getattr(src_model, name), copy=True))
    if hasattr(dst_model, "_initialized"):
        dst_model._initialized = True


def release_reconstruction_model(model) -> bool:
    """Release GPU-backed tensors for a retired reconstruction model."""
    if model is None:
        return False

    release_fn = getattr(model, "release_device_memory", None)
    if callable(release_fn):
        release_fn()
        device = getattr(model, "device", None)
        return bool(device is not None and str(device).startswith("cuda"))

    close_fn = getattr(model, "close", None)
    if callable(close_fn):
        close_fn()
    return False


def _resolve_du_iibtd_device(solver_device: Optional[str] = None) -> str:
    device_str = str(solver_device or "auto").strip()
    if not device_str or device_str.lower() == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return device_str


def _normalize_checkpoint_paths(
    checkpoint_paths: Optional[List[str]],
    backend: str = "du_iibtd",
) -> List[str]:
    paths = list(checkpoint_paths or [])
    if not paths and _is_du_iibtd_backend(backend):
        paths = _default_du_iibtd_checkpoints_for_backend(backend)
    resolved = []
    for path in paths:
        p = Path(str(path)).expanduser()
        if not p.is_absolute():
            p = Path(_PROJECT_ROOT) / p
        resolved.append(str(p))
    if not resolved:
        raise ValueError("At least one DU-IIBTD checkpoint is required.")
    return resolved


def _resolve_member_checkpoint(
    checkpoint_paths: Optional[List[str]],
    member_idx: int = 0,
    backend: str = "du_iibtd",
) -> str:
    paths = _normalize_checkpoint_paths(checkpoint_paths, backend=backend)
    return paths[int(member_idx) % len(paths)]


def _jitter_model_initial_state(model, scale: float) -> None:
    if model is None or float(scale) <= 0.0:
        return
    jitter_fn = getattr(model, "jitter_state", None)
    if callable(jitter_fn):
        jitter_fn(scale=float(scale))


def _resolve_iibtd_device(solver_device: Optional[str] = None) -> str:
    device_str = str(solver_device or "auto").strip()
    if not device_str or device_str.lower() == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return device_str


def _make_reconstruction_solver(
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int,
    mu: float,
    nu: float,
    kernel_bandwidth: float,
    warmstart: bool,
    backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoint_path: Optional[str] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
):
    backend = str(backend or "du_iibtd").strip().lower()
    if not _is_du_iibtd_backend(backend):
        raise ValueError(
            "du_iibtd_based only supports backend='du_iibtd' or "
            "backend='du_iibtd_res_sr' or backend='du_iibtd_res_sr_learn_nu', got "
            f"{backend!r}"
        )
    adapter_cls, adapter_name = _du_iibtd_adapter_for_backend(backend)
    if adapter_cls is None:
        raise RuntimeError(f"{adapter_name} solver adapter is unavailable.")
    if torch is None:
        raise RuntimeError(f"{adapter_name} backend requires torch.")
    if du_iibtd_checkpoint_path is None:
        du_iibtd_checkpoint_path = _resolve_member_checkpoint(
            None,
            member_idx=0,
            backend=backend,
        )
    checkpoint_path = str(du_iibtd_checkpoint_path)
    solver_hyperparams = _checkpoint_solver_hyperparams(
        checkpoint_path,
        fallback_nu=float(nu),
        fallback_kernel_bandwidth=float(kernel_bandwidth),
        fallback_min_sensors_for_update=(
            DEFAULT_DU_IIBTD_MIN_SENSORS_FOR_UPDATE
            if du_iibtd_min_sensors_for_update is None
            else int(du_iibtd_min_sensors_for_update)
        ),
        fallback_update_batch_size=du_iibtd_update_batch_size,
    )
    return adapter_cls(
        n_sources=n_sources,
        grid_size=grid_size,
        mu=float(mu),
        nu=float(solver_hyperparams["nu"]),
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=float(solver_hyperparams["kernel_bandwidth"]),
        warmstart=bool(warmstart),
        checkpoint_path=checkpoint_path,
        device=_resolve_du_iibtd_device(solver_device),
        dtype=torch.float32,
        min_sensors_for_update=solver_hyperparams["min_sensors_for_update"],
        update_batch_size=solver_hyperparams["update_batch_size"],
    )


def init_reconstruction_solver(
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    K: int,
    i_mask: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int = 6,
    mu: float = 1.2,
    nu: float = 1.5,
    kernel_bandwidth: float = 0.46,
    warmstart: bool = False,
    solver_backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoint_path: Optional[str] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
):
    """Create and initialize one DU-IIBTD solver adapter instance."""
    model = _make_reconstruction_solver(
        n_sources=n_sources,
        grid_size=grid_size,
        max_iter=max_iter,
        mu=mu,
        nu=nu,
        kernel_bandwidth=kernel_bandwidth,
        warmstart=warmstart,
        backend=solver_backend,
        solver_device=solver_device,
        du_iibtd_checkpoint_path=du_iibtd_checkpoint_path,
        du_iibtd_min_sensors_for_update=du_iibtd_min_sensors_for_update,
        du_iibtd_update_batch_size=du_iibtd_update_batch_size,
    )
    model.init_sequential(grid_points, bounds, K=K, I_mask=i_mask)
    return model


def fit_reconstruction_model(
    obs_locs: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    i_mask: np.ndarray,
    prev_model: Optional[object] = None,
    warmstart: bool = False,
    max_iter: int = 6,
    mu: float = 1.2,
    nu: float = 1.5,
    kernel_bandwidth: float = 0.46,
    solver_backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoint_path: Optional[str] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
    init_seed: Optional[int] = None,
    init_jitter_scale: float = DEFAULT_ENSEMBLE_INIT_JITTER_SCALE,
):
    """Fit one reconstruction model, optionally warm-started from prev_model."""
    model = _make_reconstruction_solver(
        n_sources=n_sources,
        grid_size=grid_size,
        max_iter=max_iter,
        mu=mu,
        nu=nu,
        kernel_bandwidth=kernel_bandwidth,
        warmstart=warmstart,
        backend=solver_backend,
        solver_device=solver_device,
        du_iibtd_checkpoint_path=du_iibtd_checkpoint_path,
        du_iibtd_min_sensors_for_update=du_iibtd_min_sensors_for_update,
        du_iibtd_update_batch_size=du_iibtd_update_batch_size,
    )
    if init_seed is not None:
        init_seed_int = int(init_seed)
        np_state = np.random.get_state()
        if torch is not None:
            try:
                torch.manual_seed(init_seed_int)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(init_seed_int)
            except Exception:
                pass
        np.random.seed(init_seed_int % (2 ** 32 - 1))
    else:
        np_state = None
    try:
        model.init_sequential(grid_points, bounds, K=np.atleast_2d(gamma).shape[1], I_mask=i_mask)
    finally:
        if np_state is not None:
            np.random.set_state(np_state)
    if warmstart and prev_model is not None:
        _copy_model_state(model, prev_model)
    else:
        _jitter_model_initial_state(model, scale=init_jitter_scale)
    model.fit_incremental(
        obs_locs,
        gamma,
        omega,
        grid_coords=grid_points,
        bounds=bounds,
        I_mask=i_mask,
        n_outer_iter=1,
        debugFlag=False,
    )
    return model


def _predict_observed_rows(
    map_hat: np.ndarray,
    obs_locs: np.ndarray,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Project reconstructed map to observed grid rows using nearest integer grid."""
    locs = np.asarray(obs_locs, dtype=float)
    grid_xy = np.floor(locs).astype(int)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, n1 - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, n2 - 1)
    return map_hat[grid_xy[:, 0], grid_xy[:, 1], :]


def _observation_nmse(
    map_hat: np.ndarray,
    obs_locs: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    n1: int,
    n2: int,
) -> float:
    """Evaluate fit quality on observed entries only."""
    mask = np.asarray(omega) > 0
    if not np.any(mask):
        return np.inf

    pred = _predict_observed_rows(map_hat, obs_locs, n1, n2)
    diff = pred[mask] - gamma[mask]
    denom = np.linalg.norm(gamma[mask]) ** 2
    return float(np.linalg.norm(diff) ** 2 / (denom + 1e-9))


def _weighted_map_statistics(
    maps: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean/variance over ensemble members."""
    maps = np.asarray(maps, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if maps.shape[0] == 1:
        return maps[0], np.zeros_like(maps[0]), np.ones(1, dtype=float)

    weights = np.where(np.isfinite(scores), np.maximum(scores, 0.0), 0.0)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.full(maps.shape[0], 1.0 / maps.shape[0], dtype=float)
    else:
        weights = weights / np.sum(weights)
    mean_map = np.tensordot(weights, maps, axes=(0, 0))
    diff = maps - mean_map[np.newaxis, ...]
    var_map = np.tensordot(weights, diff ** 2, axes=(0, 0))
    return mean_map, var_map, weights


def _summarize_ensemble_models(
    member_models: List[object],
    *,
    fused_obs_locs: np.ndarray,
    fused_gamma: np.ndarray,
    fused_omega: np.ndarray,
    quality_weighted: bool = True,
    member_observation_counts: Optional[np.ndarray] = None,
    member_last_incremental_accept_counts: Optional[np.ndarray] = None,
    member_incremental_updated_mask: Optional[np.ndarray] = None,
    member_kernel_bandwidths: Optional[np.ndarray] = None,
    member_nus: Optional[np.ndarray] = None,
    member_du_iibtd_checkpoints: Optional[List[str]] = None,
    fusion_meta: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Aggregate member maps into ensemble mean/variance plus bookkeeping."""
    member_models = list(member_models or [])
    if not member_models:
        raise RuntimeError("No ensemble members are available.")

    maps: List[np.ndarray] = []
    obs_nmse_scores: List[float] = []
    for model in member_models:
        map_m = model.get_current_map()
        maps.append(np.asarray(map_m, dtype=float))
        obs_nmse_scores.append(
            _observation_nmse(
                map_m,
                fused_obs_locs,
                fused_gamma,
                fused_omega,
                map_m.shape[0],
                map_m.shape[1],
            )
        )

    stack = np.stack(maps, axis=0)
    if quality_weighted:
        quality_scores = 1.0 / (np.asarray(obs_nmse_scores, dtype=float) + 1e-8)
        mean_map, var_map, weights = _weighted_map_statistics(stack, quality_scores)
    else:
        mean_map = np.mean(stack, axis=0)
        if stack.shape[0] > 1:
            var_map = np.var(stack, axis=0, ddof=1)
        else:
            var_map = np.zeros_like(mean_map)
        weights = np.full(stack.shape[0], 1.0 / stack.shape[0], dtype=float)

    info = {
        "member_models": list(member_models),
        "member_observation_counts": (
            np.asarray(member_observation_counts, dtype=int)
            if member_observation_counts is not None
            else np.full(len(member_models), -1, dtype=int)
        ),
        "member_last_incremental_accept_counts": (
            np.asarray(member_last_incremental_accept_counts, dtype=int)
            if member_last_incremental_accept_counts is not None
            else np.zeros(len(member_models), dtype=int)
        ),
        "member_incremental_updated_mask": (
            np.asarray(member_incremental_updated_mask, dtype=bool)
            if member_incremental_updated_mask is not None
            else np.zeros(len(member_models), dtype=bool)
        ),
        "member_kernel_bandwidths": (
            np.asarray(member_kernel_bandwidths, dtype=float)
            if member_kernel_bandwidths is not None
            else np.asarray([getattr(model, "h", np.nan) for model in member_models], dtype=float)
        ),
        "member_nus": (
            np.asarray(member_nus, dtype=float)
            if member_nus is not None
            else np.asarray([getattr(model, "nu", np.nan) for model in member_models], dtype=float)
        ),
        "member_du_iibtd_checkpoints": (
            list(member_du_iibtd_checkpoints)
            if member_du_iibtd_checkpoints is not None
            else [str(getattr(model, "checkpoint_path", "")) for model in member_models]
        ),
        "member_du_iibtd_min_sensors_for_update": [
            int(getattr(model, "min_sensors_for_update", -1)) for model in member_models
        ],
        "member_du_iibtd_update_batch_sizes": [
            None if getattr(model, "update_batch_size", None) is None else int(model.update_batch_size)
            for model in member_models
        ],
        "weights": np.asarray(weights, dtype=float),
        "obs_nmse": np.asarray(obs_nmse_scores, dtype=float),
        "fusion_meta": dict(fusion_meta or {}),
    }
    return mean_map, var_map, stack, info


def _share_incremental_rows_across_members(
    new_obs_locs: np.ndarray,
    new_gamma: np.ndarray,
    new_omega: np.ndarray,
    *,
    member_count: int,
    member_observation_counts: np.ndarray,
) -> Tuple[List[Optional[Dict[str, np.ndarray]]], np.ndarray, np.ndarray, np.ndarray]:
    """Give every DU-IIBTD ensemble member the same newly fused observations."""
    new_obs_locs = np.asarray(new_obs_locs, dtype=float)
    new_gamma = np.asarray(new_gamma, dtype=float)
    new_omega = np.asarray(new_omega, dtype=np.int32)
    member_count = int(member_count)
    current_counts = np.asarray(member_observation_counts, dtype=int).copy()
    if current_counts.shape[0] != member_count:
        raise RuntimeError("member_observation_counts does not match member count.")

    row_count = int(new_obs_locs.shape[0])
    if row_count == 0:
        return (
            [None] * member_count,
            current_counts,
            np.zeros(member_count, dtype=int),
            np.zeros(member_count, dtype=bool),
        )

    current_counts += row_count
    member_batches = [
        {
            "obs_locs": new_obs_locs.copy(),
            "gamma": new_gamma.copy(),
            "omega": new_omega.copy(),
        }
        for _ in range(member_count)
    ]
    accept_counts = np.full(member_count, row_count, dtype=int)
    return member_batches, current_counts, accept_counts, np.ones(member_count, dtype=bool)


def fuse_observations_by_grid(
    obs_locs: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    n1: int,
    n2: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Merge repeated observations in the same integer grid into one sparse row."""
    obs_locs = np.asarray(obs_locs, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=np.int32)

    if obs_locs.shape[0] == 0:
        empty_locs = np.empty((0, 2), dtype=float)
        empty_gamma = np.empty((0, gamma.shape[1] if gamma.ndim == 2 else 0), dtype=float)
        empty_omega = np.empty((0, omega.shape[1] if omega.ndim == 2 else 0), dtype=np.int32)
        return empty_locs, empty_gamma, empty_omega, {
            "raw_count": 0,
            "fused_count": 0,
            "compression_ratio": 1.0,
            "raw_per_fused": np.empty((0,), dtype=int),
        }

    k = gamma.shape[1]
    grid_xy = np.floor(obs_locs).astype(int)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, n1 - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, n2 - 1)

    fused: Dict[Tuple[int, int], Dict[str, np.ndarray | float | int]] = {}
    for idx, (gx, gy) in enumerate(grid_xy):
        key = (int(gx), int(gy))
        row = fused.setdefault(
            key,
            {
                "loc_sum": np.zeros(2, dtype=float),
                "loc_count": 0,
                "gamma_sum": np.zeros(k, dtype=float),
                "gamma_count": np.zeros(k, dtype=float),
                "omega": np.zeros(k, dtype=np.int32),
                "last_idx": -1,
                "raw_count": 0,
            },
        )
        row["loc_sum"] = np.asarray(row["loc_sum"], dtype=float) + obs_locs[idx]
        row["loc_count"] = int(row["loc_count"]) + 1
        row["raw_count"] = int(row["raw_count"]) + 1
        row["last_idx"] = idx

        observed = omega[idx] > 0
        if np.any(observed):
            row["omega"] = np.asarray(row["omega"], dtype=np.int32)
            row["gamma_sum"] = np.asarray(row["gamma_sum"], dtype=float)
            row["gamma_count"] = np.asarray(row["gamma_count"], dtype=float)
            row["omega"][observed] = 1
            row["gamma_sum"][observed] += gamma[idx, observed]
            row["gamma_count"][observed] += 1.0

    ordered_keys = sorted(fused.keys(), key=lambda key: int(fused[key]["last_idx"]))
    fused_locs = np.zeros((len(ordered_keys), 2), dtype=float)
    fused_gamma = np.zeros((len(ordered_keys), k), dtype=float)
    fused_omega = np.zeros((len(ordered_keys), k), dtype=np.int32)

    raw_counts: List[int] = []
    for row_idx, key in enumerate(ordered_keys):
        row = fused[key]
        loc_count = max(1, int(row["loc_count"]))
        gamma_count = np.asarray(row["gamma_count"], dtype=float)
        fused_locs[row_idx] = np.asarray(row["loc_sum"], dtype=float) / loc_count
        fused_omega[row_idx] = np.asarray(row["omega"], dtype=np.int32)
        valid = gamma_count > 0
        fused_gamma[row_idx, valid] = np.asarray(row["gamma_sum"], dtype=float)[valid] / gamma_count[valid]
        raw_counts.append(int(row["raw_count"]))

    return fused_locs, fused_gamma, fused_omega, {
        "raw_count": int(obs_locs.shape[0]),
        "fused_count": int(len(ordered_keys)),
        "compression_ratio": float(obs_locs.shape[0] / max(1, len(ordered_keys))),
        "raw_per_fused": np.asarray(raw_counts, dtype=int),
    }


def select_reconstruction_outer_iters(
    effective_count: int,
    warmstart: bool = False,
) -> int:
    """Use more outer iterations early and fewer later."""
    effective_count = int(max(1, effective_count))
    if warmstart:
        if effective_count < 12:
            return 4
        if effective_count < 24:
            return 3
        return 2
    if effective_count < 12:
        return 5
    if effective_count < 24:
        return 4
    if effective_count < 48:
        return 3
    return 3


def ensemble_reconstruct_maps(
    obs_locs: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    i_mask: np.ndarray,
    m_ens: int = 6,
    seed: int = 42,
    btd_max_iter: int = 6,
    mu: float = 1.2,
    nu: float = 1.5,
    kernel_bandwidth: float = 0.46,
    member_max_iter: Optional[int] = None,
    quality_weighted: bool = True,
    solver_backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoints: Optional[List[str]] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
    member_init_jitter_scale: float = DEFAULT_ENSEMBLE_INIT_JITTER_SCALE,
    return_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Fit DU-IIBTD ensemble members and return mean/variance maps."""
    n1, n2 = grid_size
    fused_obs_locs, fused_gamma, fused_omega, fusion_meta = fuse_observations_by_grid(
        obs_locs,
        gamma,
        omega,
        n1,
        n2,
    )
    total = fused_obs_locs.shape[0]
    if total == 0:
        k = gamma.shape[1]
        zeros = np.zeros((n1, n2, k), dtype=float)
        stack = zeros[np.newaxis, ...]
        info = {
            "member_models": [],
            "member_observation_counts": np.empty((0,), dtype=int),
            "member_last_incremental_accept_counts": np.empty((0,), dtype=int),
            "member_incremental_updated_mask": np.empty((0,), dtype=bool),
            "member_kernel_bandwidths": np.empty((0,), dtype=float),
            "member_nus": np.empty((0,), dtype=float),
            "member_du_iibtd_checkpoints": [],
            "weights": np.ones(1, dtype=float),
            "obs_nmse": np.array([], dtype=float),
            "fusion_meta": fusion_meta,
        }
        if return_info:
            return zeros, zeros, stack, info
        return zeros, zeros, stack

    if member_max_iter is None:
        member_max_iter = min(
            max(2, int(btd_max_iter)),
            select_reconstruction_outer_iters(total, warmstart=False),
        )

    member_models: List[object] = []
    member_observation_counts: List[int] = []
    active_member_kernel_bandwidths: List[float] = []
    active_member_nus: List[float] = []
    active_member_checkpoints: List[str] = []
    member_count = max(1, int(m_ens))
    checkpoint_paths = _normalize_checkpoint_paths(
        du_iibtd_checkpoints,
        backend=solver_backend,
    )
    member_checkpoints = [
        _resolve_member_checkpoint(checkpoint_paths, m, backend=solver_backend)
        for m in range(member_count)
    ]
    member_kernel_bandwidths = np.asarray(
        [
            _checkpoint_config_float(path, "kernel_bandwidth", float(kernel_bandwidth))
            for path in member_checkpoints
        ],
        dtype=float,
    )
    member_nus = np.asarray(
        [
            _checkpoint_config_float(path, "nu", float(nu))
            for path in member_checkpoints
        ],
        dtype=float,
    )
    for m in range(member_count):
        idx = np.arange(total, dtype=int)
        try:
            model = fit_reconstruction_model(
                fused_obs_locs[idx],
                fused_gamma[idx],
                fused_omega[idx],
                n_sources=n_sources,
                grid_size=grid_size,
                grid_points=grid_points,
                bounds=bounds,
                i_mask=i_mask,
                prev_model=None,
                warmstart=False,
                max_iter=member_max_iter,
                mu=mu,
                nu=member_nus[m],
                kernel_bandwidth=member_kernel_bandwidths[m],
                solver_backend=solver_backend,
                solver_device=solver_device,
                du_iibtd_checkpoint_path=member_checkpoints[m],
                du_iibtd_min_sensors_for_update=du_iibtd_min_sensors_for_update,
                du_iibtd_update_batch_size=du_iibtd_update_batch_size,
                init_seed=seed + 20_000 + m,
                init_jitter_scale=member_init_jitter_scale,
            )
        except Exception as exc:
            raise RuntimeError(
                "DU-IIBTD ensemble member initialization failed "
                f"for member {m} using checkpoint {member_checkpoints[m]!r}."
            ) from exc
        member_models.append(model)
        member_observation_counts.append(int(idx.size))
        active_member_kernel_bandwidths.append(float(member_kernel_bandwidths[m]))
        active_member_nus.append(float(member_nus[m]))
        active_member_checkpoints.append(str(member_checkpoints[m]))

    if not member_models:
        raise RuntimeError("All ensemble member fits failed during full refresh.")

    mean_map, var_map, stack, info = _summarize_ensemble_models(
        member_models,
        fused_obs_locs=fused_obs_locs,
        fused_gamma=fused_gamma,
        fused_omega=fused_omega,
        quality_weighted=quality_weighted,
        member_observation_counts=np.asarray(member_observation_counts, dtype=int),
        member_last_incremental_accept_counts=np.zeros(len(member_models), dtype=int),
        member_incremental_updated_mask=np.zeros(len(member_models), dtype=bool),
        member_kernel_bandwidths=np.asarray(active_member_kernel_bandwidths, dtype=float),
        member_nus=np.asarray(active_member_nus, dtype=float),
        member_du_iibtd_checkpoints=active_member_checkpoints,
        fusion_meta=fusion_meta,
    )
    if return_info:
        return mean_map, var_map, stack, info
    return mean_map, var_map, stack


def _incremental_update_reconstruction_model(
    model,
    *,
    new_obs_locs: np.ndarray,
    new_gamma: np.ndarray,
    new_omega: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    i_mask: np.ndarray,
    n_outer_iter: int = 2,
    max_svt_iter: int = 20,
    mu: float = 1.2,
    nu: float = 1.5,
    kernel_bandwidth: float = 0.46,
    solver_backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoint_path: Optional[str] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
):
    """Incrementally update one reconstruction model with newly fused observations."""
    new_obs_locs = np.asarray(new_obs_locs, dtype=float)
    new_gamma = np.asarray(new_gamma, dtype=float)
    new_omega = np.asarray(new_omega, dtype=np.int32)
    if new_obs_locs.shape[0] == 0:
        return model

    if model is None:
        raise RuntimeError("DU-IIBTD incremental refresh received a missing ensemble member.")

    if not hasattr(model, "fit_incremental"):
        raise TypeError("DU-IIBTD ensemble members must expose fit_incremental().")
    model.fit_incremental(
        new_obs_locs,
        new_gamma,
        new_omega,
        grid_coords=grid_points,
        bounds=bounds,
        I_mask=i_mask,
        n_outer_iter=max(1, int(n_outer_iter)),
        max_svt_iter=max(1, int(max_svt_iter)),
        debugFlag=False,
    )
    return model


def incremental_refresh_ensemble_models(
    *,
    member_models,
    member_observation_counts,
    new_obs_locs: np.ndarray,
    new_gamma: np.ndarray,
    new_omega: np.ndarray,
    fused_obs_locs: np.ndarray,
    fused_gamma: np.ndarray,
    fused_omega: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    i_mask: np.ndarray,
    n_outer_iter: int = 2,
    max_svt_iter: int = 20,
    quality_weighted: bool = True,
    mu: float = 1.2,
    nu: float = 1.5,
    kernel_bandwidth: float = 0.46,
    solver_backend: str = "du_iibtd",
    solver_device: str = "auto",
    du_iibtd_checkpoints: Optional[List[str]] = None,
    du_iibtd_min_sensors_for_update: Optional[int] = None,
    du_iibtd_update_batch_size: Optional[int] = None,
    member_kernel_bandwidths: Optional[np.ndarray] = None,
    member_nus: Optional[np.ndarray] = None,
    fusion_meta: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Incrementally update member solvers, then recompute ensemble mean/variance."""
    member_models = list(member_models or [])
    if not member_models:
        raise RuntimeError("No ensemble members are available for incremental refresh.")
    (
        member_batches,
        updated_observation_counts,
        member_accept_counts,
        member_updated_mask,
    ) = _share_incremental_rows_across_members(
        new_obs_locs,
        new_gamma,
        new_omega,
        member_count=len(member_models),
        member_observation_counts=member_observation_counts,
    )

    checkpoint_paths = _normalize_checkpoint_paths(
        du_iibtd_checkpoints,
        backend=solver_backend,
    )
    member_checkpoints = [
        str(
            getattr(
                model,
                "checkpoint_path",
                _resolve_member_checkpoint(
                    checkpoint_paths,
                    idx,
                    backend=solver_backend,
                ),
            )
        )
        for idx, model in enumerate(member_models)
    ]
    if member_kernel_bandwidths is None:
        member_kernel_bandwidths = np.asarray(
            [
                float(getattr(model, "h", _checkpoint_config_float(member_checkpoints[idx], "kernel_bandwidth", kernel_bandwidth)))
                for idx, model in enumerate(member_models)
            ],
            dtype=float,
        )
    else:
        member_kernel_bandwidths = np.asarray(member_kernel_bandwidths, dtype=float)
    if member_nus is None:
        member_nus = np.asarray(
            [
                float(
                    getattr(
                        model,
                        "nu",
                        _checkpoint_config_float(member_checkpoints[idx], "nu", float(nu)),
                    )
                )
                for idx, model in enumerate(member_models)
            ],
            dtype=float,
        )
    else:
        member_nus = np.asarray(member_nus, dtype=float)

    updated_member_models: List[object] = []
    for member_idx, (model, batch) in enumerate(zip(member_models, member_batches)):
        if batch is None:
            updated_model = model
        else:
            try:
                updated_model = _incremental_update_reconstruction_model(
                    model,
                    new_obs_locs=batch["obs_locs"],
                    new_gamma=batch["gamma"],
                    new_omega=batch["omega"],
                    n_sources=n_sources,
                    grid_size=grid_size,
                    grid_points=grid_points,
                    bounds=bounds,
                    i_mask=i_mask,
                    n_outer_iter=n_outer_iter,
                    max_svt_iter=max_svt_iter,
                    mu=mu,
                    nu=member_nus[member_idx],
                    kernel_bandwidth=member_kernel_bandwidths[member_idx],
                    solver_backend=solver_backend,
                    solver_device=solver_device,
                    du_iibtd_checkpoint_path=member_checkpoints[member_idx],
                    du_iibtd_min_sensors_for_update=du_iibtd_min_sensors_for_update,
                    du_iibtd_update_batch_size=du_iibtd_update_batch_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    "DU-IIBTD ensemble incremental refresh failed "
                    f"for member {member_idx} using checkpoint {member_checkpoints[member_idx]!r}."
                ) from exc
        updated_member_models.append(updated_model)

    return _summarize_ensemble_models(
        updated_member_models,
        fused_obs_locs=fused_obs_locs,
        fused_gamma=fused_gamma,
        fused_omega=fused_omega,
        quality_weighted=quality_weighted,
        member_observation_counts=updated_observation_counts,
        member_last_incremental_accept_counts=member_accept_counts,
        member_incremental_updated_mask=member_updated_mask,
        member_kernel_bandwidths=member_kernel_bandwidths,
        member_nus=member_nus,
        member_du_iibtd_checkpoints=member_checkpoints,
        fusion_meta=dict(fusion_meta or {}),
    )


def build_acquisition_space(
    var_map: np.ndarray,
    lambda_u: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build spatial acquisition score map using uncertainty only."""
    uncertainty_space = np.mean(var_map, axis=2)
    acquisition_space = lambda_u * uncertainty_space
    return acquisition_space, {"uncertainty_space": uncertainty_space}


def select_top_k_grid_candidates(
    acquisition_space: np.ndarray,
    var_map: np.ndarray,
    sampled_mask: np.ndarray,
    action_visit: np.ndarray,
    top_k: int,
    beta_f: float = 0.2,
    redundancy_length: float = 0.0,
    candidate_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Select top-k grid candidates with best spatial+frequency score."""
    sampled_mask = np.asarray(sampled_mask, dtype=bool)
    spatial_unvisited = ~np.any(sampled_mask, axis=2)
    if np.any(spatial_unvisited):
        spatial_available = spatial_unvisited
    else:
        spatial_available = ~np.all(sampled_mask, axis=2)
    if candidate_mask is not None:
        spatial_available = np.logical_and(spatial_available, np.asarray(candidate_mask, dtype=bool))
    grid_indices = np.argwhere(spatial_available)
    if grid_indices.size == 0:
        if candidate_mask is not None:
            grid_indices = np.argwhere(np.asarray(candidate_mask, dtype=bool))
        if grid_indices.size == 0:
            grid_indices = np.argwhere(np.ones(acquisition_space.shape, dtype=bool))

    candidates: List[Dict[str, float]] = []
    for gx, gy in grid_indices.tolist():
        gx = int(gx)
        gy = int(gy)
        spatial_score = float(acquisition_space[gx, gy])

        freq_unc = var_map[gx, gy, :].astype(float)
        freq_penalty = action_visit[gx, gy, :].astype(float)
        if np.max(freq_penalty) > 0:
            freq_penalty = freq_penalty / np.max(freq_penalty)
        freq_score = freq_unc - float(beta_f) * freq_penalty

        unsampled_bands = ~sampled_mask[gx, gy, :]
        if np.any(unsampled_bands):
            masked_freq_score = freq_score.copy()
            masked_freq_score[~unsampled_bands] = -np.inf
            center_freq = int(np.argmax(masked_freq_score))
            best_freq_score = float(masked_freq_score[center_freq])
        else:
            center_freq = int(np.argmax(freq_score))
            best_freq_score = float(freq_score[center_freq])

        total_score = spatial_score + best_freq_score
        candidates.append(
            {
                "x": float(gx),
                "y": float(gy),
                "gx": gx,
                "gy": gy,
                "center_freq": center_freq,
                "score": total_score,
                "spatial_score": spatial_score,
                "freq_score": best_freq_score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidate_k = int(max(1, top_k))
    if candidate_k <= 1 or float(redundancy_length) <= 0.0 or len(candidates) <= 1:
        return candidates[:candidate_k]

    redundancy_length = float(redundancy_length)
    selected: List[Dict[str, float]] = [candidates[0]]
    remaining = list(candidates[1:])

    while len(selected) < candidate_k and remaining:
        best_idx = 0
        best_key = None
        for idx, candidate in enumerate(remaining):
            candidate_xy = np.array([candidate["gx"], candidate["gy"]], dtype=float)
            min_dist = min(
                float(
                    np.linalg.norm(
                        candidate_xy - np.array([chosen["gx"], chosen["gy"]], dtype=float)
                    )
                )
                for chosen in selected
            )
            if min_dist >= redundancy_length:
                key = (2, float(candidate["score"]), min_dist)
            else:
                key = (1, min_dist, float(candidate["score"]))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return selected[:candidate_k]

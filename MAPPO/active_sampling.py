"""
Active-sampling helpers adapted from Test/iibtdEnsembleTest.py.

This module keeps environment.py focused on multi-agent orchestration.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from IIBTD.IIBTD_Optimized import II_BTD_Optimized
from IIBTD.IIBTD_Opt_GPU import II_BTD_Opt_GPU


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


def quantize_to_nbit(
    values: np.ndarray,
    n_bits: int = 8,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Uniform N-bit quantization with de-quantized float output."""
    arr = np.asarray(values, dtype=float)
    n_bits = max(1, int(n_bits))
    levels = int(2 ** n_bits)
    if arr.size == 0:
        return arr.copy(), {
            "min": 0.0,
            "max": 0.0,
            "levels": float(levels),
            "n_bits": float(n_bits),
        }

    v_min = float(np.min(arr)) if value_min is None else float(value_min)
    v_max = float(np.max(arr)) if value_max is None else float(value_max)
    if v_max <= v_min + 1e-12:
        return np.full_like(arr, v_min, dtype=float), {
            "min": v_min,
            "max": v_max,
            "levels": float(levels),
            "n_bits": float(n_bits),
        }

    norm = (arr - v_min) / (v_max - v_min)
    quantized = np.round(norm * float(levels - 1))
    dq = (quantized / float(levels - 1)) * (v_max - v_min) + v_min
    return dq, {
        "min": v_min,
        "max": v_max,
        "levels": float(levels),
        "n_bits": float(n_bits),
    }


def normalize_score_map(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a score map to [0, 1] while keeping NaNs/inf safe."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out

    finite_vals = arr[finite_mask]
    v_min = float(np.min(finite_vals))
    v_max = float(np.max(finite_vals))
    if v_max <= v_min + float(eps):
        out[finite_mask] = 0.0
        return out

    out[finite_mask] = (arr[finite_mask] - v_min) / (v_max - v_min)
    return out


def select_quantization_bits_from_uncertainty(
    uncertainty_norm: Optional[float],
    adaptive_quantization_bits: bool = False,
    quantization_bits: int = 8,
    high_quantization_bits: int = 8,
    low_quantization_bits: int = 4,
    uncertainty_quantization_threshold: float = 0.5,
    phase: str = "planned_path",
) -> Tuple[int, str]:
    """Choose the quantization width from normalized uncertainty."""
    if not adaptive_quantization_bits:
        return int(max(1, int(quantization_bits))), "fixed"
    if phase == "warmup" or uncertainty_norm is None or not np.isfinite(float(uncertainty_norm)):
        return int(max(1, int(high_quantization_bits))), "warmup_high"
    if float(uncertainty_norm) >= float(uncertainty_quantization_threshold):
        return int(max(1, int(high_quantization_bits))), "high"
    return int(max(1, int(low_quantization_bits))), "low"


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


def adaptive_keep_ratio(M_total, early_ratio=0.95, late_ratio=0.85, switch_M=80):
    """Early stage keeps more observations; later stage allows stronger perturbation."""
    if M_total <= switch_M:
        return early_ratio
    alpha = np.clip((M_total - switch_M) / max(1, switch_M), 0.0, 1.0)
    return (1.0 - alpha) * early_ratio + alpha * late_ratio


def _sample_ensemble_indices(
    total_samples: int,
    keep_ratio: float,
    keep_recent: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if total_samples <= 1:
        return np.arange(total_samples, dtype=int)

    keep_count = int(np.clip(round(keep_ratio * total_samples), 1, total_samples))
    keep_recent = int(np.clip(keep_recent, 0, total_samples))

    recent_start = max(0, total_samples - keep_recent)
    recent_idx = np.arange(recent_start, total_samples, dtype=int)
    if recent_idx.size >= keep_count:
        return recent_idx[-keep_count:]

    pool = np.arange(0, recent_start, dtype=int)
    n_rand = min(keep_count - recent_idx.size, pool.size)
    rand_idx = (
        rng.choice(pool, size=n_rand, replace=False)
        if n_rand > 0
        else np.empty((0,), dtype=int)
    )

    idx = np.concatenate([rand_idx, recent_idx])
    if idx.size < keep_count:
        missing = keep_count - idx.size
        remain = np.setdiff1d(np.arange(total_samples, dtype=int), idx, assume_unique=False)
        if remain.size > 0:
            fill = rng.choice(remain, size=min(missing, remain.size), replace=False)
            idx = np.concatenate([idx, fill])
    return np.sort(np.unique(idx))


def _copy_model_state(dst_model, src_model) -> None:
    """Copy warm-start state between solvers with the same dimensions."""
    if hasattr(dst_model, "load_state_from"):
        dst_model.load_state_from(src_model)
        return
    for name in ("Theta", "Phi", "Sr", "H_hat"):
        if hasattr(src_model, name):
            setattr(dst_model, name, np.array(getattr(src_model, name), copy=True))


def _resolve_iibtd_backend(
    backend: str,
    n_sources: int,
    solver_device: Optional[str] = None,
) -> str:
    backend = str(backend or "cpu").strip().lower()
    if backend == "gpu":
        if torch is not None and torch.cuda.is_available():
            return "gpu"
        return "cpu"
    if backend == "auto":
        device_str = str(solver_device or "auto").strip().lower()
        if (
            torch is not None
            and torch.cuda.is_available()
            and int(n_sources) == 1
            and device_str != "cpu"
        ):
            return "gpu"
        return "cpu"
    return "cpu"


def _resolve_iibtd_device(solver_device: Optional[str] = None) -> str:
    device_str = str(solver_device or "auto").strip()
    if not device_str or device_str.lower() == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return device_str


def _make_reconstruction_solver(
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int,
    kernel_bandwidth: float,
    warmstart: bool,
    backend: str = "cpu",
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
):
    backend_eff = _resolve_iibtd_backend(
        backend=backend,
        n_sources=n_sources,
        solver_device=solver_device,
    )
    common_kwargs = dict(
        n_sources=n_sources,
        grid_size=grid_size,
        mu=1.2,
        nu=1.5,
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=kernel_bandwidth,
        warmstart=bool(warmstart),
    )
    if backend_eff == "gpu":
        return II_BTD_Opt_GPU(
            **common_kwargs,
            device=_resolve_iibtd_device(solver_device),
            phi_solver=str(gpu_phi_solver).strip().lower() or "scipy",
        )
    return II_BTD_Optimized(**common_kwargs)


def init_reconstruction_solver(
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    K: int,
    i_mask: np.ndarray,
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int = 6,
    kernel_bandwidth: float = 0.46,
    warmstart: bool = False,
    solver_backend: str = "cpu",
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
):
    """Create and initialize one II-BTD solver instance."""
    model = _make_reconstruction_solver(
        n_sources=n_sources,
        grid_size=grid_size,
        max_iter=max_iter,
        kernel_bandwidth=kernel_bandwidth,
        warmstart=warmstart,
        backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
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
    prev_model: Optional[II_BTD_Optimized] = None,
    warmstart: bool = False,
    max_iter: int = 6,
    kernel_bandwidth: float = 0.46,
    solver_backend: str = "cpu",
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
):
    """Fit one reconstruction model, optionally warm-started from prev_model."""
    model = init_reconstruction_solver(
        grid_points=grid_points,
        bounds=bounds,
        K=gamma.shape[1],
        i_mask=i_mask,
        n_sources=n_sources,
        grid_size=grid_size,
        max_iter=max_iter,
        kernel_bandwidth=kernel_bandwidth,
        warmstart=warmstart,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )
    if warmstart and prev_model is not None:
        _copy_model_state(model, prev_model)
    model.fit_2(
        obs_locs,
        gamma,
        omega,
        grid_points,
        bounds,
        I_mask=i_mask,
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
    keep_ratio: float = 0.85,
    keep_recent: int = 2,
    seed: int = 42,
    btd_max_iter: int = 6,
    kernel_bandwidth: float = 0.46,
    base_model=None,
    member_max_iter: Optional[int] = None,
    quality_weighted: bool = True,
    solver_backend: str = "cpu",
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    return_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Run ensemble II-BTD and return mean/variance maps."""
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
            "base_model": base_model,
            "representative_model": base_model,
            "weights": np.ones(1, dtype=float),
            "obs_nmse": np.array([], dtype=float),
            "fusion_meta": fusion_meta,
        }
        if return_info:
            return zeros, zeros, stack, info
        return zeros, zeros, stack

    base_fit_iters = min(
        max(4, int(btd_max_iter)),
        select_reconstruction_outer_iters(total, warmstart=False),
    )
    if member_max_iter is None:
        member_max_iter = min(
            max(2, int(btd_max_iter)),
            select_reconstruction_outer_iters(total, warmstart=True),
        )

    if base_model is None:
        try:
            base_model = fit_reconstruction_model(
                fused_obs_locs,
                fused_gamma,
                fused_omega,
                n_sources=n_sources,
                grid_size=grid_size,
                grid_points=grid_points,
                bounds=bounds,
                i_mask=i_mask,
                prev_model=None,
                warmstart=False,
                max_iter=base_fit_iters,
                kernel_bandwidth=kernel_bandwidth,
                solver_backend=solver_backend,
                solver_device=solver_device,
                gpu_phi_solver=gpu_phi_solver,
            )
        except Exception as exc:
            warnings.warn(
                "ensemble_reconstruct_maps: base_model fit failed; "
                f"falling back to cold-start members ({exc!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            base_model = None

    maps: List[np.ndarray] = []
    obs_nmse_scores: List[float] = []
    representative_model = None
    representative_obs_nmse = np.inf
    for m in range(max(1, int(m_ens))):
        rng = np.random.default_rng(seed + 1000 + m)
        idx = _sample_ensemble_indices(total, keep_ratio, keep_recent, rng)

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
                prev_model=base_model,
                warmstart=base_model is not None,
                max_iter=member_max_iter,
                kernel_bandwidth=kernel_bandwidth,
                solver_backend=solver_backend,
                solver_device=solver_device,
                gpu_phi_solver=gpu_phi_solver,
            )
            map_m = model.get_current_map()
            maps.append(map_m)
            obs_nmse_m = _observation_nmse(
                map_m,
                fused_obs_locs,
                fused_gamma,
                fused_omega,
                n1,
                n2,
            )
            obs_nmse_scores.append(obs_nmse_m)
            if obs_nmse_m < representative_obs_nmse:
                representative_obs_nmse = float(obs_nmse_m)
                representative_model = model
        except Exception:
            continue

    if not maps:
        k = gamma.shape[1]
        if base_model is not None:
            base_map = base_model.get_current_map()
            zeros = np.zeros_like(base_map)
            stack = base_map[np.newaxis, ...]
            info = {
                "base_model": base_model,
                "representative_model": base_model,
                "weights": np.ones(1, dtype=float),
                "obs_nmse": np.array([], dtype=float),
                "fusion_meta": fusion_meta,
            }
            if return_info:
                return base_map, zeros, stack, info
            return base_map, zeros, stack
        zeros = np.zeros((n1, n2, k), dtype=float)
        stack = zeros[np.newaxis, ...]
        info = {
            "base_model": None,
            "representative_model": None,
            "weights": np.ones(1, dtype=float),
            "obs_nmse": np.array([], dtype=float),
            "fusion_meta": fusion_meta,
        }
        if return_info:
            return zeros, zeros, stack, info
        return zeros, zeros, stack

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
    if return_info:
        return mean_map, var_map, stack, {
            "base_model": base_model,
            "representative_model": representative_model if representative_model is not None else base_model,
            "weights": np.asarray(weights, dtype=float),
            "obs_nmse": np.asarray(obs_nmse_scores, dtype=float),
            "fusion_meta": fusion_meta,
        }
    return mean_map, var_map, stack


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
    return candidates[: int(max(1, top_k))]

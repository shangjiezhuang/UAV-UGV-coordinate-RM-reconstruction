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

try:
    from IIBTD.IIBTD_Opt_GPU import II_BTD_Opt_GPU
except ModuleNotFoundError:  # pragma: no cover - optional GPU backend
    II_BTD_Opt_GPU = None

try:
    from DU_IIBTD_res_Sr import DU_IIBTDSolverAdapter as DU_IIBTD_SR_SolverAdapter
except ModuleNotFoundError:  # pragma: no cover - optional torch/checkpoint backend
    DU_IIBTD_SR_SolverAdapter = None

try:
    from DU_IIBTD_res_Sr_learn_nu import (
        DU_IIBTDSolverAdapter as DU_IIBTD_SRLearnNu_SolverAdapter,
    )
except ModuleNotFoundError:  # pragma: no cover - optional torch/checkpoint backend
    DU_IIBTD_SRLearnNu_SolverAdapter = None

from du_iibtd_based.du_iibtd_learn_nu import (
    checkpoint_config_float as _du_checkpoint_config_float,
    checkpoint_config_int_or_none as _du_checkpoint_config_int_or_none,
)


SHARED_MEMBER_OBSERVATION_MODE = "shared_all"
SHARED_MEMBER_KERNEL_BANDWIDTH_MODE = "base_pm_delta"
SHARED_MEMBER_KERNEL_BANDWIDTH_DELTA = 0.17
SHARED_MEMBER_INIT_JITTER_SCALE = 1e-2

DU_IIBTD_SR_BACKEND = "du_iibtd_sr"
DU_IIBTD_SR_LEARN_NU_BACKEND = "du_iibtd_sr_learn_nu"
_DU_IIBTD_BACKENDS = {DU_IIBTD_SR_BACKEND, DU_IIBTD_SR_LEARN_NU_BACKEND}
_DU_IIBTD_SR_BACKEND_ALIASES = {
    "du_iibtd_sr",
    "du_iibtd_res_sr",
    "du-iibtd-sr",
    "du-iibtd-res-sr",
}
_DU_IIBTD_SR_LEARN_NU_BACKEND_ALIASES = {
    "du_iibtd_sr_learn_nu",
    "du_iibtd_res_sr_learn_nu",
    "du-iibtd-sr-learn-nu",
    "du-iibtd-res-sr-learn-nu",
}


def _resolve_du_checkpoint_path(checkpoint_path: str | os.PathLike) -> Path:
    path_text = str(checkpoint_path or "").strip()
    if not path_text:
        raise ValueError("DU-IIBTD backend requires a checkpoint path.")
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(_PROJECT_ROOT) / path
    return path


def _split_du_checkpoint_paths(checkpoint_path: str | os.PathLike) -> List[str]:
    paths = [part.strip() for part in str(checkpoint_path or "").split(",") if part.strip()]
    if not paths:
        raise ValueError("DU-IIBTD backend requires at least one checkpoint path.")
    return paths


def _du_checkpoint_path_for_member(checkpoint_path: str | os.PathLike, member_idx: int) -> str:
    paths = _split_du_checkpoint_paths(checkpoint_path)
    return paths[int(member_idx) % len(paths)]


def _du_update_batch_size(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


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


def _build_member_kernel_bandwidths(
    member_count: int,
    *,
    base_kernel_bandwidth: float,
    delta: float = SHARED_MEMBER_KERNEL_BANDWIDTH_DELTA,
) -> np.ndarray:
    """Build deterministic shared-member kernel bandwidths: base, base-delta, base+delta, ..."""
    member_count = max(1, int(member_count))
    base = float(base_kernel_bandwidth)
    delta = abs(float(delta))
    bandwidths = np.empty(member_count, dtype=float)
    bandwidths[0] = base
    for member_idx in range(1, member_count):
        sign = -1.0 if member_idx % 2 == 1 else 1.0
        bandwidths[member_idx] = max(1e-6, base + sign * delta)
    return bandwidths


def _jitter_model_initial_state(
    model,
    rng: np.random.Generator,
    scale: float = 0.0,
) -> None:
    """Apply a small member-specific perturbation to initialized solver state."""
    scale = float(scale)
    if scale <= 0.0:
        return

    if torch is not None:
        for name in ("_Theta_t", "_Phi_t", "_Sr_t", "_H_hat_t"):
            value = getattr(model, name, None)
            if value is None or not torch.is_tensor(value):
                continue
            if not torch.is_floating_point(value):
                continue
            noise = torch.randn_like(value) * scale
            jittered = value * (1.0 + noise)
            if value.numel() > 0 and bool(torch.all(value >= 0.0).item()):
                jittered = torch.clamp(jittered, min=1e-12)
            setattr(model, name, jittered)

    for name in ("Theta", "Phi", "Sr", "H_hat"):
        value = getattr(model, name, None)
        if value is None:
            continue
        if torch is not None and torch.is_tensor(value):
            if not torch.is_floating_point(value):
                continue
            noise = torch.randn_like(value) * scale
            jittered = value * (1.0 + noise)
            if value.numel() > 0 and bool(torch.all(value >= 0.0).item()):
                jittered = torch.clamp(jittered, min=1e-12)
            setattr(model, name, jittered)
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        jittered = arr * (1.0 + rng.normal(0.0, scale, size=arr.shape))
        if np.all(arr >= 0.0):
            jittered = np.maximum(jittered, 1e-12)
        setattr(model, name, jittered)


def _initialize_model_state_for_jitter(
    model,
    *,
    grid_points: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    i_mask: np.ndarray,
    num_bands: int,
    init_seed: Optional[int],
    init_jitter_scale: float,
) -> None:
    """Initialize a solver once so member-specific jitter survives fit_2."""
    if not hasattr(model, "init_sequential"):
        return

    init_seed_int = None if init_seed is None else int(init_seed)
    np_state = None
    torch_state = None
    cuda_states = None
    if init_seed_int is not None:
        np_state = np.random.get_state()
        np.random.seed(init_seed_int % ((2 ** 32) - 1))
        if torch is not None:
            try:
                torch_state = torch.random.get_rng_state()
                if torch.cuda.is_available():
                    cuda_states = torch.cuda.get_rng_state()
                torch.manual_seed(init_seed_int)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(init_seed_int)
            except Exception:
                torch_state = None
                cuda_states = None

    try:
        model.init_sequential(
            grid_points,
            bounds,
            K=int(num_bands),
            I_mask=i_mask,
        )
        if hasattr(model, "warmstart"):
            model.warmstart = True
        rng_seed = init_seed_int + 7919 if init_seed_int is not None else None
        init_rng = np.random.default_rng(rng_seed)
        _jitter_model_initial_state(model, init_rng, scale=init_jitter_scale)
        sync_fn = getattr(model, "_sync_public_state", None)
        if callable(sync_fn):
            try:
                sync_fn(include_theta=True)
            except TypeError:
                sync_fn()
    finally:
        if init_seed_int is not None and np_state is not None:
            np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            try:
                torch.random.set_rng_state(torch_state)
                if cuda_states is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_states)
            except Exception:
                pass


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


def _resolve_iibtd_backend(
    backend: str,
    n_sources: int,
    solver_device: Optional[str] = None,
) -> str:
    del solver_device  # Backend selection is explicit; device only chooses where it runs.
    if int(n_sources) != 1:
        raise ValueError("DU-IIBTD backends currently require n_sources=1.")
    backend_key = str(backend or DU_IIBTD_SR_BACKEND).strip().lower()
    if backend_key in _DU_IIBTD_SR_BACKEND_ALIASES:
        return DU_IIBTD_SR_BACKEND
    if backend_key in _DU_IIBTD_SR_LEARN_NU_BACKEND_ALIASES:
        return DU_IIBTD_SR_LEARN_NU_BACKEND
    if backend_key == "gpu":
        if II_BTD_Opt_GPU is None:
            raise RuntimeError("GPU II-BTD backend is unavailable; IIBTD.IIBTD_Opt_GPU is not importable.")
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("GPU II-BTD backend requires CUDA; use a DU-IIBTD backend for CPU-device runs.")
        return "gpu"
    raise ValueError(
        "planner.iibtd_backend must be one of "
        "du_iibtd_sr/du_iibtd_sr_learn_nu/gpu; "
        f"got {backend_key!r}. The old cpu/auto/du_iibtd backends are disabled."
    )


def _du_solver_adapter_for_backend(backend_eff: str):
    if backend_eff == DU_IIBTD_SR_BACKEND:
        return DU_IIBTD_SR_SolverAdapter, "DU_IIBTD_res_Sr"
    if backend_eff == DU_IIBTD_SR_LEARN_NU_BACKEND:
        return DU_IIBTD_SRLearnNu_SolverAdapter, "DU_IIBTD_res_Sr_learn_nu"
    raise ValueError(f"Unsupported DU-IIBTD backend: {backend_eff!r}")


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
    backend: str = DU_IIBTD_SR_BACKEND,
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
):
    backend_eff = _resolve_iibtd_backend(
        backend=backend,
        n_sources=n_sources,
        solver_device=solver_device,
    )
    common_kwargs = dict(
        n_sources=n_sources,
        grid_size=grid_size,
        mu=float(mu),
        nu=float(nu),
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=kernel_bandwidth,
        warmstart=bool(warmstart),
    )
    if backend_eff in _DU_IIBTD_BACKENDS:
        adapter_cls, package_name = _du_solver_adapter_for_backend(backend_eff)
        if adapter_cls is None:
            raise RuntimeError(
                f"{backend_eff} backend is unavailable; ensure torch and "
                f"{package_name} are importable."
            )
        checkpoint_path = str(_resolve_du_checkpoint_path(du_checkpoint_path))
        resolved_kernel_bandwidth = _du_checkpoint_config_float(
            checkpoint_path,
            "kernel_bandwidth",
            float(kernel_bandwidth),
        )
        resolved_nu = _du_checkpoint_config_float(
            checkpoint_path,
            "nu",
            float(nu),
        )
        resolved_min_sensors = _du_checkpoint_config_int_or_none(
            checkpoint_path,
            "min_sensors_for_update",
            int(du_min_sensors_for_update) if int(du_min_sensors_for_update) > 0 else None,
        )
        resolved_update_batch_size = _du_checkpoint_config_int_or_none(
            checkpoint_path,
            "update_batch_size",
            _du_update_batch_size(du_update_batch_size),
        )
        return adapter_cls(
            n_sources=n_sources,
            grid_size=grid_size,
            mu=float(mu),
            nu=float(resolved_nu),
            max_iter=max(1, int(max_iter)),
            kernel_bandwidth=float(resolved_kernel_bandwidth),
            warmstart=bool(warmstart),
            checkpoint_path=checkpoint_path,
            device=_resolve_iibtd_device(solver_device),
            min_sensors_for_update=resolved_min_sensors,
            update_batch_size=resolved_update_batch_size,
        )
    if backend_eff == "gpu":
        return II_BTD_Opt_GPU(
            **common_kwargs,
            device=_resolve_iibtd_device(solver_device),
            phi_solver=str(gpu_phi_solver).strip().lower() or "scipy",
        )
    raise ValueError(f"Unsupported II-BTD backend: {backend_eff!r}")


def _make_configured_reconstruction_solver(
    n_sources: int,
    grid_size: Tuple[int, int],
    max_iter: int,
    mu: float,
    nu: float,
    kernel_bandwidth: float,
    warmstart: bool,
    solver_backend: str,
    solver_device: str,
    gpu_phi_solver: str,
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
):
    return _make_reconstruction_solver(
        n_sources=n_sources,
        grid_size=grid_size,
        max_iter=max_iter,
        mu=mu,
        nu=nu,
        kernel_bandwidth=kernel_bandwidth,
        warmstart=warmstart,
        backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
        du_checkpoint_path=du_checkpoint_path,
        du_min_sensors_for_update=du_min_sensors_for_update,
        du_update_batch_size=du_update_batch_size,
    )


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
    solver_backend: str = DU_IIBTD_SR_BACKEND,
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
    init_seed: Optional[int] = None,
    init_jitter_scale: float = 0.0,
):
    """Fit one reconstruction model, optionally warm-started from prev_model."""
    model = _make_configured_reconstruction_solver(
        n_sources, grid_size, max_iter, mu, nu, kernel_bandwidth,
        warmstart, solver_backend, solver_device, gpu_phi_solver,
        du_checkpoint_path=du_checkpoint_path,
        du_min_sensors_for_update=du_min_sensors_for_update,
        du_update_batch_size=du_update_batch_size,
    )
    if warmstart and prev_model is not None:
        _copy_model_state(model, prev_model)
    elif float(init_jitter_scale) > 0.0:
        _initialize_model_state_for_jitter(
            model,
            grid_points=grid_points,
            bounds=bounds,
            i_mask=i_mask,
            num_bands=np.asarray(gamma).shape[1],
            init_seed=init_seed,
            init_jitter_scale=init_jitter_scale,
        )
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
            else np.full(len(member_models), np.nan, dtype=float)
        ),
        "member_du_checkpoint_paths": [
            str(getattr(model, "checkpoint_path", "")) for model in member_models
        ],
        "ensemble_observation_mode": SHARED_MEMBER_OBSERVATION_MODE,
        "ensemble_kernel_bandwidth_mode": SHARED_MEMBER_KERNEL_BANDWIDTH_MODE,
        "weights": np.asarray(weights, dtype=float),
        "obs_nmse": np.asarray(obs_nmse_scores, dtype=float),
        "fusion_meta": dict(fusion_meta or {}),
    }
    return mean_map, var_map, stack, info


def _build_shared_incremental_batches(
    new_obs_locs: np.ndarray,
    new_gamma: np.ndarray,
    new_omega: np.ndarray,
    *,
    member_count: int,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """Give every shared-member ensemble solver the same newly fused rows."""
    new_obs_locs = np.asarray(new_obs_locs, dtype=float)
    new_gamma = np.asarray(new_gamma, dtype=float)
    new_omega = np.asarray(new_omega, dtype=np.int32)
    member_count = int(member_count)
    row_count = int(new_obs_locs.shape[0])
    if row_count == 0:
        return [None] * member_count

    return [
        {
            "obs_locs": new_obs_locs.copy(),
            "gamma": new_gamma.copy(),
            "omega": new_omega.copy(),
        }
        for _ in range(member_count)
    ]


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
    solver_backend: str = DU_IIBTD_SR_BACKEND,
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
    return_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Fit shared-observation ensemble members and return mean/variance maps."""
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
            "member_du_checkpoint_paths": [],
            "ensemble_observation_mode": SHARED_MEMBER_OBSERVATION_MODE,
            "ensemble_kernel_bandwidth_mode": SHARED_MEMBER_KERNEL_BANDWIDTH_MODE,
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

    backend_eff = _resolve_iibtd_backend(
        backend=solver_backend,
        n_sources=n_sources,
        solver_device=solver_device,
    )
    du_checkpoint_paths = (
        _split_du_checkpoint_paths(du_checkpoint_path)
        if backend_eff in _DU_IIBTD_BACKENDS
        else []
    )
    use_du_model_ensemble = bool(len(du_checkpoint_paths) > 1)
    expected_member_count = max(1, int(m_ens))
    if use_du_model_ensemble and len(du_checkpoint_paths) != expected_member_count:
        raise RuntimeError(
            "DU model ensemble requires one checkpoint per ensemble member: "
            f"got {len(du_checkpoint_paths)} checkpoints for {expected_member_count} members."
        )

    member_models: List[object] = []
    member_observation_counts: List[int] = []
    if backend_eff in _DU_IIBTD_BACKENDS:
        if not du_checkpoint_paths:
            raise RuntimeError("DU-IIBTD backend requires checkpoint path(s).")
        member_kernel_bandwidths_all = np.asarray(
            [
                _du_checkpoint_config_float(
                    du_checkpoint_paths[m % len(du_checkpoint_paths)],
                    "kernel_bandwidth",
                    float(kernel_bandwidth),
                )
                for m in range(expected_member_count)
            ],
            dtype=float,
        )
    else:
        member_kernel_bandwidths_all = _build_member_kernel_bandwidths(
            expected_member_count,
            base_kernel_bandwidth=kernel_bandwidth,
        )
    member_kernel_bandwidths: List[float] = []
    member_errors: List[str] = []
    for m in range(expected_member_count):
        member_du_checkpoint_path = (
            du_checkpoint_paths[m % len(du_checkpoint_paths)]
            if du_checkpoint_paths
            else du_checkpoint_path
        )
        member_kernel_bandwidth = float(member_kernel_bandwidths_all[m])
        try:
            model = fit_reconstruction_model(
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
                max_iter=member_max_iter,
                mu=mu,
                nu=nu,
                kernel_bandwidth=member_kernel_bandwidth,
                solver_backend=solver_backend,
                solver_device=solver_device,
                gpu_phi_solver=gpu_phi_solver,
                du_checkpoint_path=member_du_checkpoint_path,
                du_min_sensors_for_update=du_min_sensors_for_update,
                du_update_batch_size=du_update_batch_size,
                init_seed=seed + 20_000 + m,
                init_jitter_scale=(
                    0.0 if use_du_model_ensemble else SHARED_MEMBER_INIT_JITTER_SCALE
                ),
            )
        except Exception as exc:
            member_errors.append(f"member {m}: {type(exc).__name__}: {exc}")
            if backend_eff in _DU_IIBTD_BACKENDS:
                raise RuntimeError(
                    "DU ensemble member fit failed during full refresh: "
                    f"member={m}, checkpoint={member_du_checkpoint_path!r}"
                ) from exc
            continue
        member_models.append(model)
        member_observation_counts.append(int(total))
        member_kernel_bandwidths.append(float(getattr(model, "h", member_kernel_bandwidth)))

    if not member_models:
        detail = "; ".join(member_errors[-3:])
        raise RuntimeError("All ensemble member fits failed during full refresh." + (f" Last errors: {detail}" if detail else ""))
    if backend_eff in _DU_IIBTD_BACKENDS and len(member_models) != expected_member_count:
        detail = "; ".join(member_errors[-3:])
        raise RuntimeError(
            "DU ensemble full refresh did not build all members: "
            f"built {len(member_models)} of {expected_member_count}."
            + (f" Last errors: {detail}" if detail else "")
        )

    mean_map, var_map, stack, info = _summarize_ensemble_models(
        member_models,
        fused_obs_locs=fused_obs_locs,
        fused_gamma=fused_gamma,
        fused_omega=fused_omega,
        quality_weighted=quality_weighted,
        member_observation_counts=np.asarray(member_observation_counts, dtype=int),
        member_last_incremental_accept_counts=np.zeros(len(member_models), dtype=int),
        member_incremental_updated_mask=np.zeros(len(member_models), dtype=bool),
        member_kernel_bandwidths=np.asarray(member_kernel_bandwidths, dtype=float),
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
    solver_backend: str = DU_IIBTD_SR_BACKEND,
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
):
    """Incrementally update one reconstruction model with newly fused observations."""
    new_obs_locs = np.asarray(new_obs_locs, dtype=float)
    new_gamma = np.asarray(new_gamma, dtype=float)
    new_omega = np.asarray(new_omega, dtype=np.int32)
    if new_obs_locs.shape[0] == 0:
        return model

    def _fit_fallback(prev_model, warmstart: bool, min_outer_iter: int):
        return fit_reconstruction_model(
            new_obs_locs,
            new_gamma,
            new_omega,
            n_sources=n_sources,
            grid_size=grid_size,
            grid_points=grid_points,
            bounds=bounds,
            i_mask=i_mask,
            prev_model=prev_model,
            warmstart=warmstart,
            max_iter=max(int(min_outer_iter), int(n_outer_iter)),
            mu=mu,
            nu=nu,
            kernel_bandwidth=kernel_bandwidth,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            du_checkpoint_path=du_checkpoint_path,
            du_min_sensors_for_update=du_min_sensors_for_update,
            du_update_batch_size=du_update_batch_size,
        )

    if model is None:
        return _fit_fallback(prev_model=None, warmstart=False, min_outer_iter=4)

    if hasattr(model, "fit_incremental"):
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

    return _fit_fallback(prev_model=model, warmstart=True, min_outer_iter=2)


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
    member_kernel_bandwidths: Optional[np.ndarray] = None,
    solver_backend: str = DU_IIBTD_SR_BACKEND,
    solver_device: str = "auto",
    gpu_phi_solver: str = "scipy",
    du_checkpoint_path: str = "",
    du_min_sensors_for_update: int = 6,
    du_update_batch_size: Optional[int] = None,
    fusion_meta: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Broadcast new fused rows to all members, then recompute ensemble mean/variance."""
    member_models = list(member_models or [])
    if not member_models:
        raise RuntimeError("No ensemble members are available for incremental refresh.")
    backend_eff = _resolve_iibtd_backend(solver_backend, n_sources, solver_device)
    strict_du_backend = backend_eff in _DU_IIBTD_BACKENDS
    if member_kernel_bandwidths is None:
        if strict_du_backend:
            member_kernel_bandwidths = np.asarray(
                [
                    float(
                        getattr(
                            model,
                            "h",
                            _du_checkpoint_config_float(
                                _du_checkpoint_path_for_member(du_checkpoint_path, idx),
                                "kernel_bandwidth",
                                float(kernel_bandwidth),
                            ),
                        )
                    )
                    for idx, model in enumerate(member_models)
                ],
                dtype=float,
            )
        else:
            member_kernel_bandwidths = _build_member_kernel_bandwidths(
                len(member_models),
                base_kernel_bandwidth=kernel_bandwidth,
            )
    else:
        member_kernel_bandwidths = np.asarray(member_kernel_bandwidths, dtype=float)
    if member_kernel_bandwidths.shape[0] != len(member_models):
        raise RuntimeError("member_kernel_bandwidths does not match member_models.")

    updated_observation_counts = np.asarray(member_observation_counts, dtype=int).copy()
    if updated_observation_counts.shape[0] != len(member_models):
        raise RuntimeError("member_observation_counts does not match member_models.")
    member_accept_counts = np.zeros(len(member_models), dtype=int)
    member_updated_mask = np.zeros(len(member_models), dtype=bool)
    member_batches = _build_shared_incremental_batches(
        new_obs_locs,
        new_gamma,
        new_omega,
        member_count=len(member_models),
    )
    row_count = int(np.asarray(new_obs_locs, dtype=float).shape[0])
    updated_member_models: List[object] = []
    for member_idx, (model, batch) in enumerate(zip(member_models, member_batches)):
        if batch is None:
            updated_model = model
        else:
            member_du_checkpoint_path = (
                _du_checkpoint_path_for_member(du_checkpoint_path, member_idx)
                if strict_du_backend
                else du_checkpoint_path
            )
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
                    nu=nu,
                    kernel_bandwidth=float(member_kernel_bandwidths[member_idx]),
                    solver_backend=solver_backend,
                    solver_device=solver_device,
                    gpu_phi_solver=gpu_phi_solver,
                    du_checkpoint_path=member_du_checkpoint_path,
                    du_min_sensors_for_update=du_min_sensors_for_update,
                    du_update_batch_size=du_update_batch_size,
                )
                updated_observation_counts[member_idx] += row_count
                member_accept_counts[member_idx] = row_count
                member_updated_mask[member_idx] = True
            except Exception as exc:
                if strict_du_backend:
                    raise RuntimeError(
                        "DU ensemble member incremental update failed: "
                        f"member={member_idx}, checkpoint={member_du_checkpoint_path!r}"
                    ) from exc
                updated_model = model
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

import os
import sys
import argparse
import json
import time
from io import BytesIO
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in work1
    torch = None

try:
    from spectrumMapTensorGen import SimConfig, generate_data
except ModuleNotFoundError:
    from Test.spectrumMapTensorGen import SimConfig, generate_data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from IIBTD.IIBTD_Optimized import II_BTD_Optimized
from IIBTD.IIBTD_Opt_GPU import II_BTD_Opt_GPU


DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT_DIR, "Test", "outputs", "uav_mean_manhattan_compare"
)


EXPERIMENT_SPECS = [
    dict(
        key="mean_var",
        label="mean(var)",
        short_label="mean(var)",
        lambda_c=0.0,
        use_bit_cost=False,
        fixed_quantization_bits=8,
        marker="o",
        color="#2ca02c",
    ),
    dict(
        key="mean_var_distance",
        label="mean(var) + Manhattan distance cost",
        short_label="distance cost",
        lambda_c=5.0,
        use_bit_cost=False,
        fixed_quantization_bits=8,
        marker="s",
        color="#1f77b4",
    ),
    dict(
        key="mean_var_distance_bits",
        label="mean(var) + Manhattan distance cost + bits cost",
        short_label="distance + bits",
        lambda_c=5.0,
        use_bit_cost=True,
        fixed_quantization_bits=None,
        marker="^",
        color="#d62728",
    ),
]

EXPERIMENT_SPEC_BY_KEY = {spec["key"]: spec for spec in EXPERIMENT_SPECS}


def _format_plot_label(spec):
    base_spec = EXPERIMENT_SPEC_BY_KEY.get(spec["key"], spec)
    return str(base_spec.get("label", spec["label"]))


def _validate_quantization_pair(high_quantization_bits, low_quantization_bits):
    high_quantization_bits = max(1, int(high_quantization_bits))
    low_quantization_bits = max(1, int(low_quantization_bits))
    if high_quantization_bits < low_quantization_bits:
        raise ValueError(
            f"high_quantization_bits must be >= low_quantization_bits, got "
            f"{high_quantization_bits} < {low_quantization_bits}."
        )
    return high_quantization_bits, low_quantization_bits


def _select_quantization_bits_from_uncertainty(
    uncertainty_norm,
    adaptive_quantization_bits=False,
    quantization_bits=8,
    high_quantization_bits=8,
    low_quantization_bits=4,
    uncertainty_quantization_threshold=0.5,
    phase="planned_path",
):
    """Choose step quantization width from normalized uncertainty."""
    if not adaptive_quantization_bits:
        return int(max(1, int(quantization_bits))), "fixed"
    if phase == "warmup" or uncertainty_norm is None or not np.isfinite(float(uncertainty_norm)):
        return int(max(1, int(high_quantization_bits))), "warmup_high"
    if float(uncertainty_norm) >= float(uncertainty_quantization_threshold):
        return int(max(1, int(high_quantization_bits))), "high"
    return int(max(1, int(low_quantization_bits))), "low"


def build_uncertainty_quantization_map(
    uncertainty_space,
    adaptive_quantization_bits=False,
    quantization_bits=8,
    high_quantization_bits=8,
    low_quantization_bits=4,
    uncertainty_quantization_threshold=0.5,
):
    """Map normalized uncertainty to the expected quantization bit-width at each grid cell."""
    uncertainty_arr = np.asarray(uncertainty_space, dtype=float)
    if not adaptive_quantization_bits:
        fixed_bits = float(max(1, int(quantization_bits)))
        return np.full_like(uncertainty_arr, fixed_bits, dtype=float)

    high_bits = float(max(1, int(high_quantization_bits)))
    low_bits = float(max(1, int(low_quantization_bits)))
    return np.where(
        np.isfinite(uncertainty_arr) & (uncertainty_arr >= float(uncertainty_quantization_threshold)),
        high_bits,
        low_bits,
    )


def summarize_uncertainty_distribution(uncertainty_maps, threshold=0.5):
    """Aggregate normalized uncertainty statistics across active planning rounds."""
    if not uncertainty_maps:
        return None

    stack = np.stack([np.asarray(item, dtype=float) for item in uncertainty_maps], axis=0)
    flat = stack[np.isfinite(stack)]
    if flat.size == 0:
        return None

    threshold = float(threshold)
    return dict(
        round_count=int(stack.shape[0]),
        grid_shape=[int(stack.shape[1]), int(stack.shape[2])],
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        p10=float(np.quantile(flat, 0.10)),
        p25=float(np.quantile(flat, 0.25)),
        p50=float(np.quantile(flat, 0.50)),
        p75=float(np.quantile(flat, 0.75)),
        p90=float(np.quantile(flat, 0.90)),
        threshold=threshold,
        above_threshold_ratio=float(np.mean(flat >= threshold)),
    )


def resolve_experiment_specs(
    strategy_keys=None,
    high_quantization_bits=8,
    low_quantization_bits=4,
    uncertainty_quantization_threshold=0.5,
):
    if strategy_keys is None:
        strategy_keys = [spec["key"] for spec in EXPERIMENT_SPECS]

    high_quantization_bits, low_quantization_bits = _validate_quantization_pair(
        high_quantization_bits,
        low_quantization_bits,
    )
    resolved = []
    for key in strategy_keys:
        if key not in EXPERIMENT_SPEC_BY_KEY:
            raise ValueError(f"Unknown strategy key: {key}")
        spec = dict(EXPERIMENT_SPEC_BY_KEY[key])
        fixed_bits = spec.get("fixed_quantization_bits")
        if fixed_bits is None and spec.get("use_bit_cost", False):
            spec["adaptive_quantization_bits"] = True
            spec["high_quantization_bits"] = int(high_quantization_bits)
            spec["low_quantization_bits"] = int(low_quantization_bits)
        else:
            fixed_bits = max(1, int(fixed_bits if fixed_bits is not None else high_quantization_bits))
            spec["adaptive_quantization_bits"] = False
            spec["high_quantization_bits"] = fixed_bits
            spec["low_quantization_bits"] = fixed_bits
        spec["uncertainty_quantization_threshold"] = float(uncertainty_quantization_threshold)
        resolved.append(spec)
    return resolved

def _sample_ensemble_indices(M_total, keep_ratio, keep_recent, rng):
    """Subsample rows for one ensemble member while always keeping recent rows."""
    if M_total <= 1:
        return np.arange(M_total, dtype=int)

    keep_count = int(np.clip(round(keep_ratio * M_total), 1, M_total))
    keep_recent = int(np.clip(keep_recent, 0, M_total))

    recent_start = max(0, M_total - keep_recent)
    recent_idx = np.arange(recent_start, M_total, dtype=int)
    if recent_idx.size >= keep_count:
        return recent_idx[-keep_count:]

    pool = np.arange(0, recent_start, dtype=int)
    n_rand = min(keep_count - recent_idx.size, pool.size)
    rand_idx = rng.choice(pool, size=n_rand, replace=False) if n_rand > 0 else np.empty((0,), dtype=int)

    idx = np.concatenate([rand_idx, recent_idx])
    if idx.size < keep_count:
        missing = keep_count - idx.size
        remain = np.setdiff1d(np.arange(M_total, dtype=int), idx, assume_unique=False)
        if remain.size > 0:
            fill = rng.choice(remain, size=min(missing, remain.size), replace=False)
            idx = np.concatenate([idx, fill])
    return np.sort(np.unique(idx))


def _copy_model_state(dst_model, src_model):
    """Copy warm-start state between solvers with the same dimensions."""
    if hasattr(dst_model, "load_state_from"):
        dst_model.load_state_from(src_model)
        return
    for name in ("Theta", "Phi", "Sr", "H_hat"):
        if hasattr(src_model, name):
            setattr(dst_model, name, np.array(getattr(src_model, name), copy=True))


def _resolve_iibtd_backend(backend, n_sources=1, solver_device="auto"):
    backend = str(backend or "cpu").strip().lower()
    device_str = str(solver_device or "auto").strip().lower()
    if backend == "gpu":
        if torch is not None and torch.cuda.is_available():
            return "gpu"
        return "cpu"
    if backend == "auto":
        if (
            torch is not None
            and torch.cuda.is_available()
            and int(n_sources) == 1
            and device_str != "cpu"
        ):
            return "gpu"
        return "cpu"
    return "cpu"


def _resolve_iibtd_device(solver_device="auto"):
    device_str = str(solver_device or "auto").strip()
    if not device_str or device_str.lower() == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return device_str


def _make_reconstruction_solver(
    cfg,
    warmstart=False,
    max_iter=6,
    solver_backend="cpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
):
    backend_eff = _resolve_iibtd_backend(
        solver_backend,
        n_sources=cfg.R,
        solver_device=solver_device,
    )
    common_kwargs = dict(
        n_sources=cfg.R,
        grid_size=(cfg.N1, cfg.N2),
        mu=1.2,
        nu=1.5,
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=0.46,
        warmstart=bool(warmstart),
    )
    if backend_eff == "gpu":
        return II_BTD_Opt_GPU(
            **common_kwargs,
            device=_resolve_iibtd_device(solver_device),
            phi_solver=str(gpu_phi_solver).strip().lower() or "scipy",
            dtype=torch.float64 if torch is not None else None,
        )
    return II_BTD_Optimized(**common_kwargs)


def _fit_reconstruction_model(
    sensor_locs,
    gamma,
    omega,
    cfg,
    grid_points,
    bounds,
    I_mask,
    prev_model=None,
    warmstart=False,
    max_iter=6,
    solver_backend="cpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
):
    """Fit one reconstruction model, optionally warm-started from prev_model."""
    model = _make_reconstruction_solver(
        cfg,
        warmstart=warmstart,
        max_iter=max_iter,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )
    model.init_sequential(grid_points, bounds, K=cfg.K, I_mask=I_mask)
    if warmstart and prev_model is not None:
        _copy_model_state(model, prev_model)
    model.fit_2(
        sensor_locs,
        gamma,
        omega,
        grid_points,
        bounds,
        I_mask=I_mask,
        debugFlag=False,
    )
    return model


def _reconstruct_model(
    sensor_locs,
    gamma,
    omega,
    cfg,
    grid_points,
    bounds,
    I_mask,
    prev_model=None,
    warmstart=False,
    max_iter=6,
    solver_backend="cpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
    reconstruction_method="history_refit",
    seq_svt_iter=20,
):
    reconstruction_method = str(reconstruction_method).strip().lower()
    if reconstruction_method == "sequential_incremental":
        model = _make_reconstruction_solver(
            cfg,
            warmstart=warmstart,
            max_iter=max_iter,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
        )
        model.init_sequential(grid_points, bounds, K=cfg.K, I_mask=I_mask)
        if warmstart and prev_model is not None:
            _copy_model_state(model, prev_model)
        if np.asarray(sensor_locs).shape[0] > 0:
            model.add_measurements(
                sensor_locs,
                gamma,
                omega,
                n_outer_iter=max(1, int(max_iter)),
                max_svt_iter=max(1, int(seq_svt_iter)),
                debugFlag=False,
            )
        return model
    return _fit_reconstruction_model(
        sensor_locs,
        gamma,
        omega,
        cfg=cfg,
        grid_points=grid_points,
        bounds=bounds,
        I_mask=I_mask,
        prev_model=prev_model,
        warmstart=warmstart,
        max_iter=max_iter,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )


def _predict_observed_rows(map_hat, sensor_locs, N1, N2):
    """Project reconstructed map to observed grid rows using nearest integer grid."""
    locs = np.asarray(sensor_locs, dtype=float)
    grid_xy = np.floor(locs).astype(int)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, N1 - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, N2 - 1)
    return map_hat[grid_xy[:, 0], grid_xy[:, 1], :]


def _observation_nmse(map_hat, sensor_locs, gamma, omega, N1, N2):
    """Evaluate fit quality on observed entries only."""
    mask = np.asarray(omega) > 0
    if not np.any(mask):
        return np.inf

    pred = _predict_observed_rows(map_hat, sensor_locs, N1, N2)
    diff = pred[mask] - gamma[mask]
    denom = np.linalg.norm(gamma[mask]) ** 2
    return float(np.linalg.norm(diff) ** 2 / (denom + 1e-9))


def _weighted_map_statistics(maps, scores):
    """Weighted mean/variance over ensemble members."""
    maps = np.asarray(maps, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if maps.shape[0] == 1:
        return maps[0], np.zeros_like(maps[0]), np.ones(1, dtype=float)

    weights = np.maximum(scores, 1e-12)
    weights = weights / np.sum(weights)
    mean_map = np.tensordot(weights, maps, axes=(0, 0))
    diff = maps - mean_map[np.newaxis, ...]
    var_map = np.tensordot(weights, diff ** 2, axes=(0, 0))
    return mean_map, var_map, weights


def _normalize_score_map(values, eps=1e-12):
    """Normalize a score map to [0, 1] so weighted terms stay comparable."""
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


def ensemble_reconstruct_maps(
    sensor_locs,
    gamma,
    omega,
    cfg,
    grid_points,
    bounds,
    I_mask,
    M_ens=8,
    keep_ratio=0.9,
    keep_recent=3,
    seed=42,
    base_model=None,
    member_max_iter=3,
    quality_weighted=True,
    solver_backend="cpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
    reconstruction_method="history_refit",
):
    """Run base warm-started ensemble II-BTD and return mean/variance maps."""

    M_total = sensor_locs.shape[0]
    if M_total == 0:
        raise ValueError("sensor_locs is empty.")

    if base_model is None:
        base_model = _reconstruct_model(
            sensor_locs,
            gamma,
            omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=I_mask,
            prev_model=None,
            warmstart=False,
            max_iter=max(4, int(member_max_iter) + 1),
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            reconstruction_method=reconstruction_method,
        )

    maps = []
    obs_nmse_scores = []
    for m in range(max(1, int(M_ens))):
        rng_m = np.random.default_rng(seed + 1000 + m)
        idx = _sample_ensemble_indices(M_total, keep_ratio, keep_recent, rng_m)

        model = _reconstruct_model(
            sensor_locs[idx],
            gamma[idx],
            omega[idx],
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=I_mask,
            prev_model=base_model,
            warmstart=True,
            max_iter=member_max_iter,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            reconstruction_method=reconstruction_method,
        )
        map_m = model.get_current_map()
        maps.append(map_m)
        obs_nmse_scores.append(
            _observation_nmse(map_m, sensor_locs, gamma, omega, cfg.N1, cfg.N2)
        )

    maps = np.stack(maps, axis=0)
    if quality_weighted:
        quality_scores = 1.0 / (np.asarray(obs_nmse_scores) + 1e-8)
        mean_map, var_map, weights = _weighted_map_statistics(maps, quality_scores)
    else:
        mean_map = np.mean(maps, axis=0)
        if maps.shape[0] > 1:
            var_map = np.var(maps, axis=0, ddof=1)
        else:
            var_map = np.zeros_like(mean_map)
        weights = np.full(maps.shape[0], 1.0 / maps.shape[0], dtype=float)
    info = dict(
        base_model=base_model,
        weights=np.asarray(weights, dtype=float),
        obs_nmse=np.asarray(obs_nmse_scores, dtype=float),
    )
    return mean_map, var_map, maps, info


def build_acquisition_space(
    var_map,
    next_points,
    current_loc,
    sampled_locs,
    lambda_u=1.0,
    lambda_c=0.15,
    lambda_r=0.08,
    lambda_b=0.0,
    bit_cost_map=None,
    redundancy_length=5.0,
    distance_cost_mode="euclidean",
):
    """Build spatial acquisition score from uncertainty/motion/redundancy/bit costs."""
    N1, N2, _ = var_map.shape
    raw_uncertainty_space = np.mean(var_map, axis=2)
    # Ensemble variance is often much smaller in magnitude than normalized
    # movement/redundancy costs; normalize it before combining the terms.
    uncertainty_space = _normalize_score_map(raw_uncertainty_space)

    move_diff = next_points - current_loc[np.newaxis, :]
    if distance_cost_mode == "manhattan":
        dist_move = np.sum(np.abs(move_diff), axis=1).reshape(N1, N2)
    elif distance_cost_mode == "euclidean":
        dist_move = np.linalg.norm(move_diff, axis=1).reshape(N1, N2)
    else:
        raise ValueError(f"Unknown distance_cost_mode: {distance_cost_mode}")
    # Keep the movement term on the same normalized scale as uncertainty so
    # lambda_u and lambda_c directly express the intended trade-off.
    move_cost = _normalize_score_map(dist_move)

    if sampled_locs.shape[0] > 0:
        diff = next_points[:, np.newaxis, :] - sampled_locs[np.newaxis, :, :]
        min_dist = np.min(np.linalg.norm(diff, axis=2), axis=1).reshape(N1, N2)
        redundancy = np.exp(-min_dist / max(1e-6, redundancy_length))
    else:
        redundancy = np.zeros((N1, N2), dtype=float)

    if bit_cost_map is None:
        bit_cost = np.zeros((N1, N2), dtype=float)
    else:
        bit_cost = _normalize_score_map(np.asarray(bit_cost_map, dtype=float).reshape(N1, N2))

    acquisition_space = (
        lambda_u * uncertainty_space
        - lambda_c * move_cost
        - lambda_r * redundancy
        - lambda_b * bit_cost
    )
    components = dict(
        raw_uncertainty_space=raw_uncertainty_space,
        uncertainty_space=uncertainty_space,
        move_cost=move_cost,
        redundancy=redundancy,
        bit_cost=bit_cost,
    )
    return acquisition_space, components


def quantize_to_nbit(values, n_bits=8, value_min=None, value_max=None):
    """Uniform N-bit quantization with de-quantized float output."""
    arr = np.asarray(values, dtype=float)
    n_bits = max(1, int(n_bits))
    levels = int(2 ** n_bits)
    if arr.size == 0:
        return arr.copy(), {"min": 0.0, "max": 0.0, "levels": levels, "n_bits": n_bits}

    if value_min is None:
        v_min = float(np.min(arr))
    else:
        v_min = float(value_min)
    if value_max is None:
        v_max = float(np.max(arr))
    else:
        v_max = float(value_max)
    if v_max <= v_min + 1e-12:
        return np.full_like(arr, v_min, dtype=float), {
            "min": v_min,
            "max": v_max,
            "levels": levels,
            "n_bits": n_bits,
        }

    norm = (arr - v_min) / (v_max - v_min)
    quantized = np.round(norm * float(levels - 1))
    dq = (quantized / float(levels - 1)) * (v_max - v_min) + v_min
    return dq, {"min": v_min, "max": v_max, "levels": levels, "n_bits": n_bits}


def estimate_transmitted_bits(omega_row, quantization_bits):
    """Payload bits for one measurement row."""
    omega_arr = np.asarray(omega_row, dtype=np.int32)
    return int(np.sum(omega_arr > 0)) * max(1, int(quantization_bits))

def build_observe_mask(K, center_freq, band_width):
    band_width = int(np.clip(band_width, 1, K))
    center_freq = int(np.clip(center_freq, 0, K - 1))

    # 0-based band window [freq_start, freq_end) with exact width.
    freq_start = center_freq - (band_width // 2)
    freq_end = freq_start + band_width
    if freq_start < 0:
        freq_start = 0
        freq_end = band_width
    elif freq_end > K:
        freq_end = K
        freq_start = K - band_width

    omega = np.zeros(K, dtype=np.int32)
    omega[freq_start:freq_end] = 1
    observed_bands = np.where(omega > 0)[0]
    return omega, observed_bands


def measure_on_grid_with_mask(data, grid_loc, omega, rng, quantization_bits=8):
    """Measure one on-grid location with fresh noise and N-bit quantization."""
    cfg = data["config"]
    K = cfg.K

    grid_loc = np.asarray(grid_loc, dtype=int)
    x_idx = int(np.clip(grid_loc[0], 0, cfg.N1 - 1))
    y_idx = int(np.clip(grid_loc[1], 0, cfg.N2 - 1))
    loc = np.array([[float(x_idx), float(y_idx)]], dtype=float)

    true_spec = data["H"][x_idx, y_idx, :].astype(float)
    noise = rng.normal(0.0, np.sqrt(float(data["sigma2_noise"])), size=K)
    observed = np.maximum(true_spec + noise, 1e-10)

    omega = np.asarray(omega, dtype=np.int32)
    observed_mask = omega > 0
    gamma_row = np.zeros(K, dtype=float)
    if np.any(observed_mask):
        gamma_row[observed_mask], _ = quantize_to_nbit(
            observed[observed_mask],
            n_bits=quantization_bits,
        )
    return loc, gamma_row[np.newaxis, :], omega[np.newaxis, :], true_spec


def _sample_random_grid_cell(rng, N1, N2):
    return np.array(
        [int(rng.integers(0, N1)), int(rng.integers(0, N2))],
        dtype=int,
    )


def _grid_neighbors4(cell, N1, N2):
    x_idx, y_idx = int(cell[0]), int(cell[1])
    neighbors = []
    if x_idx > 0:
        neighbors.append((x_idx - 1, y_idx))
    if x_idx < N1 - 1:
        neighbors.append((x_idx + 1, y_idx))
    if y_idx > 0:
        neighbors.append((x_idx, y_idx - 1))
    if y_idx < N2 - 1:
        neighbors.append((x_idx, y_idx + 1))
    return neighbors


def _choose_random_cardinal_step(current_cell, target_cell, rng, N1, N2):
    current_cell = np.asarray(current_cell, dtype=int)
    target_cell = np.asarray(target_cell, dtype=int)
    neighbors = _grid_neighbors4(current_cell, N1, N2)
    if not neighbors:
        return current_cell.copy()

    base_dist = abs(int(target_cell[0]) - int(current_cell[0])) + abs(int(target_cell[1]) - int(current_cell[1]))
    toward = [
        nb for nb in neighbors
        if abs(int(target_cell[0]) - nb[0]) + abs(int(target_cell[1]) - nb[1]) < base_dist
    ]
    candidates = toward if toward else neighbors
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    return np.array(chosen, dtype=int)


def _uncertainty_to_path_cost(uncertainty_space, mode="inverse", eps=1e-6):
    """Map uncertainty scores to path costs so higher uncertainty means lower cost."""
    uncertainty_norm = _normalize_score_map(uncertainty_space)
    if mode == "inverse":
        raw_cost = 1.0 / np.maximum(uncertainty_norm, float(eps))
        return _normalize_score_map(raw_cost)
    if mode == "one_minus":
        return 1.0 - uncertainty_norm
    raise ValueError(f"Unknown path uncertainty cost mode: {mode}")


def _plan_cardinal_path(
    current_cell,
    target_cell,
    uncertainty_space,
    step_length=1.0,
    path_eta=0.75,
    uncertainty_cost_mode="inverse",
):
    """Weighted 4-neighbor shortest path minimizing (1-eta)T(X) + eta*C(X)."""
    import heapq

    current = tuple(np.asarray(current_cell, dtype=int).tolist())
    target = tuple(np.asarray(target_cell, dtype=int).tolist())
    N1, N2 = uncertainty_space.shape
    step_length = float(step_length)
    path_eta = float(np.clip(path_eta, 0.0, 1.0))
    uncertainty_cost_map = _uncertainty_to_path_cost(
        uncertainty_space,
        mode=uncertainty_cost_mode,
    )

    if current == target:
        path = [np.array(current, dtype=float)]
        return path, 0.0, 0.0, 0.0

    best_total_cost = {current: 0.0}
    best_distance = {current: 0.0}
    best_uncertainty_cost = {current: 0.0}
    parent = {}
    heap = [(0.0, 0.0, 0.0, current)]

    while heap:
        total_cost_so_far, dist_so_far, unc_cost_so_far, node = heapq.heappop(heap)
        best_cost = best_total_cost[node]
        if total_cost_so_far > best_cost + 1e-12:
            continue
        if node == target:
            break

        for nb in _grid_neighbors4(node, N1, N2):
            new_dist = dist_so_far + step_length
            edge_unc_cost = float(uncertainty_cost_map[nb[0], nb[1]]) * step_length
            new_unc_cost = unc_cost_so_far + edge_unc_cost
            edge_total_cost = (1.0 - path_eta) * step_length + path_eta * edge_unc_cost
            new_total_cost = total_cost_so_far + edge_total_cost

            old_cost = best_total_cost.get(nb)
            if old_cost is None or new_total_cost < old_cost - 1e-12 or (
                abs(new_total_cost - old_cost) <= 1e-12 and new_dist < best_distance[nb] - 1e-12
            ):
                best_total_cost[nb] = new_total_cost
                best_distance[nb] = new_dist
                best_uncertainty_cost[nb] = new_unc_cost
                parent[nb] = node
                heapq.heappush(
                    heap,
                    (new_total_cost, new_dist, new_unc_cost, nb),
                )

    if target not in parent and target != current:
        raise RuntimeError(f"Failed to plan path from {current} to {target}.")

    path_nodes = [target]
    while path_nodes[-1] != current:
        path_nodes.append(parent[path_nodes[-1]])
    path_nodes.reverse()

    path = [np.array(node, dtype=float) for node in path_nodes]
    total_distance = best_distance[target]
    total_uncertainty_cost = best_uncertainty_cost[target]
    total_path_cost = best_total_cost[target]
    return path, float(total_distance), float(total_uncertainty_cost), float(total_path_cost)


def _build_executed_path(path_cells, move_stride=1):
    """Subsample a planned path into executed measurement waypoints."""
    nodes = [np.asarray(cell, dtype=int) for cell in path_cells]
    if not nodes:
        return [], []

    move_stride = max(1, int(move_stride))
    if len(nodes) == 1:
        return [nodes[0].copy()], [0]

    executed_cells = []
    hop_counts = []
    prev_idx = 0
    next_idx = move_stride

    while next_idx < len(nodes):
        executed_cells.append(nodes[next_idx].copy())
        hop_counts.append(int(next_idx - prev_idx))
        prev_idx = int(next_idx)
        next_idx += move_stride

    final_idx = len(nodes) - 1
    if prev_idx < final_idx:
        executed_cells.append(nodes[final_idx].copy())
        hop_counts.append(int(final_idx - prev_idx))

    return executed_cells, hop_counts


def _adaptive_keep_ratio(M_total, early_ratio=0.95, late_ratio=0.85, switch_M=80):
    """Early stage keeps more observations; later stage allows stronger perturbation."""
    if M_total <= switch_M:
        return early_ratio
    alpha = np.clip((M_total - switch_M) / max(1, switch_M), 0.0, 1.0)
    return (1.0 - alpha) * early_ratio + alpha * late_ratio


def _append_fifo_observations(sensor_locs, gamma, omega, new_sensor_locs, new_gamma, new_omega, max_len=None):
    """Append rows and optionally keep only the latest max_len rows."""
    new_sensor_locs = np.atleast_2d(np.asarray(new_sensor_locs, dtype=float))
    new_gamma = np.atleast_2d(np.asarray(new_gamma, dtype=float))
    new_omega = np.atleast_2d(np.asarray(new_omega, dtype=np.int32))

    if sensor_locs.size == 0:
        merged_locs = new_sensor_locs.copy()
        merged_gamma = new_gamma.copy()
        merged_omega = new_omega.copy()
    else:
        merged_locs = np.vstack([sensor_locs, new_sensor_locs])
        merged_gamma = np.vstack([gamma, new_gamma])
        merged_omega = np.vstack([omega, new_omega])

    dropped = 0
    if max_len is not None:
        max_len = max(1, int(max_len))
        if merged_locs.shape[0] > max_len:
            dropped = merged_locs.shape[0] - max_len
            merged_locs = merged_locs[-max_len:]
            merged_gamma = merged_gamma[-max_len:]
            merged_omega = merged_omega[-max_len:]

    return merged_locs, merged_gamma, merged_omega, dropped


def _fuse_observations_by_grid(sensor_locs, gamma, omega, N1, N2):
    """Merge repeated observations in the same integer grid into one sparse row."""
    sensor_locs = np.asarray(sensor_locs, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=np.int32)

    if sensor_locs.shape[0] == 0:
        empty_locs = np.empty((0, 2), dtype=float)
        empty_gamma = np.empty((0, gamma.shape[1] if gamma.ndim == 2 else 0), dtype=float)
        empty_omega = np.empty((0, omega.shape[1] if omega.ndim == 2 else 0), dtype=np.int32)
        return empty_locs, empty_gamma, empty_omega, dict(raw_count=0, fused_count=0, compression_ratio=1.0)

    K = gamma.shape[1]
    grid_xy = np.floor(sensor_locs).astype(int)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, N1 - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, N2 - 1)

    fused = {}
    for idx, (gx, gy) in enumerate(grid_xy):
        key = (int(gx), int(gy))
        row = fused.setdefault(
            key,
            dict(
                loc_sum=np.zeros(2, dtype=float),
                loc_count=0,
                gamma_sum=np.zeros(K, dtype=float),
                gamma_count=np.zeros(K, dtype=float),
                omega=np.zeros(K, dtype=np.int32),
                last_idx=-1,
                raw_count=0,
            ),
        )
        row["loc_sum"] += sensor_locs[idx]
        row["loc_count"] += 1
        row["raw_count"] += 1
        row["last_idx"] = idx

        observed = omega[idx] > 0
        if np.any(observed):
            row["omega"][observed] = 1
            row["gamma_sum"][observed] += gamma[idx, observed]
            row["gamma_count"][observed] += 1.0

    ordered_keys = sorted(fused.keys(), key=lambda key: fused[key]["last_idx"])
    fused_locs = np.zeros((len(ordered_keys), 2), dtype=float)
    fused_gamma = np.zeros((len(ordered_keys), K), dtype=float)
    fused_omega = np.zeros((len(ordered_keys), K), dtype=np.int32)

    raw_counts = []
    for row_idx, key in enumerate(ordered_keys):
        row = fused[key]
        fused_locs[row_idx] = row["loc_sum"] / max(1, row["loc_count"])
        fused_omega[row_idx] = row["omega"]
        valid = row["gamma_count"] > 0
        fused_gamma[row_idx, valid] = row["gamma_sum"][valid] / row["gamma_count"][valid]
        raw_counts.append(int(row["raw_count"]))

    meta = dict(
        raw_count=int(sensor_locs.shape[0]),
        fused_count=int(len(ordered_keys)),
        compression_ratio=float(sensor_locs.shape[0] / max(1, len(ordered_keys))),
        raw_per_fused=np.asarray(raw_counts, dtype=int),
    )
    return fused_locs, fused_gamma, fused_omega, meta


def _select_outer_iters(effective_count, warmstart=False):
    """Use more outer iterations early and fewer later."""
    effective_count = int(max(1, effective_count))
    if warmstart:
        if effective_count < 12:
            return 5
        if effective_count < 24:
            return 4
        return 3
    if effective_count < 12:
        return 6
    if effective_count < 24:
        return 5
    if effective_count < 48:
        return 4
    return 3


def _map_nmse(map_hat, map_true):
    map_hat = np.asarray(map_hat, dtype=float)
    map_true = np.asarray(map_true, dtype=float)
    return float(
        np.linalg.norm(map_hat - map_true) ** 2 / (np.linalg.norm(map_true) ** 2 + 1e-12)
    )


def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _proxy_error_summary(proxy_values, true_values):
    proxy_values = np.asarray(proxy_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    mask = np.isfinite(proxy_values) & np.isfinite(true_values)
    if int(np.sum(mask)) == 0:
        return dict(
            count=0,
            corr=np.nan,
            mae=np.nan,
            rmse=np.nan,
            bias=np.nan,
            mean_ratio=np.nan,
        )

    proxy = proxy_values[mask]
    target = true_values[mask]
    diff = proxy - target
    return dict(
        count=int(mask.sum()),
        corr=_safe_corrcoef(proxy, target),
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        bias=float(np.mean(diff)),
        mean_ratio=float(np.mean(proxy / np.maximum(target, 1e-12))),
    )


def _map_correlation(map_a, map_b):
    map_a = np.asarray(map_a, dtype=float)
    map_b = np.asarray(map_b, dtype=float)
    return _safe_corrcoef(map_a.reshape(-1), map_b.reshape(-1))


def _replay_sequential_incremental(
    result,
    solver_backend="gpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
    seq_svt_iter=20,
):
    data = result["data"]
    cfg = data["config"]
    bounds = ((0, cfg.L), (0, cfg.L))
    grid_points = np.asarray(data["grid_coords"], dtype=float)
    I_mask = np.asarray(data["I_mask"], dtype=bool)
    sensor_locs = np.asarray(result["sensor_locs_final"], dtype=float)
    gamma = np.asarray(result["gamma_final"], dtype=float)
    omega = np.asarray(result["omega_final"], dtype=np.int32)

    solver = _make_reconstruction_solver(
        cfg,
        warmstart=False,
        max_iter=6,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )
    solver.init_sequential(grid_points, bounds, K=cfg.K, I_mask=I_mask)

    nmse_trace = []
    time_trace_s = []
    for idx in range(sensor_locs.shape[0]):
        outer_iter = _select_outer_iters(idx + 1, warmstart=(idx > 0))
        t0 = time.perf_counter()
        solver.add_measurements(
            sensor_locs[idx:idx + 1],
            gamma[idx:idx + 1],
            omega[idx:idx + 1],
            n_outer_iter=outer_iter,
            max_svt_iter=seq_svt_iter,
            debugFlag=False,
        )
        time_trace_s.append(float(time.perf_counter() - t0))
        nmse_trace.append(_map_nmse(solver.get_current_map(), data["H"]))

    return dict(
        key="sequential_incremental",
        label="Sequential Incremental",
        final_map=solver.get_current_map(),
        nmse_trace=np.asarray(nmse_trace, dtype=float),
        time_trace_s=np.asarray(time_trace_s, dtype=float),
        cumulative_time_s=np.cumsum(np.asarray(time_trace_s, dtype=float)),
        final_nmse=float(nmse_trace[-1]) if nmse_trace else np.nan,
        total_time_s=float(np.sum(time_trace_s)),
        sample_count=int(sensor_locs.shape[0]),
    )


def _replay_history_refit(
    result,
    solver_backend="gpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
):
    data = result["data"]
    cfg = data["config"]
    bounds = ((0, cfg.L), (0, cfg.L))
    grid_points = np.asarray(data["grid_coords"], dtype=float)
    I_mask = np.asarray(data["I_mask"], dtype=bool)
    sensor_locs_all = np.asarray(result["sensor_locs_final"], dtype=float)
    gamma_all = np.asarray(result["gamma_final"], dtype=float)
    omega_all = np.asarray(result["omega_final"], dtype=np.int32)
    fifo_queue_size = result.get("fifo_queue_size")

    queue_sensor_locs = np.empty((0, 2), dtype=float)
    queue_gamma = np.empty((0, cfg.K), dtype=float)
    queue_omega = np.empty((0, cfg.K), dtype=np.int32)
    prev_model = None

    nmse_trace = []
    time_trace_s = []

    for idx in range(sensor_locs_all.shape[0]):
        queue_sensor_locs, queue_gamma, queue_omega, _ = _append_fifo_observations(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            sensor_locs_all[idx:idx + 1],
            gamma_all[idx:idx + 1],
            omega_all[idx:idx + 1],
            max_len=fifo_queue_size,
        )
        fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            cfg.N1,
            cfg.N2,
        )
        refit_max_iter = _select_outer_iters(
            fusion_meta["fused_count"],
            warmstart=(prev_model is not None),
        )
        t0 = time.perf_counter()
        model = _fit_reconstruction_model(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=I_mask,
            prev_model=prev_model,
            warmstart=(prev_model is not None),
            max_iter=refit_max_iter,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
        )
        time_trace_s.append(float(time.perf_counter() - t0))
        prev_model = model
        nmse_trace.append(_map_nmse(model.get_current_map(), data["H"]))

    return dict(
        key="history_refit",
        label="MAPPO-Style History Refit",
        final_map=prev_model.get_current_map(),
        nmse_trace=np.asarray(nmse_trace, dtype=float),
        time_trace_s=np.asarray(time_trace_s, dtype=float),
        final_nmse=float(nmse_trace[-1]) if nmse_trace else np.nan,
        total_time_s=float(np.sum(time_trace_s)),
        sample_count=int(sensor_locs_all.shape[0]),
    )


def plot_reconstruction_mode_comparison(base_result, sequential_result, history_refit_result, save_path):
    data = base_result["data"]
    cfg = data["config"]
    true_energy = np.sum(np.asarray(data["H"], dtype=float), axis=2)
    seq_energy = np.sum(np.asarray(sequential_result["final_map"], dtype=float), axis=2)
    hist_energy = np.sum(np.asarray(history_refit_result["final_map"], dtype=float), axis=2)
    sensor_locs = np.asarray(base_result["sensor_locs_final"], dtype=float)
    warmup_count = int(base_result.get("warmup_sample_target", 0))

    vmin = float(np.min(true_energy))
    vmax = float(np.max(true_energy))
    steps = np.arange(1, int(sequential_result["sample_count"]) + 1, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    im = ax.imshow(true_energy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if sensor_locs.size > 0:
        ax.plot(sensor_locs[:, 0], sensor_locs[:, 1], color="white", lw=1.2, alpha=0.8)
        ax.scatter(sensor_locs[:, 0], sensor_locs[:, 1], s=10, c=np.arange(sensor_locs.shape[0]), cmap="cool")
        ax.scatter(sensor_locs[0, 0], sensor_locs[0, 1], s=90, c="#00f5d4", marker="^", edgecolors="black")
        ax.scatter(sensor_locs[-1, 0], sensor_locs[-1, 1], s=90, c="#ff006e", marker="s", edgecolors="black")
    ax.set_title("True Energy + UAV Sequential Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, shrink=0.82)

    ax = axes[0, 1]
    im = ax.imshow(seq_energy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Sequential Incremental\nfinal NMSE={sequential_result['final_nmse']:.4f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, shrink=0.82)

    ax = axes[0, 2]
    im = ax.imshow(hist_energy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"MAPPO-Style History Refit\nfinal NMSE={history_refit_result['final_nmse']:.4f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, shrink=0.82)

    ax = axes[1, 0]
    ax.plot(steps, sequential_result["nmse_trace"], color="#1f77b4", lw=2.0, label="Sequential incremental")
    ax.plot(steps, history_refit_result["nmse_trace"], color="#d62728", lw=2.0, label="MAPPO-style history refit")
    if warmup_count > 0 and warmup_count < len(steps):
        ax.axvline(warmup_count, color="#666666", ls="--", lw=1.0, label="warmup end")
    ax.set_title("NMSE vs Sampling Step")
    ax.set_xlabel("Sampling step")
    ax.set_ylabel("NMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, sequential_result["time_trace_s"], color="#1f77b4", lw=1.8, label="Sequential incremental")
    ax.plot(steps, history_refit_result["time_trace_s"], color="#d62728", lw=1.8, label="MAPPO-style history refit")
    if warmup_count > 0 and warmup_count < len(steps):
        ax.axvline(warmup_count, color="#666666", ls="--", lw=1.0)
    ax.set_title("Per-Step Reconstruction Time")
    ax.set_xlabel("Sampling step")
    ax.set_ylabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    seq_total = float(sequential_result["total_time_s"])
    hist_total = float(history_refit_result["total_time_s"])
    speedup = float(hist_total / seq_total) if seq_total > 0 else np.inf
    summary_lines = [
        "Reconstruction Summary",
        "",
        f"Samples: {int(sequential_result['sample_count'])}",
        f"Warmup samples: {warmup_count}",
        "",
        f"Sequential final NMSE: {sequential_result['final_nmse']:.6f}",
        f"History-refit final NMSE: {history_refit_result['final_nmse']:.6f}",
        f"Final NMSE gap: {abs(sequential_result['final_nmse'] - history_refit_result['final_nmse']):.6e}",
        "",
        f"Sequential total time: {seq_total:.3f}s",
        f"History-refit total time: {hist_total:.3f}s",
        f"Sequential speedup: {speedup:.3f}x",
        "",
        f"Sequential mean step time: {np.mean(sequential_result['time_trace_s']):.4f}s",
        f"History-refit mean step time: {np.mean(history_refit_result['time_trace_s']):.4f}s",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_reconstruction_method_comparison(
    common_kwargs,
    output_dir,
    solver_backend="gpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
):
    os.makedirs(output_dir, exist_ok=True)
    base_kwargs = dict(common_kwargs)
    base_kwargs["summary_save_path"] = None

    base_result = run_active_ensemble_cyclic_experiment(**base_kwargs)
    sequential_result = _replay_sequential_incremental(
        base_result,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )
    history_refit_result = _replay_history_refit(
        base_result,
        solver_backend=solver_backend,
        solver_device=solver_device,
        gpu_phi_solver=gpu_phi_solver,
    )

    figure_path = os.path.join(output_dir, "reconstruction_mode_compare.png")
    summary_path = os.path.join(output_dir, "reconstruction_mode_compare.json")
    plot_reconstruction_mode_comparison(
        base_result,
        sequential_result,
        history_refit_result,
        figure_path,
    )

    payload = dict(
        solver_backend=str(solver_backend),
        solver_device=str(solver_device),
        gpu_phi_solver=str(gpu_phi_solver),
        common_kwargs={
            key: value
            for key, value in dict(common_kwargs).items()
            if key not in {"solver_backend", "solver_device", "gpu_phi_solver"}
        },
        sequential_incremental=dict(
            final_nmse=float(sequential_result["final_nmse"]),
            total_time_s=float(sequential_result["total_time_s"]),
            mean_step_time_s=float(np.mean(sequential_result["time_trace_s"])),
        ),
        history_refit=dict(
            final_nmse=float(history_refit_result["final_nmse"]),
            total_time_s=float(history_refit_result["total_time_s"]),
            mean_step_time_s=float(np.mean(history_refit_result["time_trace_s"])),
        ),
        speedup=float(history_refit_result["total_time_s"] / sequential_result["total_time_s"])
        if sequential_result["total_time_s"] > 0
        else float("inf"),
        figure_path=figure_path,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Reconstruction Method Comparison")
    print(
        f"Sequential incremental: final_nmse={sequential_result['final_nmse']:.6f}, "
        f"total_time={sequential_result['total_time_s']:.3f}s"
    )
    print(
        f"MAPPO-style history refit: final_nmse={history_refit_result['final_nmse']:.6f}, "
        f"total_time={history_refit_result['total_time_s']:.3f}s"
    )
    print(
        f"Sequential speedup vs history-refit: "
        f"{(history_refit_result['total_time_s'] / sequential_result['total_time_s']):.3f}x"
    )
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 72)

    return dict(
        base_result=base_result,
        sequential_result=sequential_result,
        history_refit_result=history_refit_result,
        figure_path=figure_path,
        summary_path=summary_path,
    )


def plot_active_cyclic_summary(result, save_path="active_ensemble_change_summary.png"):
    data = result["data"]
    cfg = data["config"]
    mean_map = result["mean_map"]
    var_map = result["var_map"]
    history = result["history"]
    nmse_trace = result["nmse_trace"]
    sensor_locs_init = result["sensor_locs_init"]
    sensor_locs_final = result["sensor_locs_final"]
    warmstart_probe = result.get("warmstart_probe")

    true_energy = np.sum(data["H"], axis=2)
    mean_energy = np.sum(mean_map, axis=2)
    uncertainty_2d = np.mean(var_map, axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    im0 = ax.imshow(true_energy.T, origin="lower", extent=[0, cfg.L - 1, 0, cfg.L - 1], cmap="viridis")
    ax.scatter(sensor_locs_init[:, 0], sensor_locs_init[:, 1], s=10, c="cyan", label="Init")
    if history:
        p = np.array([[h["x"], h["y"]] for h in history], dtype=float)
        ax.plot(p[:, 0], p[:, 1], "w.-", lw=1.0, ms=3, label="Active")
    ax.set_title("True Energy + Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im0, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im1 = ax.imshow(uncertainty_2d.T, origin="lower", extent=[0, cfg.L - 1, 0, cfg.L - 1], cmap="magma")
    ax.scatter(sensor_locs_final[:, 0], sensor_locs_final[:, 1], s=6, c="white", alpha=0.5)
    ax.set_title("Uncertainty (mean over f)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im1, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    im2 = ax.imshow(mean_energy.T, origin="lower", extent=[0, cfg.L - 1, 0, cfg.L - 1], cmap="viridis")
    ax.set_title("Mean Reconstructed Energy")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im2, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    ax.plot(np.arange(len(nmse_trace)), nmse_trace, "o-", color="#1f77b4", lw=2, ms=4)
    if warmstart_probe is not None and len(nmse_trace) > 0:
        step_idx = int(np.clip(warmstart_probe["step"] - 1, 0, len(nmse_trace) - 1))
        ax.axvline(step_idx, color="#d62728", ls="--", lw=1.2, label="warmstart probe")
        ax.scatter(
            [step_idx],
            [warmstart_probe["warmstart_nmse"]],
            color="#d62728",
            s=36,
            zorder=3,
        )
    ax.set_title("NMSE Trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NMSE")
    ax.grid(True, alpha=0.3)
    if warmstart_probe is not None:
        ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    if history:
        centers = np.array([h["center_freq"] for h in history], dtype=int)
        ax.plot(centers, "o-", lw=1.5, ms=3, label="center")
        ax.set_ylim(-0.5, cfg.K - 0.5)
    ax.set_title("Center Frequency (Cyclic Shift)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Frequency index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 2]
    if history:
        omega_mat = np.stack([h["omega_row"] for h in history], axis=0)
        ax.imshow(omega_mat, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title("Partial Observation Masks")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Step")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def run_active_ensemble_cyclic_experiment(
    M_budget=100,
    seed=42,
    M_init=None,
    T=None,
    M_ens=8,
    keep_recent=3,
    d_max=10.0,
    lambda_u=5.0, # 5.0
    lambda_c=2,
    lambda_r=0.08,
    lambda_b=0.05,
    beta_f=0.2,
    cyclic_band_width=7,
    fifo_queue_size=None,
    warmstart_nmse_threshold=0.0,
    quality_weighted_ensemble=True,
    distance_cost_mode="manhattan",
    uav_step_length=1.0,
    uav_move_stride=1,
    ensemble_update_interval=None,
    path_eta=0.75,
    path_uncertainty_cost_mode="inverse",
    quantization_bits=8,
    adaptive_quantization_bits=False,
    high_quantization_bits=8,
    low_quantization_bits=4,
    uncertainty_quantization_threshold=0.5,
    summary_save_path="active_ensemble_change_summary.png",
    solver_backend="cpu",
    solver_device="auto",
    gpu_phi_solver="scipy",
    reconstruction_method="history_refit",
):
    """On-grid UAV active loop with cardinal motion, random warm-up, and path following."""
    cfg = SimConfig(
        full_obs=True,
        R=1,
        M=max(1, int(M_ens)),
        n_grid_samples=0,
        N1=32,
        N2=32,
        L=32,
    )
    data = generate_data(cfg, seed=seed, addShadow=False)

    bounds = ((0, cfg.L), (0, cfg.L))
    K = cfg.K
    rng = np.random.default_rng(seed + 7000)
    grid_points = np.asarray(data["grid_coords"], dtype=float)
    warmup_sample_target = int(max(1, M_ens if M_init is None else M_init))

    print("=" * 72)
    print("On-grid UAV Active Sampling with Cardinal Motion")
    print("=" * 72)
    print(
        f"scene={cfg.N1}x{cfg.N2}, budget={M_budget}, K={K}, warmup_samples={warmup_sample_target}, "
        f"M_ens={M_ens}, step_length={float(uav_step_length):.2f}, "
        f"move_stride={int(max(1, uav_move_stride))}"
    )
    print(
        f"distance_cost_mode={distance_cost_mode}, fifo_queue_size={fifo_queue_size}, "
        f"quality_weighted={quality_weighted_ensemble}, "
        f"ensemble_update_interval={ensemble_update_interval}"
    )
    print(
        f"path objective: (1-eta)T + eta*C, eta={float(path_eta):.2f}, "
        f"uncertainty_cost_mode={path_uncertainty_cost_mode}"
    )
    print("Sampling model: on-grid truth H[x,y,:] + Gaussian noise")

    sensor_locs_t = np.empty((0, 2), dtype=float)
    gamma_t = np.empty((0, cfg.K), dtype=float)
    omega_t = np.empty((0, cfg.K), dtype=np.int32)
    queue_sensor_locs = np.empty((0, 2), dtype=float)
    queue_gamma = np.empty((0, cfg.K), dtype=float)
    queue_omega = np.empty((0, cfg.K), dtype=np.int32)

    history = []
    nmse_trace = []
    action_visit = np.zeros((cfg.N1, cfg.N2, cfg.K), dtype=float)
    last_var_map = np.zeros((cfg.N1, cfg.N2, cfg.K), dtype=float)
    model = None
    fusion_meta = dict(raw_count=0, fused_count=0, compression_ratio=1.0)
    warmstart_probe = None
    queue_dropped_total = 0
    quantization_bits = max(1, int(quantization_bits))
    high_quantization_bits, low_quantization_bits = _validate_quantization_pair(
        high_quantization_bits,
        low_quantization_bits,
    )
    adaptive_quantization_bits = bool(adaptive_quantization_bits)
    if not adaptive_quantization_bits:
        high_quantization_bits = quantization_bits
        low_quantization_bits = quantization_bits
    total_transmitted_bits = 0
    uncertainty_space_trace = []
    expected_quantization_bits_trace = []
    uncertainty_round_stats = []
    ensemble_round_ids = []
    ensemble_true_nmse_trace = []
    ensemble_weighted_obs_nmse_trace = []
    ensemble_mean_obs_nmse_trace = []

    if adaptive_quantization_bits:
        print(
            f"quantization_bits=adaptive(high={high_quantization_bits}, low={low_quantization_bits}), "
            f"uncertainty_threshold={float(uncertainty_quantization_threshold):.3f}, "
            f"lambda_b={float(lambda_b):.3f}"
        )
    else:
        print(
            f"quantization_bits=fixed({int(quantization_bits)}), "
            f"lambda_b={float(lambda_b):.3f}"
        )

    def _select_step_quantization_bits(uncertainty_norm=None, phase="planned_path"):
        return _select_quantization_bits_from_uncertainty(
            uncertainty_norm,
            adaptive_quantization_bits=adaptive_quantization_bits,
            quantization_bits=quantization_bits,
            high_quantization_bits=high_quantization_bits,
            low_quantization_bits=low_quantization_bits,
            uncertainty_quantization_threshold=uncertainty_quantization_threshold,
            phase=phase,
        )

    def _step_bit_penalty(step_quantization_bits):
        if high_quantization_bits <= low_quantization_bits:
            return 0.0
        normalized = (
            float(step_quantization_bits) - float(low_quantization_bits)
        ) / (float(high_quantization_bits) - float(low_quantization_bits))
        return float(lambda_b) * float(np.clip(normalized, 0.0, 1.0))

    def _append_measurement(
        grid_cell,
        omega_row,
        center_freq,
        phase,
        round_idx,
        target_cell,
        acquisition_value=np.nan,
        reward_space=np.nan,
        reward_freq=np.nan,
        reward_total=np.nan,
        uncertainty_value=np.nan,
        keep_ratio_value=np.nan,
        uncertainty_term=np.nan,
        move_term=np.nan,
        redundancy_term=np.nan,
        step_bit_term=np.nan,
        bit_selection_uncertainty=np.nan,
        bit_selection_mode="fixed",
        step_quantization_bits=None,
        path_step_index=1,
        path_step_total=1,
        planned_path_distance=0.0,
        planned_path_uncertainty_distance=0.0,
        planned_path_weighted_cost=0.0,
        executed_move_distance=0.0,
        executed_hops=0,
    ):
        nonlocal sensor_locs_t, gamma_t, omega_t
        nonlocal queue_sensor_locs, queue_gamma, queue_omega, queue_dropped_total
        nonlocal total_transmitted_bits

        grid_cell = np.asarray(grid_cell, dtype=int)
        target_cell = np.asarray(target_cell, dtype=int)
        if step_quantization_bits is None:
            step_quantization_bits = quantization_bits
        step_quantization_bits = max(1, int(step_quantization_bits))
        observed_bands = np.where(np.asarray(omega_row) > 0)[0]
        raw_queue_before = int(queue_sensor_locs.shape[0])
        fused_before = int(fusion_meta["fused_count"])
        step_bits = estimate_transmitted_bits(omega_row, step_quantization_bits)

        loc_row, gamma_row, omega_row_2d, true_spec = measure_on_grid_with_mask(
            data,
            grid_cell,
            omega_row,
            rng,
            quantization_bits=step_quantization_bits,
        )

        sensor_locs_t = np.vstack([sensor_locs_t, loc_row])
        gamma_t = np.vstack([gamma_t, gamma_row])
        omega_t = np.vstack([omega_t, omega_row_2d])

        gx, gy = int(grid_cell[0]), int(grid_cell[1])
        action_visit[gx, gy, observed_bands] += 1.0

        queue_sensor_locs, queue_gamma, queue_omega, dropped_now = _append_fifo_observations(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            loc_row,
            gamma_row,
            omega_row_2d,
            max_len=fifo_queue_size,
        )
        queue_dropped_total += dropped_now
        total_transmitted_bits += int(step_bits)

        history.append(
            dict(
                step=len(history) + 1,
                round=int(round_idx),
                phase=str(phase),
                path_step_index=int(path_step_index),
                path_step_total=int(path_step_total),
                x=float(loc_row[0, 0]),
                y=float(loc_row[0, 1]),
                x_idx=gx,
                y_idx=gy,
                selected_grid_x=int(target_cell[0]),
                selected_grid_y=int(target_cell[1]),
                center_freq=int(center_freq),
                observed_bands=observed_bands.copy(),
                observed_band_count=int(observed_bands.size),
                omega_row=np.asarray(omega_row, dtype=np.int32).copy(),
                quantization_bits=int(step_quantization_bits),
                adaptive_quantization_bits=bool(adaptive_quantization_bits),
                bit_selection_mode=str(bit_selection_mode),
                bit_selection_uncertainty=float(bit_selection_uncertainty),
                uncertainty_quantization_threshold=float(uncertainty_quantization_threshold),
                step_transmitted_bits=int(step_bits),
                cumulative_transmitted_bits=int(total_transmitted_bits),
                acquisition=float(acquisition_value),
                reward_space=float(reward_space),
                reward_freq=float(reward_freq),
                reward_total=float(reward_total),
                uncertainty=float(uncertainty_value),
                keep_ratio=float(keep_ratio_value),
                uncertainty_term=float(uncertainty_term),
                move_term=float(move_term),
                redundancy_term=float(redundancy_term),
                step_bit_term=float(step_bit_term),
                observed_center_value=float(gamma_row[0, center_freq]),
                true_center_value=float(true_spec[center_freq]),
                raw_queue_size_before=raw_queue_before,
                fused_count_before=fused_before,
                raw_queue_size_after=int(queue_sensor_locs.shape[0]),
                fused_count_after=fused_before,
                compression_ratio=float(fusion_meta["compression_ratio"]),
                queue_dropped_total=int(queue_dropped_total),
                queue_dropped_now=int(dropped_now),
                planned_path_distance=float(planned_path_distance),
                planned_path_uncertainty_distance=float(planned_path_uncertainty_distance),
                planned_path_weighted_cost=float(planned_path_weighted_cost),
                executed_move_distance=float(executed_move_distance),
                executed_hops=int(executed_hops),
                uav_step_length=float(uav_step_length),
                uav_move_stride=int(max(1, uav_move_stride)),
                target_distance_mode=str(distance_cost_mode),
            )
        )
        return int(step_bits)

    current_cell = _sample_random_grid_cell(rng, cfg.N1, cfg.N2)
    virtual_target = _sample_random_grid_cell(rng, cfg.N1, cfg.N2)

    warmup_start_idx = len(history)
    while sensor_locs_t.shape[0] < M_budget and sensor_locs_t.shape[0] < warmup_sample_target:
        center_freq = int(rng.integers(0, cfg.K))
        omega_row, _ = build_observe_mask(cfg.K, center_freq, band_width=cyclic_band_width)
        warmup_quantization_bits, warmup_bit_mode = _select_step_quantization_bits(
            uncertainty_norm=None,
            phase="warmup",
        )
        _append_measurement(
            current_cell,
            omega_row,
            center_freq=center_freq,
            phase="warmup",
            round_idx=0,
            target_cell=virtual_target,
            bit_selection_mode=warmup_bit_mode,
            step_quantization_bits=warmup_quantization_bits,
            path_step_index=sensor_locs_t.shape[0] + 1,
            path_step_total=warmup_sample_target,
        )

        if sensor_locs_t.shape[0] >= M_budget or sensor_locs_t.shape[0] >= warmup_sample_target:
            break

        if np.array_equal(current_cell, virtual_target):
            while np.array_equal(current_cell, virtual_target):
                virtual_target = _sample_random_grid_cell(rng, cfg.N1, cfg.N2)
        current_cell = _choose_random_cardinal_step(
            current_cell,
            virtual_target,
            rng,
            cfg.N1,
            cfg.N2,
        )

    n_init_sampled = int(sensor_locs_t.shape[0])
    if n_init_sampled == 0:
        raise ValueError("No on-grid warmup samples were collected.")

    fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
        queue_sensor_locs,
        queue_gamma,
        queue_omega,
        cfg.N1,
        cfg.N2,
    )
    if str(reconstruction_method).strip().lower() == "sequential_incremental":
        init_max_iter = _select_outer_iters(sensor_locs_t.shape[0], warmstart=False)
        model = _reconstruct_model(
            sensor_locs_t,
            gamma_t,
            omega_t,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=data["I_mask"],
            prev_model=None,
            warmstart=False,
            max_iter=init_max_iter,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            reconstruction_method=reconstruction_method,
        )
    else:
        init_max_iter = _select_outer_iters(fusion_meta["fused_count"], warmstart=False)
        model = _reconstruct_model(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=data["I_mask"],
            prev_model=None,
            warmstart=False,
            max_iter=init_max_iter,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            reconstruction_method=reconstruction_method,
        )

    init_nmse = float(
        model.evaluate_reconstruction2(
            model.Sr, model.Phi, data["S"], data["Phi"], drawFlag=False
        )
    )
    nmse_trace.append(init_nmse)
    for idx in range(warmup_start_idx, len(history)):
        history[idx]["nmse"] = float(init_nmse)
        history[idx]["base_nmse"] = float(init_nmse)
        history[idx]["recon_mode"] = "init_on_grid"
        history[idx]["raw_queue_size_after"] = int(queue_sensor_locs.shape[0])
        history[idx]["fused_count_after"] = int(fusion_meta["fused_count"])
        history[idx]["compression_ratio"] = float(fusion_meta["compression_ratio"])
        history[idx]["queue_dropped_total"] = int(queue_dropped_total)
        history[idx]["warmstart_probe_nmse"] = None
        history[idx]["warmstart_probe_gain"] = None

    current_cell = np.asarray(sensor_locs_t[-1], dtype=int)
    if T is None:
        T = max(0, int(M_budget))
    T = int(max(0, T))
    move_stride = int(max(1, uav_move_stride))
    if ensemble_update_interval is None:
        update_interval = None
    else:
        update_interval = int(max(1, ensemble_update_interval))

    print("-" * 72)
    print(
        f"Warmup finished: samples={n_init_sampled}, current=({current_cell[0]},{current_cell[1]}), "
        f"fused={fusion_meta['fused_count']}, init_nmse={init_nmse:.4f}"
    )
    print("-" * 72)

    t = 0
    while t < T and sensor_locs_t.shape[0] < M_budget:
        keep_ratio_t = _adaptive_keep_ratio(
            fusion_meta["fused_count"], early_ratio=0.85, late_ratio=0.75, switch_M=20
        )

        mean_map_t, var_map_t, _, ensemble_info = ensemble_reconstruct_maps(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=data["I_mask"],
            M_ens=M_ens,
            keep_ratio=keep_ratio_t,
            keep_recent=keep_recent,
            seed=seed + 31 * (t + 1),
            base_model=model,
            member_max_iter=_select_outer_iters(fusion_meta["fused_count"], warmstart=True),
            quality_weighted=quality_weighted_ensemble,
            solver_backend=solver_backend,
            solver_device=solver_device,
            gpu_phi_solver=gpu_phi_solver,
            reconstruction_method=reconstruction_method,
        )
        last_var_map = var_map_t
        ensemble_round_ids.append(int(t + 1))
        ensemble_true_nmse_trace.append(_map_nmse(mean_map_t, data["H"]))
        ensemble_weighted_obs_nmse_trace.append(
            float(
                np.sum(
                    np.asarray(ensemble_info["weights"], dtype=float)
                    * np.asarray(ensemble_info["obs_nmse"], dtype=float)
                )
            )
        )
        ensemble_mean_obs_nmse_trace.append(
            _observation_nmse(
                mean_map_t,
                fused_sensor_locs,
                fused_gamma,
                fused_omega,
                cfg.N1,
                cfg.N2,
            )
        )

        normalized_uncertainty_space = _normalize_score_map(np.mean(var_map_t, axis=2))
        bit_cost_map = build_uncertainty_quantization_map(
            normalized_uncertainty_space,
            adaptive_quantization_bits=adaptive_quantization_bits,
            quantization_bits=quantization_bits,
            high_quantization_bits=high_quantization_bits,
            low_quantization_bits=low_quantization_bits,
            uncertainty_quantization_threshold=uncertainty_quantization_threshold,
        )
        acquisition_space, acq_comp = build_acquisition_space(
            var_map_t,
            next_points=grid_points,
            current_loc=current_cell.astype(float),
            sampled_locs=sensor_locs_t,
            lambda_u=lambda_u,
            lambda_c=lambda_c,
            lambda_r=lambda_r,
            lambda_b=lambda_b,
            bit_cost_map=bit_cost_map,
            redundancy_length=5.0,
            distance_cost_mode=distance_cost_mode,
        )
        uncertainty_space_snapshot = np.asarray(acq_comp["uncertainty_space"], dtype=float).copy()
        uncertainty_space_trace.append(uncertainty_space_snapshot)
        expected_quantization_bits_trace.append(np.asarray(bit_cost_map, dtype=float).copy())
        uncertainty_flat = uncertainty_space_snapshot[np.isfinite(uncertainty_space_snapshot)]
        high_bit_ratio = float(
            np.mean(np.asarray(bit_cost_map, dtype=float) >= float(high_quantization_bits))
        )
        if uncertainty_flat.size > 0:
            uncertainty_round_stats.append(
                dict(
                    round=int(t + 1),
                    mean=float(np.mean(uncertainty_flat)),
                    std=float(np.std(uncertainty_flat)),
                    p10=float(np.quantile(uncertainty_flat, 0.10)),
                    p50=float(np.quantile(uncertainty_flat, 0.50)),
                    p90=float(np.quantile(uncertainty_flat, 0.90)),
                    threshold=float(uncertainty_quantization_threshold),
                    high_bit_ratio=high_bit_ratio,
                )
            )

        visited_mask = np.sum(action_visit, axis=2) > 0
        candidate_mask = ~visited_mask
        if not np.any(candidate_mask):
            candidate_mask = np.ones((cfg.N1, cfg.N2), dtype=bool)

        if d_max is not None:
            xx, yy = np.meshgrid(np.arange(cfg.N1), np.arange(cfg.N2), indexing="ij")
            if distance_cost_mode == "manhattan":
                dist_grid = (
                    np.abs(xx - current_cell[0]) + np.abs(yy - current_cell[1])
                ) * float(uav_step_length)
            else:
                dist_grid = np.sqrt((xx - current_cell[0]) ** 2 + (yy - current_cell[1]) ** 2) * float(uav_step_length)
            feasible = dist_grid <= float(d_max)
            if np.any(candidate_mask & feasible):
                candidate_mask = candidate_mask & feasible

        masked_acq = np.where(candidate_mask, acquisition_space, -np.inf)
        if not np.any(np.isfinite(masked_acq)):
            masked_acq = acquisition_space.copy()

        target_flat = int(np.argmax(masked_acq))
        x_idx, y_idx = np.unravel_index(target_flat, (cfg.N1, cfg.N2))
        target_cell = np.array([x_idx, y_idx], dtype=int)
        acq_value = float(masked_acq[x_idx, y_idx])

        freq_uncertainty = var_map_t[x_idx, y_idx, :].astype(float)
        freq_penalty = action_visit[x_idx, y_idx, :].astype(float)
        if np.max(freq_penalty) > 0:
            freq_penalty = freq_penalty / np.max(freq_penalty)
        freq_acq = freq_uncertainty - float(beta_f) * freq_penalty
        center_freq = int(np.argmax(freq_acq))
        freq_score_chosen = float(freq_acq[center_freq])

        uncertainty_term = float(lambda_u * acq_comp["uncertainty_space"][x_idx, y_idx])
        move_term = float(lambda_c * acq_comp["move_cost"][x_idx, y_idx])
        redundancy_term = float(lambda_r * acq_comp["redundancy"][x_idx, y_idx])
        bit_term = float(lambda_b * acq_comp["bit_cost"][x_idx, y_idx])
        bit_selection_uncertainty = float(acq_comp["uncertainty_space"][x_idx, y_idx])
        step_quantization_bits, bit_selection_mode = _select_step_quantization_bits(
            uncertainty_norm=bit_selection_uncertainty,
            phase="planned_path",
        )
        step_bit_term = _step_bit_penalty(step_quantization_bits)
        reward_total = float(acq_value + freq_score_chosen - step_bit_term)

        path_cells, path_distance, path_uncertainty_distance, path_weighted_cost = _plan_cardinal_path(
            current_cell,
            target_cell,
            acq_comp["uncertainty_space"],
            step_length=uav_step_length,
            path_eta=path_eta,
            uncertainty_cost_mode=path_uncertainty_cost_mode,
        )
        executed_path, executed_hops = _build_executed_path(path_cells, move_stride=move_stride)
        omega_row, _ = build_observe_mask(cfg.K, center_freq, band_width=cyclic_band_width)

        segment_start = len(history)
        segment_samples = 0
        segment_distance = 0.0
        segment_bits = 0
        segment_replanned_early = False
        segment_locs = []
        segment_gamma = []
        segment_omega = []
        for seq, (cell, hop_count) in enumerate(zip(executed_path, executed_hops), start=1):
            if sensor_locs_t.shape[0] >= M_budget:
                break

            cell_idx = np.asarray(cell, dtype=int)
            move_distance = float(hop_count) * float(uav_step_length)
            step_bits = _append_measurement(
                cell_idx,
                omega_row,
                center_freq=center_freq,
                phase="planned_path",
                round_idx=t + 1,
                target_cell=target_cell,
                acquisition_value=acq_value,
                reward_space=acq_value,
                reward_freq=freq_score_chosen,
                reward_total=reward_total,
                uncertainty_value=float(var_map_t[x_idx, y_idx, center_freq]),
                keep_ratio_value=keep_ratio_t,
                uncertainty_term=uncertainty_term,
                move_term=move_term,
                redundancy_term=redundancy_term,
                step_bit_term=step_bit_term,
                bit_selection_uncertainty=bit_selection_uncertainty,
                bit_selection_mode=bit_selection_mode,
                step_quantization_bits=step_quantization_bits,
                path_step_index=seq,
                path_step_total=len(executed_path),
                planned_path_distance=path_distance,
                planned_path_uncertainty_distance=path_uncertainty_distance,
                planned_path_weighted_cost=path_weighted_cost,
                executed_move_distance=move_distance,
                executed_hops=hop_count,
            )
            segment_locs.append(sensor_locs_t[-1].copy())
            segment_gamma.append(gamma_t[-1].copy())
            segment_omega.append(omega_t[-1].copy())
            current_cell = cell_idx.copy()
            segment_samples += 1
            segment_distance += move_distance
            segment_bits += int(step_bits)

            if update_interval is not None and segment_samples >= update_interval and seq < len(executed_path):
                segment_replanned_early = True
                break

        if segment_samples == 0:
            print(f"[{t + 1:02d}/{T:02d}] no sample added due to budget limit.")
            break

        fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            cfg.N1,
            cfg.N2,
        )
        if str(reconstruction_method).strip().lower() == "sequential_incremental":
            refit_max_iter = _select_outer_iters(sensor_locs_t.shape[0], warmstart=True)
            segment_locs_arr = np.asarray(segment_locs, dtype=float).reshape(-1, 2)
            segment_gamma_arr = np.asarray(segment_gamma, dtype=float).reshape(-1, cfg.K)
            segment_omega_arr = np.asarray(segment_omega, dtype=np.int32).reshape(-1, cfg.K)
            t_recon_start = time.perf_counter()
            model.add_measurements(
                segment_locs_arr,
                segment_gamma_arr,
                segment_omega_arr,
                n_outer_iter=refit_max_iter,
                max_svt_iter=20,
                debugFlag=False,
            )
            recon_elapsed_s = float(time.perf_counter() - t_recon_start)
            model_obs_locs = sensor_locs_t
            model_gamma = gamma_t
            model_omega = omega_t
        else:
            refit_max_iter = _select_outer_iters(fusion_meta["fused_count"], warmstart=True)
            t_recon_start = time.perf_counter()
            model = _reconstruct_model(
                fused_sensor_locs,
                fused_gamma,
                fused_omega,
                cfg=cfg,
                grid_points=grid_points,
                bounds=bounds,
                I_mask=data["I_mask"],
                prev_model=model,
                warmstart=True,
                max_iter=refit_max_iter,
                solver_backend=solver_backend,
                solver_device=solver_device,
                gpu_phi_solver=gpu_phi_solver,
                reconstruction_method=reconstruction_method,
            )
            recon_elapsed_s = float(time.perf_counter() - t_recon_start)
            model_obs_locs = fused_sensor_locs
            model_gamma = fused_gamma
            model_omega = fused_omega

        base_nmse_t = float(
            model.evaluate_reconstruction2(
                model.Sr, model.Phi, data["S"], data["Phi"], drawFlag=False
            )
        )
        nmse_t = base_nmse_t
        warmstart_probe_nmse = None
        warmstart_probe_gain = None
        recon_mode = "warmstart"

        if warmstart_probe is None and base_nmse_t < warmstart_nmse_threshold:
            probe_model = _reconstruct_model(
                model_obs_locs,
                model_gamma,
                model_omega,
                cfg=cfg,
                grid_points=grid_points,
                bounds=bounds,
                I_mask=data["I_mask"],
                prev_model=model,
                warmstart=True,
                max_iter=max(2, refit_max_iter),
                solver_backend=solver_backend,
                solver_device=solver_device,
                gpu_phi_solver=gpu_phi_solver,
                reconstruction_method=reconstruction_method,
            )
            warmstart_probe_nmse = float(
                probe_model.evaluate_reconstruction2(
                    probe_model.Sr,
                    probe_model.Phi,
                    data["S"],
                    data["Phi"],
                    drawFlag=False,
                )
            )
            warmstart_probe_gain = float(base_nmse_t - warmstart_probe_nmse)
            warmstart_probe = dict(
                step=t + 1,
                base_nmse=float(base_nmse_t),
                warmstart_nmse=float(warmstart_probe_nmse),
                improvement=float(warmstart_probe_gain),
                raw_queue_size=int(queue_sensor_locs.shape[0]),
                fused_count=int(fusion_meta["fused_count"]),
                compression_ratio=float(fusion_meta["compression_ratio"]),
            )
            if warmstart_probe_nmse <= base_nmse_t:
                model = probe_model
                nmse_t = warmstart_probe_nmse
                recon_mode = "warmstart_probe_accepted"
            else:
                recon_mode = "warmstart_probe_rejected"

        nmse_trace.append(nmse_t)

        for idx in range(segment_start, len(history)):
            history[idx]["nmse"] = float(nmse_t)
            history[idx]["base_nmse"] = float(base_nmse_t)
            history[idx]["recon_mode"] = recon_mode
            history[idx]["reconstruction_time_s"] = float(recon_elapsed_s)
            history[idx]["raw_queue_size_after"] = int(queue_sensor_locs.shape[0])
            history[idx]["fused_count_after"] = int(fusion_meta["fused_count"])
            history[idx]["compression_ratio"] = float(fusion_meta["compression_ratio"])
            history[idx]["queue_dropped_total"] = int(queue_dropped_total)
            history[idx]["warmstart_probe_nmse"] = (
                None if warmstart_probe_nmse is None else float(warmstart_probe_nmse)
            )
            history[idx]["warmstart_probe_gain"] = (
                None if warmstart_probe_gain is None else float(warmstart_probe_gain)
            )

        weight_summary = np.asarray(ensemble_info["weights"], dtype=float)
        warmstart_msg = ""
        if warmstart_probe_nmse is not None:
            warmstart_msg = f" warm_probe={warmstart_probe_nmse:.4f}"
        print(
            f"[{t + 1:02d}/{T:02d}] "
            f"target=({x_idx:02d},{y_idx:02d},f={center_freq:02d}) "
            f"path_steps={segment_samples}/{len(executed_path)} "
            f"path_dist={path_distance:.2f} "
            f"segment_dist={segment_distance:.2f} "
            f"segment_bits={segment_bits} total_bits={total_transmitted_bits} "
            f"path_unc_cost={path_uncertainty_distance:.4f} "
            f"path_cost={path_weighted_cost:.4f} "
            f"reward_space={acq_value:.4f} reward_freq={freq_score_chosen:.4f} reward_total={reward_total:.4f} "
            f"(u={uncertainty_term:.4f}, -c={move_term:.4f}, -r={redundancy_term:.4f}, "
            f"-b_pred={bit_term:.4f}, -b_step={step_bit_term:.4f}) "
            f"bit_mode={bit_selection_mode} qbits={step_quantization_bits} "
            f"keep={keep_ratio_t:.2f} "
            f"queue={queue_sensor_locs.shape[0]} fused={fusion_meta['fused_count']} "
            f"comp={fusion_meta['compression_ratio']:.2f} "
            f"recon_time={recon_elapsed_s:.3f}s "
            f"wmax={np.max(weight_summary):.3f} mode={recon_mode} "
            f"replan_early={segment_replanned_early} "
            f"nmse={nmse_t:.4f}{warmstart_msg}"
        )
        t += 1

    mean_map = model.get_current_map()
    var_map = last_var_map
    if nmse_trace:
        final_nmse = float(nmse_trace[-1])
    else:
        final_nmse = float(
            model.evaluate_reconstruction2(
                model.Sr, model.Phi, data["S"], data["Phi"], drawFlag=False
            )
        )
        nmse_trace.append(final_nmse)
    uncertainty_distribution_summary = summarize_uncertainty_distribution(
        uncertainty_space_trace,
        threshold=uncertainty_quantization_threshold,
    )

    result = dict(
        data=data,
        mean_map=mean_map,
        var_map=var_map,
        history=history,
        nmse_trace=nmse_trace,
        sensor_locs_init=sensor_locs_t[:n_init_sampled].copy(),
        sensor_locs_final=sensor_locs_t.copy(),
        gamma_final=gamma_t.copy(),
        omega_final=omega_t.copy(),
        final_nmse=final_nmse,
        warmstart_probe=warmstart_probe,
        queue_dropped_total=int(queue_dropped_total),
        raw_queue_size=int(queue_sensor_locs.shape[0]),
        fused_count=int(fusion_meta["fused_count"]),
        compression_ratio=float(fusion_meta["compression_ratio"]),
        fifo_queue_size=fifo_queue_size,
        distance_cost_mode=distance_cost_mode,
        warmup_sample_target=int(warmup_sample_target),
        adaptive_quantization_bits=bool(adaptive_quantization_bits),
        quantization_bits=(None if adaptive_quantization_bits else int(quantization_bits)),
        high_quantization_bits=int(high_quantization_bits),
        low_quantization_bits=int(low_quantization_bits),
        uncertainty_quantization_threshold=float(uncertainty_quantization_threshold),
        total_transmitted_bits=int(total_transmitted_bits),
        lambda_b=float(lambda_b),
        uav_step_length=float(uav_step_length),
        uav_move_stride=int(move_stride),
        ensemble_update_interval=update_interval,
        path_eta=float(path_eta),
        path_uncertainty_cost_mode=str(path_uncertainty_cost_mode),
        solver_backend=str(_resolve_iibtd_backend(solver_backend, n_sources=cfg.R, solver_device=solver_device)),
        solver_device=str(_resolve_iibtd_device(solver_device)),
        gpu_phi_solver=str(gpu_phi_solver),
        reconstruction_method=str(reconstruction_method),
        uncertainty_space_trace=uncertainty_space_trace,
        expected_quantization_bits_trace=expected_quantization_bits_trace,
        uncertainty_round_stats=uncertainty_round_stats,
        uncertainty_distribution_summary=uncertainty_distribution_summary,
        ensemble_round_ids=np.asarray(ensemble_round_ids, dtype=int),
        ensemble_true_nmse_trace=np.asarray(ensemble_true_nmse_trace, dtype=float),
        ensemble_weighted_obs_nmse_trace=np.asarray(ensemble_weighted_obs_nmse_trace, dtype=float),
        ensemble_mean_obs_nmse_trace=np.asarray(ensemble_mean_obs_nmse_trace, dtype=float),
    )

    if summary_save_path:
        plot_active_cyclic_summary(result, save_path=summary_save_path)

    print("-" * 72)
    print(f"Final samples: {sensor_locs_t.shape[0]}")
    print(
        f"Final reconstruction rows: raw_queue={queue_sensor_locs.shape[0]}, "
        f"fused={fusion_meta['fused_count']}, "
        f"compression={fusion_meta['compression_ratio']:.2f}"
    )
    print(f"Total transmitted bits: {int(total_transmitted_bits)}")
    print(f"Final NMSE: {final_nmse:.4f}")
    if warmstart_probe is not None:
        print(
            f"Warm-start probe at step {warmstart_probe['step']}: "
            f"{warmstart_probe['base_nmse']:.4f} -> {warmstart_probe['warmstart_nmse']:.4f}"
        )
    if summary_save_path:
        print(f"Saved: {summary_save_path}")
    print("=" * 72)
    return result


def extract_active_step_nmse_trace(result):
    history = result.get("history", [])
    active_history = [item for item in history if str(item.get("phase", "")) != "warmup"]
    if not active_history:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    # Re-index to active-only steps so plots and summaries match their labels.
    steps = np.arange(1, len(active_history) + 1, dtype=int)
    nmse_values = np.asarray([float(item["nmse"]) for item in active_history], dtype=float)
    return steps, nmse_values


def extract_active_step_distance_trace(result):
    history = result.get("history", [])
    active_indices = [
        idx for idx, item in enumerate(history)
        if str(item.get("phase", "")) != "warmup"
    ]
    if not active_indices:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    steps = np.arange(1, len(active_indices) + 1, dtype=int)
    cumulative_distance = np.zeros(len(active_indices), dtype=float)
    running = 0.0
    step_length = float(result.get("uav_step_length", 1.0))
    prev_loc = None
    for idx, hist_idx in enumerate(active_indices):
        item = history[hist_idx]
        move_distance = item.get("executed_move_distance")
        if move_distance is None:
            loc = np.array([float(item["x"]), float(item["y"])], dtype=float)
            if prev_loc is None:
                if hist_idx > 0:
                    prev_loc = np.array(
                        [float(history[hist_idx - 1]["x"]), float(history[hist_idx - 1]["y"])],
                        dtype=float,
                    )
                else:
                    prev_loc = loc.copy()
            move_distance = step_length * float(np.sum(np.abs(loc - prev_loc)))
            prev_loc = loc
        running += float(move_distance)
        cumulative_distance[idx] = running
    return steps, cumulative_distance


def extract_active_step_bit_trace(result):
    history = result.get("history", [])
    active_history = [item for item in history if str(item.get("phase", "")) != "warmup"]
    if not active_history:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    steps = np.arange(1, len(active_history) + 1, dtype=int)
    step_bits = np.asarray(
        [float(item.get("step_transmitted_bits", 0.0)) for item in active_history],
        dtype=float,
    )
    cumulative_bits = np.cumsum(step_bits)
    return steps, cumulative_bits


def plot_normalized_uncertainty_diagnostics(result, save_path):
    """Visualize the normalized uncertainty distribution used by the bit-cost strategy."""
    uncertainty_maps = result.get("uncertainty_space_trace", [])
    if not uncertainty_maps:
        return None

    stack = np.stack([np.asarray(item, dtype=float) for item in uncertainty_maps], axis=0)
    flat = stack[np.isfinite(stack)]
    if flat.size == 0:
        return None

    threshold = float(result.get("uncertainty_quantization_threshold", 0.5))
    mean_uncertainty_map = np.mean(stack, axis=0)
    summary = result.get("uncertainty_distribution_summary") or {}
    active_history = [
        item for item in result.get("history", [])
        if str(item.get("phase", "")) != "warmup"
    ]
    selected_round_ids = np.asarray(
        [
            int(item.get("round", idx + 1))
            for idx, item in enumerate(active_history)
            if np.isfinite(float(item.get("bit_selection_uncertainty", np.nan)))
        ],
        dtype=int,
    )
    selected_uncertainty = np.asarray(
        [
            float(item.get("bit_selection_uncertainty", np.nan))
            for item in active_history
            if np.isfinite(float(item.get("bit_selection_uncertainty", np.nan)))
        ],
        dtype=float,
    )
    round_stats = result.get("uncertainty_round_stats", [])
    round_ids = np.asarray([int(item["round"]) for item in round_stats], dtype=int)
    high_bit_ratio = np.asarray([float(item["high_bit_ratio"]) for item in round_stats], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(flat, bins=bins, color="#457b9d", alpha=0.82, density=True, label="All grid cells")
    if selected_uncertainty.size > 0:
        ax.hist(
            selected_uncertainty,
            bins=bins,
            color="#e76f51",
            alpha=0.55,
            density=True,
            label="Selected targets",
        )
    ax.axvline(threshold, color="#d62728", ls="--", lw=1.6, label=f"threshold={threshold:.2f}")
    ax.set_title("Normalized Uncertainty Histogram")
    ax.set_xlabel("Normalized uncertainty")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    sorted_vals = np.sort(flat)
    ecdf = np.arange(1, sorted_vals.size + 1, dtype=float) / float(sorted_vals.size)
    ax.plot(sorted_vals, ecdf, color="#1d3557", lw=2.0, label="ECDF")
    ax.axvline(threshold, color="#d62728", ls="--", lw=1.6, label=f"threshold={threshold:.2f}")
    above_ratio = float(summary.get("above_threshold_ratio", np.mean(flat >= threshold)))
    ax.set_title("Normalized Uncertainty ECDF")
    ax.set_xlabel("Normalized uncertainty")
    ax.set_ylabel("Cumulative probability")
    ax.text(
        0.03,
        0.08,
        f"above-threshold ratio={above_ratio:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    im = ax.imshow(mean_uncertainty_map.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Mean Normalized Uncertainty Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, shrink=0.82)

    ax = axes[1, 1]
    if round_ids.size > 0:
        ax.plot(round_ids, high_bit_ratio, color="#2a9d8f", lw=2.0, marker="o", ms=4, label="High-bit candidate ratio")
    if selected_uncertainty.size > 0:
        ax.plot(
            selected_round_ids,
            selected_uncertainty,
            color="#e76f51",
            lw=1.8,
            marker="s",
            ms=3.8,
            label="Selected target uncertainty",
        )
    ax.axhline(threshold, color="#d62728", ls="--", lw=1.4, label=f"threshold={threshold:.2f}")
    ax.set_title("Threshold Activity Over Active Rounds")
    ax.set_xlabel("Active round")
    ax.set_ylabel("Ratio / uncertainty")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_uav_comparison(results, save_path, specs=None):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    if specs is None:
        default_high_bits = 8
        default_low_bits = 4
        default_threshold = 0.5
        for spec in EXPERIMENT_SPECS:
            if spec["key"] in results and spec.get("use_bit_cost", False):
                result = results[spec["key"]]
                default_high_bits = int(result.get("high_quantization_bits", default_high_bits))
                default_low_bits = int(result.get("low_quantization_bits", default_low_bits))
                default_threshold = float(
                    result.get("uncertainty_quantization_threshold", default_threshold)
                )
                break
        specs = resolve_experiment_specs(
            [spec["key"] for spec in EXPERIMENT_SPECS if spec["key"] in results],
            high_quantization_bits=default_high_bits,
            low_quantization_bits=default_low_bits,
            uncertainty_quantization_threshold=default_threshold,
        )

    fig, (ax_nmse, ax_dist, ax_bits) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    for idx, spec in enumerate(specs):
        plot_label = _format_plot_label(spec)
        steps_nmse, nmse_values = extract_active_step_nmse_trace(results[spec["key"]])
        if nmse_values.size > 0:
            marker_step = max(1, nmse_values.size // 12)
            ax_nmse.plot(
                steps_nmse,
                nmse_values,
                color=spec["color"],
                marker=spec["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                markevery=(idx % marker_step, marker_step),
                label=plot_label,
            )

        steps_dist, distance_values = extract_active_step_distance_trace(results[spec["key"]])
        if distance_values.size > 0:
            marker_step = max(1, distance_values.size // 12)
            ax_dist.plot(
                steps_dist,
                distance_values,
                color=spec["color"],
                marker=spec["marker"],
                linestyle="--",
                linewidth=1.8,
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.1,
                alpha=0.9,
                markevery=(idx % marker_step, marker_step),
                label=plot_label,
            )

        steps_bits, bit_values = extract_active_step_bit_trace(results[spec["key"]])
        if bit_values.size > 0:
            marker_step = max(1, bit_values.size // 12)
            ax_bits.plot(
                steps_bits,
                bit_values,
                color=spec["color"],
                marker=spec["marker"],
                linestyle="-.",
                linewidth=1.8,
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.1,
                alpha=0.95,
                markevery=(idx % marker_step, marker_step),
                label=plot_label,
            )

    ax_nmse.set_title("UAV Strategy Comparison: Active-Phase NMSE vs Step")
    ax_nmse.set_ylabel("NMSE")
    ax_nmse.grid(True, alpha=0.3)
    ax_nmse.legend(loc="best", fontsize=9)

    ax_dist.set_title("UAV Strategy Comparison: Active-Phase Manhattan Distance vs Step")
    ax_dist.set_ylabel("Cumulative Manhattan Distance")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.legend(loc="best", fontsize=9)

    ax_bits.set_title("Active-Phase Cumulative Transmitted Bits vs Step")
    ax_bits.set_xlabel("Active Sampling Step")
    ax_bits.set_ylabel("Cumulative Bits")
    ax_bits.grid(True, alpha=0.3)
    ax_bits.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_uav_trace_gif(result, save_path, frame_stride=1, fps=3, final_hold_frames=8):
    history = result.get("history", [])
    if not history:
        return None

    data = result["data"]
    cfg = data["config"]
    true_energy = np.sum(data["H"], axis=2)
    locs = np.asarray([[float(item["x"]), float(item["y"])] for item in history], dtype=float)
    targets = np.asarray(
        [[float(item["selected_grid_x"]), float(item["selected_grid_y"])] for item in history],
        dtype=float,
    )
    phases = [str(item.get("phase", "")) for item in history]
    rounds = np.asarray([int(item.get("round", 0)) for item in history], dtype=int)
    freqs = np.asarray([int(item.get("center_freq", 0)) for item in history], dtype=int)
    nmse_vals = np.asarray([float(item.get("nmse", np.nan)) for item in history], dtype=float)

    frame_indices = list(range(0, len(history), max(1, int(frame_stride))))
    if frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)

    frames = []
    for frame_idx in frame_indices:
        so_far = locs[: frame_idx + 1]
        target_so_far = targets[: frame_idx + 1]
        phase_so_far = phases[: frame_idx + 1]
        round_so_far = rounds[: frame_idx + 1]

        target_change_idx = []
        for idx in range(frame_idx + 1):
            if idx == 0:
                target_change_idx.append(idx)
                continue
            same_target = np.array_equal(target_so_far[idx], target_so_far[idx - 1])
            same_round = round_so_far[idx] == round_so_far[idx - 1]
            if (not same_target) or (not same_round):
                target_change_idx.append(idx)

        target_hist = target_so_far[np.asarray(target_change_idx, dtype=int)]
        warmup_mask = np.asarray([phase == "warmup" for phase in phase_so_far], dtype=bool)
        active_mask = ~warmup_mask

        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        ax.imshow(
            true_energy.T,
            origin="lower",
            extent=[0, cfg.L - 1, 0, cfg.L - 1],
            cmap="viridis",
        )

        if so_far.shape[0] > 1:
            ax.plot(so_far[:, 0], so_far[:, 1], color="white", lw=2.0, alpha=0.9, label="UAV path")
        if np.any(warmup_mask):
            ax.scatter(
                so_far[warmup_mask, 0],
                so_far[warmup_mask, 1],
                s=24,
                c="#4cc9f0",
                edgecolors="black",
                linewidths=0.4,
                label="Warm-up samples",
            )
        if np.any(active_mask):
            ax.scatter(
                so_far[active_mask, 0],
                so_far[active_mask, 1],
                s=22,
                c="#ffd166",
                edgecolors="black",
                linewidths=0.4,
                label="Planned-path samples",
            )

        if target_hist.shape[0] > 0:
            ax.scatter(
                target_hist[:, 0],
                target_hist[:, 1],
                s=56,
                c="#ff6b6b",
                marker="x",
                linewidths=1.8,
                alpha=0.55,
                label="Past targets",
            )

        current_loc = so_far[-1]
        current_target = target_so_far[-1]
        ax.scatter(
            [current_target[0]],
            [current_target[1]],
            s=120,
            c="#ff2d55",
            marker="x",
            linewidths=2.6,
            label="Current target",
        )
        ax.scatter(
            [current_loc[0]],
            [current_loc[1]],
            s=120,
            c="#ffffff",
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            label="Current UAV",
        )
        ax.plot(
            [current_loc[0], current_target[0]],
            [current_loc[1], current_target[1]],
            color="#ffb703",
            ls="--",
            lw=1.4,
            alpha=0.9,
        )

        step_id = int(history[frame_idx]["step"])
        ax.set_title(
            f"UAV Trace | step={step_id}/{len(history)} | "
            f"target=({int(current_target[0])},{int(current_target[1])},f={freqs[frame_idx]}) | "
            f"nmse={nmse_vals[frame_idx]:.4f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-0.5, cfg.N1 - 0.5)
        ax.set_ylim(-0.5, cfg.N2 - 0.5)
        ax.grid(True, alpha=0.15, color="white")
        ax.legend(loc="upper right", fontsize=8)

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))

    if not frames:
        return None

    frames.extend([frames[-1].copy() for _ in range(max(0, int(final_hold_frames)))])
    duration_ms = int(round(1000.0 / max(1, int(fps))))
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return save_path


def save_uav_comparison_summary(results, common_kwargs, save_path, specs=None):
    if specs is None:
        default_high_bits = int(common_kwargs.get("high_quantization_bits", 8))
        default_low_bits = int(common_kwargs.get("low_quantization_bits", 4))
        default_threshold = float(common_kwargs.get("uncertainty_quantization_threshold", 0.5))
        specs = resolve_experiment_specs(
            [spec["key"] for spec in EXPERIMENT_SPECS if spec["key"] in results],
            high_quantization_bits=default_high_bits,
            low_quantization_bits=default_low_bits,
            uncertainty_quantization_threshold=default_threshold,
        )

    payload = dict(
        common_kwargs=dict(common_kwargs),
        selected_strategies=[spec["key"] for spec in specs],
        comparison_assumption="Comparison is run over the selected strategies among mean(var), mean(var)+Manhattan distance cost, and mean(var)+Manhattan distance cost+bits cost; the first two strategies always use fixed 8-bit quantization, while the bits-cost strategy maps the normalized spatial uncertainty at each candidate grid to configurable high/low quantization bit-widths via an uncertainty threshold and uses that expected bit-width map as the bit-cost term; lambda_r is fixed to 0 and ensemble weighting is disabled.",
        path_planning_objective="Shortest path on the grid minimizing (1-eta)T(X) + etaC(X), where C(X) is the accumulated uncertainty-derived path cost.",
        distance_definition="Cumulative Manhattan distance over active-phase samples only, starting from the last warm-up location.",
        transmission_bit_definition="Per-step transmitted bits are counted as observed_band_count x quantization_bits; quantizer side metadata is not included.",
        experiments=[],
    )

    for spec in specs:
        history = results[spec["key"]].get("history", [])
        warmup_steps = sum(1 for item in history if str(item.get("phase", "")) == "warmup")
        active_rounds = sorted({
            int(item.get("round", 0))
            for item in history
            if str(item.get("phase", "")) != "warmup"
        })
        steps_nmse, nmse_values = extract_active_step_nmse_trace(results[spec["key"]])
        _, distance_values = extract_active_step_distance_trace(results[spec["key"]])
        uncertainty_distribution_summary = results[spec["key"]].get("uncertainty_distribution_summary")
        payload["experiments"].append(
            dict(
                key=spec["key"],
                label=spec["label"],
                lambda_c=float(spec["lambda_c"]),
                distance_cost_mode="manhattan",
                uncertainty_mode="mean(var)",
                quality_weighted=False,
                lambda_r=0.0,
                use_bit_cost=bool(spec.get("use_bit_cost", False)),
                adaptive_quantization_bits=bool(results[spec["key"]].get("adaptive_quantization_bits", False)),
                uav_step_length=float(results[spec["key"]].get("uav_step_length", 1.0)),
                uav_move_stride=int(results[spec["key"]].get("uav_move_stride", 1)),
                ensemble_update_interval=results[spec["key"]].get("ensemble_update_interval"),
                path_eta=float(results[spec["key"]].get("path_eta", 0.75)),
                path_uncertainty_cost_mode=str(results[spec["key"]].get("path_uncertainty_cost_mode", "inverse")),
                quantization_bits=results[spec["key"]].get("quantization_bits"),
                high_quantization_bits=int(results[spec["key"]].get("high_quantization_bits", 8)),
                low_quantization_bits=int(results[spec["key"]].get("low_quantization_bits", 8)),
                uncertainty_quantization_threshold=float(
                    results[spec["key"]].get("uncertainty_quantization_threshold", 0.5)
                ),
                lambda_b=float(results[spec["key"]].get("lambda_b", 0.0)),
                history_steps_total=int(len(history)),
                warmup_steps=int(warmup_steps),
                steps=int(steps_nmse.size),
                active_steps=int(steps_nmse.size),
                planning_rounds=int(len(active_rounds)),
                final_nmse=(
                    float(nmse_values[-1])
                    if nmse_values.size > 0
                    else float(results[spec["key"]].get("final_nmse"))
                ),
                final_cumulative_manhattan_distance=(
                    float(distance_values[-1]) if distance_values.size > 0 else None
                ),
                final_total_transmitted_bits=int(results[spec["key"]].get("total_transmitted_bits", 0)),
                uncertainty_distribution_summary=uncertainty_distribution_summary,
            )
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_uav_comparison(common_kwargs, output_dir, strategies=None, reconstruction_method="history_refit"):
    os.makedirs(output_dir, exist_ok=True)
    high_quantization_bits = int(common_kwargs.get("high_quantization_bits", 8))
    low_quantization_bits = int(common_kwargs.get("low_quantization_bits", 4))
    uncertainty_quantization_threshold = float(
        common_kwargs.get("uncertainty_quantization_threshold", 0.5)
    )
    specs = resolve_experiment_specs(
        strategies,
        high_quantization_bits=high_quantization_bits,
        low_quantization_bits=low_quantization_bits,
        uncertainty_quantization_threshold=uncertainty_quantization_threshold,
    )

    results = {}
    gif_paths = {}
    uncertainty_diag_paths = {}
    for spec in specs:
        print("-" * 72)
        print(f"Running {spec['label']}")
        spec_lambda_b = float(common_kwargs.get("lambda_b", 0.0)) if spec.get("use_bit_cost", False) else 0.0
        spec_kwargs = dict(common_kwargs)
        spec_kwargs["lambda_c"] = float(spec["lambda_c"])
        spec_kwargs["lambda_r"] = 0.0
        spec_kwargs["lambda_b"] = spec_lambda_b
        spec_kwargs["quantization_bits"] = int(spec["high_quantization_bits"])
        spec_kwargs["adaptive_quantization_bits"] = bool(spec.get("adaptive_quantization_bits", False))
        spec_kwargs["high_quantization_bits"] = int(spec["high_quantization_bits"])
        spec_kwargs["low_quantization_bits"] = int(spec["low_quantization_bits"])
        spec_kwargs["uncertainty_quantization_threshold"] = float(
            spec["uncertainty_quantization_threshold"]
        )
        spec_kwargs["quality_weighted_ensemble"] = False
        spec_kwargs["distance_cost_mode"] = "manhattan"
        spec_kwargs["summary_save_path"] = None
        spec_kwargs["reconstruction_method"] = reconstruction_method
        results[spec["key"]] = run_active_ensemble_cyclic_experiment(**spec_kwargs)
        gif_path = os.path.join(output_dir, f"{spec['key']}_uav_trace.gif")
        save_uav_trace_gif(results[spec["key"]], gif_path)
        gif_paths[spec["key"]] = gif_path
        if bool(results[spec["key"]].get("adaptive_quantization_bits", False)):
            diag_path = os.path.join(
                output_dir,
                f"{spec['key']}_normalized_uncertainty_diagnostics.png",
            )
            if plot_normalized_uncertainty_diagnostics(results[spec["key"]], diag_path) is not None:
                uncertainty_diag_paths[spec["key"]] = diag_path

    method_tag = str(reconstruction_method).strip().lower()
    figure_path = os.path.join(output_dir, f"uav_mean_manhattan_nmse_distance_vs_step_{method_tag}.png")
    summary_path = os.path.join(output_dir, f"uav_mean_manhattan_nmse_distance_vs_step_{method_tag}.json")

    plot_uav_comparison(results, figure_path, specs=specs)
    save_uav_comparison_summary(results, common_kwargs, summary_path, specs=specs)

    print("=" * 72)
    for spec in specs:
        steps_nmse, nmse_values = extract_active_step_nmse_trace(results[spec["key"]])
        _, distance_values = extract_active_step_distance_trace(results[spec["key"]])
        final_nmse = float(nmse_values[-1]) if nmse_values.size > 0 else np.nan
        final_distance = float(distance_values[-1]) if distance_values.size > 0 else np.nan
        final_bits = int(results[spec["key"]].get("total_transmitted_bits", 0))
        print(
            f"{spec['label']}: "
            f"steps={steps_nmse.size}, final_nmse={final_nmse:.4f}, "
            f"final_distance={final_distance:.4f}, final_bits={final_bits}, "
            f"reconstruction_method={method_tag}"
        )
        print(f"Saved GIF: {gif_paths[spec['key']]}")
        if spec["key"] in uncertainty_diag_paths:
            print(f"Saved uncertainty diagnostics: {uncertainty_diag_paths[spec['key']]}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 72)

    return dict(
        results=results,
        gif_paths=gif_paths,
        uncertainty_diag_paths=uncertainty_diag_paths,
        figure_path=figure_path,
        summary_path=summary_path,
    )


def _collect_strategy_metrics(comparison_payload, specs):
    rows = []
    results = comparison_payload["results"]
    for spec in specs:
        result = results[spec["key"]]
        steps_nmse, nmse_values = extract_active_step_nmse_trace(result)
        _, distance_values = extract_active_step_distance_trace(result)
        _, bit_values = extract_active_step_bit_trace(result)
        rows.append(
            dict(
                key=spec["key"],
                label=spec["label"],
                reconstruction_method=str(result.get("reconstruction_method", "unknown")),
                active_steps=int(steps_nmse.size),
                final_nmse=float(nmse_values[-1]) if nmse_values.size > 0 else np.nan,
                final_distance=float(distance_values[-1]) if distance_values.size > 0 else np.nan,
                final_bits=float(bit_values[-1]) if bit_values.size > 0 else 0.0,
                total_time_s=float(np.nansum([item.get("reconstruction_time_s", 0.0) for item in result.get("history", [])])),
            )
        )
    return rows


def plot_strategy_reconstruction_method_summary(summary_rows, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    methods = ["history_refit", "sequential_incremental"]
    method_labels = {
        "history_refit": "MAPPO-style history refit",
        "sequential_incremental": "Sequential incremental",
    }
    strategy_keys = [spec["key"] for spec in EXPERIMENT_SPECS]
    strategy_labels = {
        spec["key"]: _format_plot_label(spec) for spec in EXPERIMENT_SPECS
    }
    color_map = {
        "history_refit": "#d62728",
        "sequential_incremental": "#1f77b4",
    }

    metric_names = ["final_nmse", "total_time_s", "final_distance", "final_bits"]
    metric_titles = {
        "final_nmse": "Final NMSE",
        "total_time_s": "Total Reconstruction Time (s)",
        "final_distance": "Final Cumulative Manhattan Distance",
        "final_bits": "Final Cumulative Transmitted Bits",
    }

    lookup = {
        (row["key"], row["reconstruction_method"]): row
        for row in summary_rows
    }
    x = np.arange(len(strategy_keys), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for ax, metric_name in zip(axes, metric_names):
        for method_idx, method in enumerate(methods):
            values = [
                float(lookup.get((key, method), {}).get(metric_name, np.nan))
                for key in strategy_keys
            ]
            offset = (method_idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=method_labels[method],
                color=color_map[method],
                alpha=0.88,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{value:.3f}" if metric_name != "final_bits" else f"{int(round(value))}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )
        ax.set_title(metric_titles[metric_name])
        ax.set_xticks(x)
        ax.set_xticklabels([strategy_labels[key] for key in strategy_keys], rotation=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_strategy_reconstruction_method_comparison(
    common_kwargs,
    output_dir,
    strategies=None,
):
    os.makedirs(output_dir, exist_ok=True)
    high_quantization_bits = int(common_kwargs.get("high_quantization_bits", 8))
    low_quantization_bits = int(common_kwargs.get("low_quantization_bits", 4))
    uncertainty_quantization_threshold = float(
        common_kwargs.get("uncertainty_quantization_threshold", 0.5)
    )
    specs = resolve_experiment_specs(
        strategies,
        high_quantization_bits=high_quantization_bits,
        low_quantization_bits=low_quantization_bits,
        uncertainty_quantization_threshold=uncertainty_quantization_threshold,
    )

    comparison_payloads = {}
    summary_rows = []
    for reconstruction_method in ("history_refit", "sequential_incremental"):
        method_dir = os.path.join(output_dir, reconstruction_method)
        payload = run_uav_comparison(
            common_kwargs,
            method_dir,
            strategies=[spec["key"] for spec in specs],
            reconstruction_method=reconstruction_method,
        )
        comparison_payloads[reconstruction_method] = payload
        summary_rows.extend(_collect_strategy_metrics(payload, specs))

    summary_figure_path = os.path.join(output_dir, "strategy_reconstruction_method_summary.png")
    summary_json_path = os.path.join(output_dir, "strategy_reconstruction_method_summary.json")
    plot_strategy_reconstruction_method_summary(summary_rows, summary_figure_path)

    payload = dict(
        common_kwargs={
            key: value
            for key, value in dict(common_kwargs).items()
            if key not in {"reconstruction_method"}
        },
        strategies=[spec["key"] for spec in specs],
        summary_rows=summary_rows,
        method_outputs={
            method: dict(
                figure_path=comparison_payloads[method]["figure_path"],
                summary_path=comparison_payloads[method]["summary_path"],
            )
            for method in comparison_payloads
        },
        summary_figure_path=summary_figure_path,
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Strategy × Reconstruction Method Comparison")
    for row in summary_rows:
        print(
            f"{row['label']} | {row['reconstruction_method']}: "
            f"nmse={row['final_nmse']:.4f}, time={row['total_time_s']:.3f}s, "
            f"distance={row['final_distance']:.3f}, bits={int(round(row['final_bits']))}"
        )
    print(f"Saved summary figure: {summary_figure_path}")
    print(f"Saved summary json: {summary_json_path}")
    print("=" * 72)

    return dict(
        comparison_payloads=comparison_payloads,
        summary_rows=summary_rows,
        summary_figure_path=summary_figure_path,
        summary_json_path=summary_json_path,
    )


def plot_ensemble_nmse_proxy_comparison(results, save_path, specs=None):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    if specs is None:
        specs = [spec for spec in EXPERIMENT_SPECS if spec["key"] in results]

    fig, axes = plt.subplots(len(specs), 3, figsize=(16, 4.8 * len(specs)), squeeze=False)

    for row_idx, spec in enumerate(specs):
        result = results[spec["key"]]
        rounds = np.asarray(result.get("ensemble_round_ids", []), dtype=int)
        true_nmse = np.asarray(result.get("ensemble_true_nmse_trace", []), dtype=float)
        weighted_proxy = np.asarray(result.get("ensemble_weighted_obs_nmse_trace", []), dtype=float)
        mean_obs_proxy = np.asarray(result.get("ensemble_mean_obs_nmse_trace", []), dtype=float)

        weighted_stats = _proxy_error_summary(weighted_proxy, true_nmse)
        mean_obs_stats = _proxy_error_summary(mean_obs_proxy, true_nmse)

        ax = axes[row_idx, 0]
        ax.plot(rounds, true_nmse, color="#111111", lw=2.2, marker="o", ms=4, label="True full-map NMSE")
        ax.plot(
            rounds,
            weighted_proxy,
            color="#d62728",
            lw=1.9,
            marker="s",
            ms=3.8,
            label="Ensemble weighted obs-NMSE",
        )
        ax.plot(
            rounds,
            mean_obs_proxy,
            color="#1f77b4",
            lw=1.9,
            marker="^",
            ms=3.8,
            label="Ensemble mean-map obs-NMSE",
        )
        ax.set_title(f"{_format_plot_label(spec)}: Proxy vs True NMSE")
        ax.set_xlabel("Active round")
        ax.set_ylabel("NMSE")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        for col_idx, (proxy_values, title, color, stats) in enumerate(
            (
                (weighted_proxy, "Weighted obs-NMSE proxy", "#d62728", weighted_stats),
                (mean_obs_proxy, "Mean-map obs-NMSE proxy", "#1f77b4", mean_obs_stats),
            ),
            start=1,
        ):
            ax = axes[row_idx, col_idx]
            mask = np.isfinite(true_nmse) & np.isfinite(proxy_values)
            true_plot = true_nmse[mask]
            proxy_plot = proxy_values[mask]
            if true_plot.size > 0:
                ax.scatter(true_plot, proxy_plot, color=color, alpha=0.85, s=28)
                diag_min = float(min(np.min(true_plot), np.min(proxy_plot)))
                diag_max = float(max(np.max(true_plot), np.max(proxy_plot)))
                if diag_max <= diag_min + 1e-12:
                    diag_max = diag_min + 1.0
                ax.plot([diag_min, diag_max], [diag_min, diag_max], color="#666666", ls="--", lw=1.2)
                ax.set_xlim(diag_min, diag_max)
                ax.set_ylim(diag_min, diag_max)
            ax.set_title(f"{_format_plot_label(spec)}: {title}")
            ax.set_xlabel("True full-map NMSE")
            ax.set_ylabel("Proxy value")
            ax.grid(True, alpha=0.25)
            stats_text = "\n".join(
                [
                    f"corr={stats['corr']:.3f}" if np.isfinite(stats["corr"]) else "corr=nan",
                    f"MAE={stats['mae']:.4f}" if np.isfinite(stats["mae"]) else "MAE=nan",
                    f"RMSE={stats['rmse']:.4f}" if np.isfinite(stats["rmse"]) else "RMSE=nan",
                    f"bias={stats['bias']:.4f}" if np.isfinite(stats["bias"]) else "bias=nan",
                    f"ratio={stats['mean_ratio']:.3f}" if np.isfinite(stats["mean_ratio"]) else "ratio=nan",
                ]
            )
            ax.text(
                0.04,
                0.96,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_ensemble_nmse_proxy_comparison(
    common_kwargs,
    output_dir,
    strategies=None,
):
    os.makedirs(output_dir, exist_ok=True)
    method_dir = os.path.join(output_dir, "history_refit")
    comparison_payload = run_uav_comparison(
        common_kwargs,
        method_dir,
        strategies=strategies,
        reconstruction_method="history_refit",
    )

    high_quantization_bits = int(common_kwargs.get("high_quantization_bits", 8))
    low_quantization_bits = int(common_kwargs.get("low_quantization_bits", 4))
    uncertainty_quantization_threshold = float(
        common_kwargs.get("uncertainty_quantization_threshold", 0.5)
    )
    specs = resolve_experiment_specs(
        strategies,
        high_quantization_bits=high_quantization_bits,
        low_quantization_bits=low_quantization_bits,
        uncertainty_quantization_threshold=uncertainty_quantization_threshold,
    )

    figure_path = os.path.join(output_dir, "ensemble_nmse_proxy_compare.png")
    summary_json_path = os.path.join(output_dir, "ensemble_nmse_proxy_compare.json")
    plot_ensemble_nmse_proxy_comparison(
        comparison_payload["results"],
        figure_path,
        specs=specs,
    )

    summary_rows = []
    for spec in specs:
        result = comparison_payload["results"][spec["key"]]
        true_nmse = np.asarray(result.get("ensemble_true_nmse_trace", []), dtype=float)
        weighted_proxy = np.asarray(result.get("ensemble_weighted_obs_nmse_trace", []), dtype=float)
        mean_obs_proxy = np.asarray(result.get("ensemble_mean_obs_nmse_trace", []), dtype=float)
        summary_rows.append(
            dict(
                key=spec["key"],
                label=spec["label"],
                rounds=int(true_nmse.size),
                final_true_nmse=float(true_nmse[-1]) if true_nmse.size > 0 else np.nan,
                final_weighted_obs_nmse=float(weighted_proxy[-1]) if weighted_proxy.size > 0 else np.nan,
                final_mean_obs_nmse=float(mean_obs_proxy[-1]) if mean_obs_proxy.size > 0 else np.nan,
                weighted_obs_proxy=_proxy_error_summary(weighted_proxy, true_nmse),
                mean_map_obs_proxy=_proxy_error_summary(mean_obs_proxy, true_nmse),
            )
        )

    payload = dict(
        common_kwargs=dict(common_kwargs),
        reconstruction_method="history_refit",
        strategies=[spec["key"] for spec in specs],
        summary_rows=summary_rows,
        history_refit_outputs=dict(
            figure_path=comparison_payload["figure_path"],
            summary_path=comparison_payload["summary_path"],
        ),
        proxy_figure_path=figure_path,
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Ensemble Proxy vs True NMSE")
    for row in summary_rows:
        weighted = row["weighted_obs_proxy"]
        mean_obs = row["mean_map_obs_proxy"]
        print(
            f"{row['label']}: "
            f"weighted corr={weighted['corr']:.3f}, MAE={weighted['mae']:.4f}; "
            f"mean-map corr={mean_obs['corr']:.3f}, MAE={mean_obs['mae']:.4f}"
        )
    print(f"Saved proxy figure: {figure_path}")
    print(f"Saved proxy summary: {summary_json_path}")
    print("=" * 72)

    return dict(
        comparison_payload=comparison_payload,
        summary_rows=summary_rows,
        figure_path=figure_path,
        summary_json_path=summary_json_path,
    )


def plot_ensemble_mean_surrogate_comparison(summary_rows, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(summary_rows), 4, figsize=(18, 4.8 * len(summary_rows)), squeeze=False)
    for row_idx, row in enumerate(summary_rows):
        full_energy = np.asarray(row["full_energy"], dtype=float)
        ensemble_energy = np.asarray(row["ensemble_energy"], dtype=float)
        diff_energy = np.asarray(row["abs_energy_diff"], dtype=float)
        vmin = float(min(np.min(full_energy), np.min(ensemble_energy)))
        vmax = float(max(np.max(full_energy), np.max(ensemble_energy)))

        ax = axes[row_idx, 0]
        im = ax.imshow(full_energy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{row['label']}\nFull-data refit map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, shrink=0.82)

        ax = axes[row_idx, 1]
        im = ax.imshow(ensemble_energy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{row['label']}\nEnsemble mean map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, shrink=0.82)

        ax = axes[row_idx, 2]
        im = ax.imshow(diff_energy.T, origin="lower", cmap="magma")
        ax.set_title(f"{row['label']}\n|Energy difference|")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, shrink=0.82)

        ax = axes[row_idx, 3]
        ax.axis("off")
        lines = [
            row["label"],
            "",
            f"replacement NMSE: {row['replacement_nmse']:.6f}",
            f"map corr: {row['map_corr']:.4f}" if np.isfinite(row["map_corr"]) else "map corr: nan",
            "",
            f"full true NMSE: {row['full_true_nmse']:.6f}",
            f"ensemble true NMSE: {row['ensemble_true_nmse']:.6f}",
            f"true-NMSE gap: {row['true_nmse_gap']:.6f}",
            "",
            f"full obs-NMSE: {row['full_obs_nmse']:.6f}",
            f"ensemble obs-NMSE: {row['ensemble_obs_nmse']:.6f}",
            f"obs-NMSE gap: {row['ensemble_obs_nmse'] - row['full_obs_nmse']:.6f}",
            "",
            f"fused rows: {row['fused_count']}",
            f"keep_ratio: {row['keep_ratio']:.3f}",
            f"M_ens: {row['M_ens']}",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#cccccc"),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_ensemble_mean_surrogate_comparison(
    common_kwargs,
    output_dir,
    strategies=None,
):
    os.makedirs(output_dir, exist_ok=True)
    method_dir = os.path.join(output_dir, "history_refit")
    comparison_payload = run_uav_comparison(
        common_kwargs,
        method_dir,
        strategies=strategies,
        reconstruction_method="history_refit",
    )

    high_quantization_bits = int(common_kwargs.get("high_quantization_bits", 8))
    low_quantization_bits = int(common_kwargs.get("low_quantization_bits", 4))
    uncertainty_quantization_threshold = float(
        common_kwargs.get("uncertainty_quantization_threshold", 0.5)
    )
    specs = resolve_experiment_specs(
        strategies,
        high_quantization_bits=high_quantization_bits,
        low_quantization_bits=low_quantization_bits,
        uncertainty_quantization_threshold=uncertainty_quantization_threshold,
    )

    summary_rows = []
    for spec in specs:
        result = comparison_payload["results"][spec["key"]]
        data = result["data"]
        cfg = data["config"]
        bounds = ((0, cfg.L), (0, cfg.L))
        grid_points = np.asarray(data["grid_coords"], dtype=float)
        I_mask = np.asarray(data["I_mask"], dtype=bool)
        sensor_locs = np.asarray(result["sensor_locs_final"], dtype=float)
        gamma = np.asarray(result["gamma_final"], dtype=float)
        omega = np.asarray(result["omega_final"], dtype=np.int32)

        fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
            sensor_locs,
            gamma,
            omega,
            cfg.N1,
            cfg.N2,
        )
        full_max_iter = _select_outer_iters(fusion_meta["fused_count"], warmstart=False)
        full_model = _fit_reconstruction_model(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=I_mask,
            prev_model=None,
            warmstart=False,
            max_iter=full_max_iter,
            solver_backend=common_kwargs.get("solver_backend", "cpu"),
            solver_device=common_kwargs.get("solver_device", "auto"),
            gpu_phi_solver=common_kwargs.get("gpu_phi_solver", "scipy"),
        )
        full_map = np.asarray(full_model.get_current_map(), dtype=float)

        keep_ratio = _adaptive_keep_ratio(
            fusion_meta["fused_count"], early_ratio=0.85, late_ratio=0.75, switch_M=20
        )
        ensemble_mean_map, _, _, _ = ensemble_reconstruct_maps(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=grid_points,
            bounds=bounds,
            I_mask=I_mask,
            M_ens=int(common_kwargs.get("M_ens", 8)),
            keep_ratio=keep_ratio,
            keep_recent=int(common_kwargs.get("keep_recent", 3)),
            seed=int(common_kwargs.get("seed", 42)) + 5000,
            base_model=full_model,
            member_max_iter=_select_outer_iters(fusion_meta["fused_count"], warmstart=True),
            quality_weighted=False,
            solver_backend=common_kwargs.get("solver_backend", "cpu"),
            solver_device=common_kwargs.get("solver_device", "auto"),
            gpu_phi_solver=common_kwargs.get("gpu_phi_solver", "scipy"),
            reconstruction_method="history_refit",
        )
        ensemble_mean_map = np.asarray(ensemble_mean_map, dtype=float)

        summary_rows.append(
            dict(
                key=spec["key"],
                label=spec["label"],
                fused_count=int(fusion_meta["fused_count"]),
                keep_ratio=float(keep_ratio),
                M_ens=int(common_kwargs.get("M_ens", 8)),
                replacement_nmse=_map_nmse(ensemble_mean_map, full_map),
                map_corr=_map_correlation(ensemble_mean_map, full_map),
                full_true_nmse=_map_nmse(full_map, data["H"]),
                ensemble_true_nmse=_map_nmse(ensemble_mean_map, data["H"]),
                true_nmse_gap=_map_nmse(ensemble_mean_map, data["H"]) - _map_nmse(full_map, data["H"]),
                full_obs_nmse=_observation_nmse(
                    full_map,
                    fused_sensor_locs,
                    fused_gamma,
                    fused_omega,
                    cfg.N1,
                    cfg.N2,
                ),
                ensemble_obs_nmse=_observation_nmse(
                    ensemble_mean_map,
                    fused_sensor_locs,
                    fused_gamma,
                    fused_omega,
                    cfg.N1,
                    cfg.N2,
                ),
                full_energy=np.sum(full_map, axis=2),
                ensemble_energy=np.sum(ensemble_mean_map, axis=2),
                abs_energy_diff=np.abs(np.sum(ensemble_mean_map - full_map, axis=2)),
            )
        )

    figure_path = os.path.join(output_dir, "ensemble_mean_surrogate_compare.png")
    summary_json_path = os.path.join(output_dir, "ensemble_mean_surrogate_compare.json")
    plot_ensemble_mean_surrogate_comparison(summary_rows, figure_path)

    json_rows = []
    for row in summary_rows:
        json_rows.append(
            dict(
                key=row["key"],
                label=row["label"],
                fused_count=row["fused_count"],
                keep_ratio=row["keep_ratio"],
                M_ens=row["M_ens"],
                replacement_nmse=float(row["replacement_nmse"]),
                map_corr=float(row["map_corr"]) if np.isfinite(row["map_corr"]) else np.nan,
                full_true_nmse=float(row["full_true_nmse"]),
                ensemble_true_nmse=float(row["ensemble_true_nmse"]),
                true_nmse_gap=float(row["true_nmse_gap"]),
                full_obs_nmse=float(row["full_obs_nmse"]),
                ensemble_obs_nmse=float(row["ensemble_obs_nmse"]),
            )
        )

    payload = dict(
        common_kwargs=dict(common_kwargs),
        reconstruction_method="history_refit",
        strategies=[spec["key"] for spec in specs],
        summary_rows=json_rows,
        history_refit_outputs=dict(
            figure_path=comparison_payload["figure_path"],
            summary_path=comparison_payload["summary_path"],
        ),
        surrogate_figure_path=figure_path,
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Ensemble Mean Map vs Full-Data Refit Map")
    for row in json_rows:
        print(
            f"{row['label']}: replacement_nmse={row['replacement_nmse']:.6f}, "
            f"map_corr={row['map_corr']:.4f}, "
            f"true_nmse_gap={row['true_nmse_gap']:.6f}"
        )
    print(f"Saved surrogate figure: {figure_path}")
    print(f"Saved surrogate summary: {summary_json_path}")
    print("=" * 72)

    return dict(
        comparison_payload=comparison_payload,
        summary_rows=json_rows,
        figure_path=figure_path,
        summary_json_path=summary_json_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="On-grid 32x32 UAV comparison with configurable strategy sets over mean(var), distance cost, and bits cost."
    )
    parser.add_argument(
        "--analysis-mode",
        choices=[
            "strategies",
            "reconstruction_compare",
            "strategy_reconstruction_compare",
            "ensemble_nmse_proxy_compare",
            "ensemble_mean_surrogate_compare",
        ],
        default="strategy_reconstruction_compare",
        help="strategies: compare acquisition strategies; reconstruction_compare: compare sequential incremental vs MAPPO-style history-refit on one fixed UAV sampling path; strategy_reconstruction_compare: run all selected strategies under both reconstruction methods; ensemble_nmse_proxy_compare: compare ensemble-derived proxy values against true full-map NMSE under history-refit only; ensemble_mean_surrogate_compare: compare ensemble mean map directly against the full-data refit radio map.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[spec["key"] for spec in EXPERIMENT_SPECS],
        default=[spec["key"] for spec in EXPERIMENT_SPECS],
        help="Strategies to run and plot, in the requested order.",
    )
    parser.add_argument("--M-budget", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--M-init", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--M-ens", type=int, default=10)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=7.0)
    parser.add_argument("--lambda-u", type=float, default=4.0)
    parser.add_argument("--lambda-b", type=float, default=0.05)
    parser.add_argument("--beta-f", type=float, default=0.2)
    parser.add_argument(
        "--high-quantization-bits",
        type=int,
        default=8,
        help="High-bit quantization used by the uncertainty-aware bits-cost strategy.",
    )
    parser.add_argument(
        "--low-quantization-bits",
        "--quantization-bits",
        dest="low_quantization_bits",
        type=int,
        default=4,
        help="Low-bit quantization used by the uncertainty-aware bits-cost strategy; --quantization-bits is kept as an alias.",
    )
    parser.add_argument(
        "--uncertainty-quantization-threshold",
        type=float,
        default=0.75,
        help="Normalized uncertainty threshold for switching between high/low quantization bits in the bits-cost strategy.",
    )
    parser.add_argument("--cyclic-band-width", type=int, default=6)
    parser.add_argument(
        "--fifo-queue-size",
        type=int,
        default=None,
        help="Maximum number of recent observations kept for reconstruction. Omit or set <= 0 to disable FIFO truncation.",
    )
    parser.add_argument("--warmstart-nmse-threshold", type=float, default=0.5)
    parser.add_argument("--uav-step-length", type=float, default=1.0)
    parser.add_argument(
        "--uav-move-stride",
        type=int,
        default=3,
        help="How many grid hops the UAV executes between two measurement waypoints along a planned path.",
    )
    parser.add_argument(
        "--ensemble-update-interval",
        type=int,
        default=2,
        help="Recompute ensemble and replan after this many executed path waypoints. Omit or set <= 0 to only update after reaching the current target.",
    )
    parser.add_argument("--path-eta", type=float, default=1)
    parser.add_argument(
        "--path-uncertainty-cost-mode",
        choices=["inverse", "one_minus"],
        default="inverse",
    )
    parser.add_argument(
        "--solver-backend",
        choices=["auto", "cpu", "gpu"],
        default="gpu",
        help="II-BTD backend used inside this script.",
    )
    parser.add_argument(
        "--solver-device",
        type=str,
        default="cuda",
        help="Device used by the GPU II-BTD backend.",
    )
    parser.add_argument(
        "--gpu-phi-solver",
        choices=["scipy", "pgd"],
        default="scipy",
        help="Phi solver used by the GPU II-BTD backend.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fifo_queue_size = args.fifo_queue_size
    if fifo_queue_size is not None and fifo_queue_size <= 0:
        fifo_queue_size = None
    ensemble_update_interval = args.ensemble_update_interval
    if ensemble_update_interval is not None and ensemble_update_interval <= 0:
        ensemble_update_interval = None
    high_quantization_bits, low_quantization_bits = _validate_quantization_pair(
        args.high_quantization_bits,
        args.low_quantization_bits,
    )
    uncertainty_quantization_threshold = float(
        np.clip(args.uncertainty_quantization_threshold, 0.0, 1.0)
    )

    common_kwargs = dict(
        M_budget=args.M_budget,
        seed=args.seed,
        M_init=args.M_init,
        T=args.T,
        M_ens=args.M_ens,
        keep_recent=args.keep_recent,
        d_max=args.d_max,
        lambda_u=args.lambda_u,
        lambda_b=args.lambda_b,
        beta_f=args.beta_f,
        high_quantization_bits=int(high_quantization_bits),
        low_quantization_bits=int(low_quantization_bits),
        uncertainty_quantization_threshold=uncertainty_quantization_threshold,
        cyclic_band_width=args.cyclic_band_width,
        fifo_queue_size=fifo_queue_size,
        warmstart_nmse_threshold=args.warmstart_nmse_threshold,
        uav_step_length=args.uav_step_length,
        uav_move_stride=max(1, int(args.uav_move_stride)),
        ensemble_update_interval=ensemble_update_interval,
        path_eta=args.path_eta,
        path_uncertainty_cost_mode=args.path_uncertainty_cost_mode,
        solver_backend=args.solver_backend,
        solver_device=args.solver_device,
        gpu_phi_solver=args.gpu_phi_solver,
    )
    if args.analysis_mode == "reconstruction_compare":
        run_reconstruction_method_comparison(
            common_kwargs,
            args.output_dir,
            solver_backend=args.solver_backend,
            solver_device=args.solver_device,
            gpu_phi_solver=args.gpu_phi_solver,
        )
    elif args.analysis_mode == "strategy_reconstruction_compare":
        run_strategy_reconstruction_method_comparison(
            common_kwargs,
            args.output_dir,
            strategies=args.strategies,
        )
    elif args.analysis_mode == "ensemble_nmse_proxy_compare":
        run_ensemble_nmse_proxy_comparison(
            common_kwargs,
            args.output_dir,
            strategies=args.strategies,
        )
    elif args.analysis_mode == "ensemble_mean_surrogate_compare":
        run_ensemble_mean_surrogate_comparison(
            common_kwargs,
            args.output_dir,
            strategies=args.strategies,
        )
    else:
        run_uav_comparison(
            common_kwargs,
            args.output_dir,
            strategies=args.strategies,
            reconstruction_method="history_refit",
        )


if __name__ == "__main__":
    main()

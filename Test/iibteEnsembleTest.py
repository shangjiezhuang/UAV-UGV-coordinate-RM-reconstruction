import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from spectrumMapTensorGen import SimConfig, generate_data, nmse
except ModuleNotFoundError:
    from Test.spectrumMapTensorGen import SimConfig, generate_data, nmse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from IIBTD.IIBTD_Optimized import II_BTD_Optimized


def sort_sensors_by_trajectory(locs, trajectory_type="tsp"):
    """Sort points to emulate a UAV flight path."""
    M = len(locs)
    if M <= 1:
        return locs.copy(), np.arange(M, dtype=int)

    if trajectory_type == "tsp":
        visited = np.zeros(M, dtype=bool)
        order = [0]
        visited[0] = True
        for _ in range(M - 1):
            cur = order[-1]
            d = np.linalg.norm(locs - locs[cur], axis=1)
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            order.append(nxt)
            visited[nxt] = True
        order = np.asarray(order, dtype=int)
    elif trajectory_type == "snake":
        y_sort = np.argsort(locs[:, 1])
        n_rows = int(np.sqrt(M))
        per_row = max(1, M // max(1, n_rows))
        order = []
        for i in range(n_rows + 1):
            s = i * per_row
            e = min((i + 1) * per_row, M)
            if s >= e:
                continue
            row = y_sort[s:e]
            row = row[np.argsort(locs[row, 0])]
            if i % 2 == 1:
                row = row[::-1]
            order.extend(row.tolist())
        order = np.asarray(order[:M], dtype=int)
    elif trajectory_type == "angle":
        c = locs.mean(axis=0)
        diff = locs - c
        ang = np.arctan2(diff[:, 1], diff[:, 0])
        rad = np.linalg.norm(diff, axis=1)
        order = np.lexsort((rad, ang))
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")
    return locs[order], order


def _smooth_2d_average(field, passes=1):
    out = field.astype(float, copy=True)
    for _ in range(max(0, int(passes))):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
            pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0
    return out


def _apply_random_init(model, cfg, rng, sr_scale=0.1, smooth_passes=1):
    """Random non-negative initialization for Phi and Sr."""
    phi = rng.random((cfg.R, cfg.K)) + 1e-6
    phi = phi * (cfg.K / np.sum(phi, axis=1, keepdims=True))

    sr = rng.random((cfg.R, cfg.N1, cfg.N2)) * sr_scale
    for r in range(cfg.R):
        sr[r] = _smooth_2d_average(sr[r], passes=smooth_passes)

    model.Phi = np.maximum(phi, 1e-10)
    model.Sr = np.maximum(sr, 1e-10)
    model.H_hat = np.einsum("rxy,rk->xyk", model.Sr, model.Phi)


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
    for name in ("Theta", "Phi", "Sr", "H_hat"):
        if hasattr(src_model, name):
            setattr(dst_model, name, np.array(getattr(src_model, name), copy=True))


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
):
    """Fit one reconstruction model, optionally warm-started from prev_model."""
    model = II_BTD_Optimized(
        n_sources=cfg.R,
        grid_size=(cfg.N1, cfg.N2),
        mu=1.2,
        nu=1.5,
        max_iter=max(1, int(max_iter)),
        kernel_bandwidth=0.46,
        warmstart=bool(warmstart),
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
):
    """Run base warm-started ensemble II-BTD and return mean/variance maps."""

    M_total = sensor_locs.shape[0]
    if M_total == 0:
        raise ValueError("sensor_locs is empty.")

    if base_model is None:
        base_model = _fit_reconstruction_model(
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
        )

    maps = []
    obs_nmse_scores = []
    for m in range(max(1, int(M_ens))):
        rng_m = np.random.default_rng(seed + 1000 + m)
        idx = _sample_ensemble_indices(M_total, keep_ratio, keep_recent, rng_m)

        model = _fit_reconstruction_model(
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
    action_visit,
    lambda_u=1.0,
    lambda_c=0.15,
    lambda_r=0.08,
    redundancy_length=5.0,
):
    """Build spatial acquisition score from uncertainty/cost/redundancy/priority."""
    N1, N2, _ = var_map.shape
    uncertainty_space = np.max(var_map, axis=2)

    dist_move = np.linalg.norm(next_points - current_loc[np.newaxis, :], axis=1).reshape(N1, N2)
    move_cost = dist_move / (np.max(dist_move) + 1e-9)

    if sampled_locs.shape[0] > 0:
        diff = next_points[:, np.newaxis, :] - sampled_locs[np.newaxis, :, :]
        min_dist = np.min(np.linalg.norm(diff, axis=2), axis=1).reshape(N1, N2)
        redundancy = np.exp(-min_dist / max(1e-6, redundancy_length))
    else:
        redundancy = np.zeros((N1, N2), dtype=float)

    repeat_map = np.sum(action_visit, axis=2)
    if np.max(repeat_map) > 0:
        repeat_map = repeat_map / np.max(repeat_map)
    priority_map = 1.0 - repeat_map

    acquisition_space = (
        lambda_u * uncertainty_space
        - lambda_c * move_cost
        - lambda_r * redundancy
    )
    components = dict(
        uncertainty_space=uncertainty_space,
        move_cost=move_cost,
        redundancy=redundancy,
        repeat_map=repeat_map,
        priority_map=priority_map,
    )
    return acquisition_space, components


def quantize_to_8bit(values):
    """Uniform 8-bit quantization with de-quantized float output."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy(), {"min": 0.0, "max": 0.0, "levels": 256}

    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max <= v_min + 1e-12:
        return np.full_like(arr, v_min, dtype=float), {"min": v_min, "max": v_max, "levels": 256}

    norm = (arr - v_min) / (v_max - v_min)
    q_uint8 = np.round(norm * 255.0).astype(np.uint8)
    dq = (q_uint8.astype(float) / 255.0) * (v_max - v_min) + v_min
    return dq, {"min": v_min, "max": v_max, "levels": 256}


def cluster_sensor_locs_to_integer_grid(sensor_locs, N1, N2):
    """
    Cluster continuous sensor locations by integer grid.
    Example: x in [0,1), y in [0,1) -> grid (0,0).
    """
    locs = np.asarray(sensor_locs, dtype=float)
    grid_xy = np.floor(locs).astype(int)
    grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, N1 - 1)
    grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, N2 - 1)

    grid_to_sensor_indices = {}
    for idx, (gx, gy) in enumerate(grid_xy):
        key = (int(gx), int(gy))
        grid_to_sensor_indices.setdefault(key, []).append(int(idx))
    return grid_xy, grid_to_sensor_indices


def build_clustered_grid_acquisition(acquisition_space, sensor_grid_xy, sensor_candidate_mask, N1, N2):
    """
    Aggregate per-sensor score into integer-grid score:
    grid_score(i,j) = sum of scores of sensors located in grid (i,j).
    """
    grid_sum = np.zeros((N1, N2), dtype=float)
    grid_count = np.zeros((N1, N2), dtype=int)
    grid_acq = np.full((N1, N2), -np.inf, dtype=float)

    if sensor_grid_xy.shape[0] == 0 or not np.any(sensor_candidate_mask):
        return grid_acq, grid_sum, grid_count

    idx = np.where(sensor_candidate_mask)[0]
    gx = sensor_grid_xy[idx, 0]
    gy = sensor_grid_xy[idx, 1]
    sensor_scores = acquisition_space[gx, gy]

    np.add.at(grid_sum, (gx, gy), sensor_scores)
    np.add.at(grid_count, (gx, gy), 1)

    valid = grid_count > 0
    grid_acq[valid] = grid_sum[valid]
    return grid_acq, grid_sum, grid_count


def select_next_location_and_center_freq(
    var_map,
    acquisition_space,
    next_points,
    sensor_locs,
    current_loc,
    action_visit,
    d_max=None,
    candidate_mask=None,
    sampled_locs=None,
    score_interp="bilinear",
    beta_f=0.2,
):
    """Select next location from waypoint set and center frequency."""
    N1, N2, _ = var_map.shape
    if candidate_mask is None:
        candidate_mask = np.ones((N1, N2), dtype=bool)

    waypoints = np.asarray(sensor_locs, dtype=float)
    if waypoints.shape[0] == 0:
        masked = np.where(candidate_mask, acquisition_space, -np.inf)
        flat_idx = int(np.argmax(masked))
        x_idx, y_idx = np.unravel_index(flat_idx, (N1, N2))
        x_loc, y_loc = float(x_idx), float(y_idx)
        acq_value = float(masked[x_idx, y_idx])
    else:
        # Project acquisition map onto waypoint set.
        wp_x = np.clip(waypoints[:, 0], 0.0, N1 - 1.0)
        wp_y = np.clip(waypoints[:, 1], 0.0, N2 - 1.0)
        wp_x_idx = np.clip(np.round(wp_x).astype(int), 0, N1 - 1)
        wp_y_idx = np.clip(np.round(wp_y).astype(int), 0, N2 - 1)

        if score_interp == "nearest":
            wp_scores = acquisition_space[wp_x_idx, wp_y_idx]
        else:
            x0 = np.floor(wp_x).astype(int)
            y0 = np.floor(wp_y).astype(int)
            x1 = np.clip(x0 + 1, 0, N1 - 1)
            y1 = np.clip(y0 + 1, 0, N2 - 1)
            dx = wp_x - x0
            dy = wp_y - y0
            v00 = acquisition_space[x0, y0]
            v10 = acquisition_space[x1, y0]
            v01 = acquisition_space[x0, y1]
            v11 = acquisition_space[x1, y1]
            wp_scores = (
                (1.0 - dx) * (1.0 - dy) * v00
                + dx * (1.0 - dy) * v10
                + (1.0 - dx) * dy * v01
                + dx * dy * v11
            )

        wp_candidate = candidate_mask[wp_x_idx, wp_y_idx]
        if d_max is None:
            wp_reachable = np.ones(waypoints.shape[0], dtype=bool)
        else:
            wp_reachable = np.linalg.norm(waypoints - current_loc[np.newaxis, :], axis=1) <= float(d_max)
        wp_feasible = wp_candidate & wp_reachable
        if not np.any(wp_feasible):
            wp_feasible = wp_candidate if np.any(wp_candidate) else np.ones(waypoints.shape[0], dtype=bool)

        if sampled_locs is not None and sampled_locs.shape[0] > 0:
            diff = waypoints[:, np.newaxis, :] - sampled_locs[np.newaxis, :, :]
            wp_visited_by_loc = np.any(np.linalg.norm(diff, axis=2) < 1e-6, axis=1)
        else:
            wp_visited_by_loc = np.zeros(waypoints.shape[0], dtype=bool)
        wp_visited_by_grid = np.sum(action_visit[wp_x_idx, wp_y_idx, :], axis=1) > 0
        wp_visited = wp_visited_by_loc | wp_visited_by_grid

        wp_selectable = wp_feasible & (~wp_visited)
        if not np.any(wp_selectable):
            wp_selectable = wp_feasible

        masked_wp = np.where(wp_selectable, wp_scores, -np.inf)
        best_wp_idx = int(np.argmax(masked_wp))
        if not np.isfinite(masked_wp[best_wp_idx]):
            best_wp_idx = int(np.argmax(wp_scores))
            acq_value = float(wp_scores[best_wp_idx])
        else:
            acq_value = float(masked_wp[best_wp_idx])

        x_loc = float(waypoints[best_wp_idx, 0])
        y_loc = float(waypoints[best_wp_idx, 1])
        x_idx = int(np.clip(np.round(x_loc), 0, N1 - 1))
        y_idx = int(np.clip(np.round(y_loc), 0, N2 - 1))
        flat_idx = int(np.ravel_multi_index((x_idx, y_idx), (N1, N2)))


    freq_uncertainty = var_map[x_idx, y_idx, :].astype(float)
    freq_penalty = action_visit[x_idx, y_idx, :].astype(float)
    if np.max(freq_penalty) > 0:
        freq_penalty = freq_penalty / np.max(freq_penalty)
    freq_acq = freq_uncertainty - float(beta_f) * freq_penalty
    center_freq = int(np.argmax(freq_acq))

    chosen_freq_score = float(freq_acq[center_freq])
    chosen_freq_penalty = float(freq_penalty[center_freq])

    return (
        x_loc,
        y_loc,
        x_idx,
        y_idx,
        center_freq,
        acq_value,
        chosen_freq_score,
        chosen_freq_penalty,
    )


def _window_indices(center, K, width):
    """Circular frequency window centered at center."""
    width = int(np.clip(width, 1, K))
    left = width // 2
    right = width - left
    offsets = np.arange(-left, right, dtype=int)
    return ((center + offsets) % K).astype(int)


def _ordered_unique(indices):
    seen = set()
    out = []
    for idx in indices:
        i = int(idx)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out

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
    


def build_shifted_cyclic_mask(K, center_freq, next_center, band_width=7, shift_count=2):
    """
    Build cyclic partial-frequency mask.

    The mask keeps most bands around current center, and moves part of them
    towards the next-step center so current measurement overlaps with next-step
    target bands.
    """
    band_width = int(np.clip(band_width, 1, K))
    shift_count = int(np.clip(shift_count, 0, max(0, band_width - 1)))

    base = _window_indices(center_freq, K, band_width)
    nxt = _window_indices(next_center, K, band_width)

    keep_n = band_width - shift_count
    keep = list(base[:keep_n])
    preview_pool = [b for b in nxt if b not in keep]
    moved = preview_pool[:shift_count]

    merged = _ordered_unique(keep + moved + list(base) + list(nxt) + list(range(K)))
    chosen = merged[:band_width]

    omega_row = np.zeros(K, dtype=np.int32)
    omega_row[np.asarray(chosen, dtype=int)] = 1
    return omega_row, np.asarray(chosen, dtype=int), np.asarray(moved, dtype=int)


def measure_from_oracle_with_mask(data, idx, omega, rng):
    """Measure one location with partial frequency mask omega_row."""
    cfg = data["config"]
    K = cfg.K

    x_idx = idx // cfg.N2
    y_idx = idx % cfg.N2
    loc = data["sensor_locs"][idx:idx + 1].copy()

    sensor_observed = data["Gamma_obs"][x_idx, y_idx, :]
    true_spec = data["H"][x_idx, y_idx, :].astype(float)
    noise = rng.normal(0.0, np.sqrt(float(data["sigma2_noise"])), size=K)
    observed = np.maximum(true_spec + noise, 1e-10)

    gamma_row = observed * omega.astype(float)
    return loc, gamma_row[np.newaxis, :], omega[np.newaxis, :], true_spec


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
    return 2


def plot_active_cyclic_summary(result, save_path="active_ensemble_summary.png"):
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
    uncertainty_2d = np.max(var_map, axis=2)

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
    ax.set_title("Uncertainty (max over f)")
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
        nexts = np.array([h["next_center"] for h in history], dtype=int)
        ax.plot(centers, "o-", lw=1.5, ms=3, label="center")
        ax.plot(nexts, "s--", lw=1.0, ms=3, label="next")
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
    trajectory_type="tsp",
    M_init=20,
    T=None,
    M_ens=8,
    keep_recent=3,
    d_max=10.0,
    lambda_u=5.0,
    lambda_c=0.15,
    lambda_r=0.08,
    beta_f=0.2,
    lambda_p=0.0,
    cyclic_band_width=7,
    cyclic_shift_count=2,
    cyclic_step=2,
    fifo_queue_size=None,
    warmstart_nmse_threshold=0.0,
    quality_weighted_ensemble=True,
):
    """Active loop with ensemble uncertainty and cyclic partial-frequency masks."""
    # 1) 生成完整数据并对 Gamma_obs 做 8-bit 量化（不改数据生成代码）
    cfg = SimConfig(full_obs=True, R=1, M=M_budget, N1=31, N2=31, L=31)
    data = generate_data(cfg, seed=seed, addShadow=False)
    data["Gamma_obs"], gamma_q_meta = quantize_to_8bit(data["Gamma_obs"])

    bounds = ((0, cfg.L), (0, cfg.L))
    K = cfg.K

    rng = np.random.default_rng(seed + 7000)
    sensor_locs = data["sensor_locs"]
    total_grid = cfg.N1 * cfg.N2
    M_init = int(np.clip(M_init, 1, min(total_grid, sensor_locs.shape[0])))

    # 2) 程序开始先做整数网格聚类
    sensor_grid_xy, grid_to_sensor_indices = cluster_sensor_locs_to_integer_grid(sensor_locs, cfg.N1, cfg.N2)
    n_clusters = len(grid_to_sensor_indices)

    # 3) 初始观测仍按轨迹取 M_init 个点
    sensor_locs_sorted, order = sort_sensors_by_trajectory(
        sensor_locs,
        trajectory_type=trajectory_type
    )
    Gamma_sorted = data["Gamma_obs"][order, :]
    init_sensor_indices = order[:M_init]

    init_locs = sensor_locs_sorted[:M_init]

    print("=" * 60)
    print(" 基于不确定性的固定观测带宽限制下radio map重建实验")
    print("=" * 60)
    print(f"最大传感器数: {M_init}, 网格: {cfg.N1}x{cfg.N2}, 总频段: {K}")
    print(
        f"Gamma_obs 8-bit量化: min={gamma_q_meta['min']:.6e}, "
        f"max={gamma_q_meta['max']:.6e}, levels={gamma_q_meta['levels']}"
    )
    print(f"传感器整数网格聚类数: {n_clusters} / {cfg.N1 * cfg.N2}")

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
    sampled_sensor_mask = np.zeros(sensor_locs.shape[0], dtype=bool)
    model = None
    fusion_meta = dict(raw_count=0, fused_count=0, compression_ratio=1.0)
    warmstart_probe = None
    queue_dropped_total = 0

    # 4) 先进行少量初始观测
    for i, loc in enumerate(init_locs):
        if sensor_locs_t.shape[0] >= M_budget:
            break

        center = int(rng.integers(0, cfg.K))
        omega, _ = build_observe_mask(cfg.K, center, band_width=cyclic_band_width)
        observed_bands = np.where(omega > 0)[0]

        loc = sensor_locs_sorted[i]
        gamma = np.asarray(Gamma_sorted[i], dtype=float)
        gamma = gamma * omega.astype(float)

        sensor_locs_t = np.vstack([sensor_locs_t, loc])
        gamma_t = np.vstack([gamma_t, gamma])
        omega_t = np.vstack([omega_t, omega])

        sensor_idx = int(init_sensor_indices[i])
        sampled_sensor_mask[sensor_idx] = True
        gx, gy = sensor_grid_xy[sensor_idx]

        action_visit[gx, gy, observed_bands] += 1.0
        queue_sensor_locs, queue_gamma, queue_omega, dropped_now = _append_fifo_observations(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            loc,
            gamma,
            omega,
            max_len=fifo_queue_size,
        )
        queue_dropped_total += dropped_now

    n_init_sampled = int(sensor_locs_t.shape[0])
    if n_init_sampled == 0:
        raise ValueError("No initial samples were collected.")

    fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
        queue_sensor_locs,
        queue_gamma,
        queue_omega,
        cfg.N1,
        cfg.N2,
    )
    init_max_iter = _select_outer_iters(fusion_meta["fused_count"], warmstart=False)
    model = _fit_reconstruction_model(
        fused_sensor_locs,
        fused_gamma,
        fused_omega,
        cfg=cfg,
        grid_points=data["grid_coords"],
        bounds=bounds,
        I_mask=data["I_mask"],
        prev_model=None,
        warmstart=False,
        max_iter=init_max_iter,
    )

    current_loc = sensor_locs_t[-1].copy()
    if T is None:
        T = max(0, int(M_budget - sensor_locs_t.shape[0]))
    T = int(max(0, T))

    print("=" * 72)
    print("Active Ensemble + Cyclic Partial-Frequency Observation")
    print("=" * 72)
    print(f"Init={sensor_locs_t.shape[0]}, budget={M_budget}, active_round_cap={T}, M_ens={M_ens}")
    print(f"band_width={cyclic_band_width}, shift_count={cyclic_shift_count}, cyclic_step={cyclic_step}")
    print(
        f"fifo_queue_size={fifo_queue_size}, "
        f"fused_init={fusion_meta['fused_count']}, "
        f"compression={fusion_meta['compression_ratio']:.2f}, "
        f"quality_weighted={quality_weighted_ensemble}"
    )

    # 5) 网格级主动采样：每轮选一个网格，采完该网格内所有点，再下一轮重建
    t = 0
    while t < T and sensor_locs_t.shape[0] < M_budget:
        if not np.any(~sampled_sensor_mask):
            print("All sensors are already sampled. Stopping.")
            break

        keep_ratio_t = _adaptive_keep_ratio(
            fusion_meta["fused_count"], early_ratio=0.85, late_ratio=0.75, switch_M=20
        )

        mean_map_t, var_map_t, _, ensemble_info = ensemble_reconstruct_maps(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=data["grid_coords"],
            bounds=bounds,
            I_mask=data["I_mask"],
            M_ens=M_ens,
            keep_ratio=keep_ratio_t,
            keep_recent=keep_recent,
            seed=seed + 31 * (t + 1),
            base_model=model,
            member_max_iter=_select_outer_iters(fusion_meta["fused_count"], warmstart=True),
            quality_weighted=quality_weighted_ensemble,
        )
        last_var_map = var_map_t

        acquisition_space, acq_comp = build_acquisition_space(
            var_map_t,
            next_points=data["grid_coords"],
            current_loc=current_loc,
            sampled_locs=sensor_locs_t,
            action_visit=action_visit,
            lambda_u=lambda_u,
            lambda_c=lambda_c,
            lambda_r=lambda_r,
            redundancy_length=5.0,
        )

        unsampled_mask = ~sampled_sensor_mask
        grid_acq, _, grid_counts = build_clustered_grid_acquisition(
            acquisition_space,
            sensor_grid_xy=sensor_grid_xy,
            sensor_candidate_mask=unsampled_mask,
            N1=cfg.N1,
            N2=cfg.N2,
        )

        if d_max is not None:
            d_max_val = float(d_max)
            xx, yy = np.meshgrid(np.arange(cfg.N1), np.arange(cfg.N2), indexing="ij")
            dist_grid = np.sqrt((xx - current_loc[0]) ** 2 + (yy - current_loc[1]) ** 2)
            reachable = dist_grid <= d_max_val
            feasible = reachable & (grid_counts > 0)
            if np.any(feasible):
                grid_acq = np.where(feasible, grid_acq, -np.inf)

        if not np.any(np.isfinite(grid_acq)):
            print("No feasible clustered grid available. Stopping.")
            break

        best_flat = int(np.argmax(grid_acq))
        x_idx, y_idx = np.unravel_index(best_flat, (cfg.N1, cfg.N2))
        acq = float(grid_acq[x_idx, y_idx])

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
        priority_term = float(lambda_p * acq_comp["priority_map"][x_idx, y_idx])
        reward_total = float(acq + freq_score_chosen)

        # next_center = int((center_freq + cyclic_step) % cfg.K)
        next_center = center_freq
        new_omega, _ = build_observe_mask(cfg.K, next_center, band_width=cyclic_band_width)
        observed_bands = np.where(new_omega > 0)[0]

        cluster_indices = [
            idx for idx in grid_to_sensor_indices.get((int(x_idx), int(y_idx)), [])
            if not sampled_sensor_mask[idx]
        ]
        if len(cluster_indices) == 0:
            print(f"[{t + 1:02d}/{T:02d}] grid=({x_idx},{y_idx}) has no unsampled sensors, skip.")
            t += 1
            continue

        target_grid_loc = np.array([float(x_idx), float(y_idx)], dtype=float)
        current_loc = target_grid_loc.copy()
        cluster_locs = sensor_locs[np.asarray(cluster_indices, dtype=int)]
        dist_order = np.argsort(np.linalg.norm(cluster_locs - target_grid_loc[np.newaxis, :], axis=1))
        ordered_sensor_indices = [cluster_indices[int(i)] for i in dist_order]

        sampled_in_grid = 0
        batch_locs = []
        batch_gamma = []
        batch_omega = []
        for seq, sensor_idx in enumerate(ordered_sensor_indices, start=1):
            if sensor_locs_t.shape[0] >= M_budget:
                break

            new_loc = np.asarray(sensor_locs[sensor_idx], dtype=float)[np.newaxis, :]
            new_gamma = np.asarray(data["Gamma_obs"][sensor_idx], dtype=float)[np.newaxis, :]
            new_gamma = new_gamma * new_omega[np.newaxis, :].astype(float)
            true_spec = np.asarray(data["H"][x_idx, y_idx, :], dtype=float)

            sensor_locs_t = np.vstack([sensor_locs_t, new_loc])
            gamma_t = np.vstack([gamma_t, new_gamma])
            omega_t = np.vstack([omega_t, new_omega])
            sampled_sensor_mask[sensor_idx] = True

            gx_s, gy_s = sensor_grid_xy[sensor_idx]
            action_visit[gx_s, gy_s, observed_bands] += 1.0
            current_loc = new_loc[0].copy()
            sampled_in_grid += 1
            batch_locs.append(new_loc[0].copy())
            batch_gamma.append(new_gamma[0].copy())
            batch_omega.append(new_omega.copy())

            history.append(
                dict(
                    step=len(history) + 1,
                    round=t + 1,
                    sample_in_round=seq,
                    samples_in_grid=int(len(ordered_sensor_indices)),
                    sensor_idx=int(sensor_idx),
                    x=float(new_loc[0, 0]),
                    y=float(new_loc[0, 1]),
                    x_idx=int(gx_s),
                    y_idx=int(gy_s),
                    selected_grid_x=int(x_idx),
                    selected_grid_y=int(y_idx),
                    center_freq=int(center_freq),
                    next_center=int(next_center),
                    observed_bands=observed_bands.copy(),
                    omega_row=new_omega.copy(),
                    acquisition=float(acq),
                    reward_space=float(acq),
                    reward_freq=float(freq_score_chosen),
                    reward_total=float(reward_total),
                    uncertainty=float(var_map_t[x_idx, y_idx, center_freq]),
                    keep_ratio=float(keep_ratio_t),
                    uncertainty_term=float(uncertainty_term),
                    move_term=float(move_term),
                    redundancy_term=float(redundancy_term),
                    priority_term=float(priority_term),
                    observed_center_value=float(new_gamma[0, center_freq]),
                    true_center_value=float(true_spec[center_freq]),
                    raw_queue_size_before=int(queue_sensor_locs.shape[0]),
                    fused_count_before=int(fusion_meta["fused_count"]),
                )
            )

        if sampled_in_grid == 0:
            print(f"[{t + 1:02d}/{T:02d}] grid=({x_idx},{y_idx}) no sample added due to budget.")
            break

        batch_locs = np.asarray(batch_locs, dtype=float)
        batch_gamma = np.asarray(batch_gamma, dtype=float)
        batch_omega = np.asarray(batch_omega, dtype=np.int32)
        queue_sensor_locs, queue_gamma, queue_omega, dropped_now = _append_fifo_observations(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            batch_locs,
            batch_gamma,
            batch_omega,
            max_len=fifo_queue_size,
        )
        queue_dropped_total += dropped_now

        fused_sensor_locs, fused_gamma, fused_omega, fusion_meta = _fuse_observations_by_grid(
            queue_sensor_locs,
            queue_gamma,
            queue_omega,
            cfg.N1,
            cfg.N2,
        )
        refit_max_iter = _select_outer_iters(fusion_meta["fused_count"], warmstart=True)
        model = _fit_reconstruction_model(
            fused_sensor_locs,
            fused_gamma,
            fused_omega,
            cfg=cfg,
            grid_points=data["grid_coords"],
            bounds=bounds,
            I_mask=data["I_mask"],
            prev_model=model,
            warmstart=True,
            max_iter=refit_max_iter,
        )

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
            probe_model = _fit_reconstruction_model(
                fused_sensor_locs,
                fused_gamma,
                fused_omega,
                cfg=cfg,
                grid_points=data["grid_coords"],
                bounds=bounds,
                I_mask=data["I_mask"],
                prev_model=model,
                warmstart=True,
                max_iter=max(2, refit_max_iter),
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

        for i in range(1, sampled_in_grid + 1):
            history[-i]["nmse"] = float(nmse_t)
            history[-i]["base_nmse"] = float(base_nmse_t)
            history[-i]["recon_mode"] = recon_mode
            history[-i]["raw_queue_size_after"] = int(queue_sensor_locs.shape[0])
            history[-i]["fused_count_after"] = int(fusion_meta["fused_count"])
            history[-i]["compression_ratio"] = float(fusion_meta["compression_ratio"])
            history[-i]["queue_dropped_total"] = int(queue_dropped_total)
            history[-i]["queue_dropped_now"] = int(dropped_now)
            history[-i]["warmstart_probe_nmse"] = (
                None if warmstart_probe_nmse is None else float(warmstart_probe_nmse)
            )
            history[-i]["warmstart_probe_gain"] = (
                None if warmstart_probe_gain is None else float(warmstart_probe_gain)
            )

        weight_summary = np.asarray(ensemble_info["weights"], dtype=float)
        warmstart_msg = ""
        if warmstart_probe_nmse is not None:
            warmstart_msg = f" warm_probe={warmstart_probe_nmse:.4f}"
        print(
            f"[{t + 1:02d}/{T:02d}] "
            f"grid=({x_idx:02d},{y_idx:02d}) "
            f"samples_in_grid={sampled_in_grid} total={sensor_locs_t.shape[0]} "
            f"center={center_freq:02d} "
            f"reward_space={acq:.4f} reward_freq={freq_score_chosen:.4f} reward_total={reward_total:.4f} "
            f"(u={uncertainty_term:.4f}, -c={move_term:.4f}, -r={redundancy_term:.4f}) "
            f"keep={keep_ratio_t:.2f} "
            f"queue={queue_sensor_locs.shape[0]} fused={fusion_meta['fused_count']} "
            f"comp={fusion_meta['compression_ratio']:.2f} "
            f"wmax={np.max(weight_summary):.3f} mode={recon_mode} "
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
    )

    plot_active_cyclic_summary(result, save_path="active_ensemble_summary.png")

    print("-" * 72)
    print(f"Final samples: {sensor_locs_t.shape[0]}")
    print(
        f"Final reconstruction rows: raw_queue={queue_sensor_locs.shape[0]}, "
        f"fused={fusion_meta['fused_count']}, "
        f"compression={fusion_meta['compression_ratio']:.2f}"
    )
    print(f"Final NMSE: {final_nmse:.4f}")
    if warmstart_probe is not None:
        print(
            f"Warm-start probe at step {warmstart_probe['step']}: "
            f"{warmstart_probe['base_nmse']:.4f} -> {warmstart_probe['warmstart_nmse']:.4f}"
        )
    print("Saved: active_ensemble_summary.png")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_active_ensemble_cyclic_experiment(
        M_budget=160,
        seed=42,
        trajectory_type="snake",
        M_init=10,
        M_ens=6,
        keep_recent=2,
        d_max=10.0,
        cyclic_band_width=6,
        cyclic_shift_count=2,
        cyclic_step=2,
        fifo_queue_size=100,
        warmstart_nmse_threshold=0.5,
        quality_weighted_ensemble=True,
    )

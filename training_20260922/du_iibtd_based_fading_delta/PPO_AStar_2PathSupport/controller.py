"""Two-path predictive communication-support target selector for PPO+A*.

Two axis-priority routes form a deterministic envelope around future UAV
motion. With roofs at or above flight altitude, executable macro-action A*
routes replace the open-space Manhattan routes and check every crossed cell.  Each route contributes at most one ensemble refresh batch of future
macro-action endpoints.  Those UAV cells are references for communication
scoring only; the UGV goal is always a cell in its connected road component.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ..channel_loss import excess_loss_db, link_blocked_length_m
from ..power_control import current_tx_power_dbm
from scipy.ndimage import distance_transform_edt

from du_iibtd_based_fading_delta.PPO_AStar.controller import select_astar_ugv_action


GridCell = Tuple[int, int]


def _as_cell(value) -> GridCell:
    point = np.asarray(value, dtype=float).reshape(2)
    return int(np.rint(point[0])), int(np.rint(point[1]))


def axis_priority_path_prefix(
    start: GridCell,
    goal: GridCell,
    *,
    macro_step: int,
    horizon: int,
    first_axis: int,
) -> Tuple[GridCell, ...]:
    """Return future UAV macro-action endpoints for one axis priority."""
    if int(first_axis) not in (0, 1):
        raise ValueError(f"first_axis must be 0 or 1, got {first_axis}")
    current = [int(start[0]), int(start[1])]
    target = (int(goal[0]), int(goal[1]))
    macro_step = max(1, int(macro_step))
    horizon = max(1, int(horizon))
    if tuple(current) == target:
        return (target,)

    result: List[GridCell] = []
    axis_order = (int(first_axis), 1 - int(first_axis))
    for _ in range(horizon):
        moved = False
        for axis in axis_order:
            delta = int(target[axis] - current[axis])
            if delta == 0:
                continue
            step = min(abs(delta), macro_step)
            current[axis] += step if delta > 0 else -step
            moved = True
            break
        if not moved:
            break
        result.append((int(current[0]), int(current[1])))
        if tuple(current) == target:
            break
    return tuple(result)


def two_axis_path_prefixes(
    start: GridCell,
    goal: GridCell,
    *,
    macro_step: int,
    horizon: int,
) -> Tuple[Tuple[GridCell, ...], ...]:
    """Return the unique x-first and y-first UAV shortest-path prefixes."""
    paths = (
        axis_priority_path_prefix(
            start,
            goal,
            macro_step=macro_step,
            horizon=horizon,
            first_axis=0,
        ),
        axis_priority_path_prefix(
            start,
            goal,
            macro_step=macro_step,
            horizon=horizon,
            first_axis=1,
        ),
    )
    unique: List[Tuple[GridCell, ...]] = []
    for path in paths:
        if path and path not in unique:
            unique.append(path)
    return tuple(unique)


def _road_proxy(env, reference: GridCell) -> Optional[GridCell]:
    ugv_cell = _as_cell(env.ugv_pos)
    component = int(env._ugv_component_labels[ugv_cell])
    projection_cache = getattr(env, "_ugv_2path_road_projection_cache", None)
    if projection_cache is None:
        projection_cache = {}
        env._ugv_2path_road_projection_cache = projection_cache
    projection = projection_cache.get(component)
    if projection is None:
        component_mask = np.asarray(
            env._ugv_component_labels == component,
            dtype=bool,
        )
        if not bool(np.any(component_mask)):
            return None
        nearest_indices = distance_transform_edt(
            ~component_mask,
            return_distances=False,
            return_indices=True,
        )
        projection = np.moveaxis(nearest_indices, 0, -1).astype(
            np.int32,
            copy=False,
        )
        projection_cache[component] = projection

    gx = int(np.clip(int(reference[0]), 0, projection.shape[0] - 1))
    gy = int(np.clip(int(reference[1]), 0, projection.shape[1] - 1))
    proxy = projection[gx, gy]
    return int(proxy[0]), int(proxy[1])


def _large_scale_capacity(env, uav_cell: GridCell, ugv_cell: GridCell, *, power_dbm=None, comm_units=None) -> float:
    """Return deterministic outage-gated capacity without consuming channel RNG."""
    power_dbm = current_tx_power_dbm(env) if power_dbm is None else float(power_dbm)
    comm_units = int(env.current_comm_units) if comm_units is None else int(comm_units)
    cache: Dict[tuple, float] = getattr(
        env,
        "_ugv_2path_capacity_cache",
        None,
    )
    if cache is None:
        cache = {}
        env._ugv_2path_capacity_cache = cache
    key = (
        int(uav_cell[0]),
        int(uav_cell[1]),
        int(ugv_cell[0]),
        int(ugv_cell[1]),
        comm_units,
        power_dbm,
    )
    cached = cache.get(key)
    if cached is not None:
        return float(cached)

    bandwidth = (
        float(comm_units)
        * float(env.config.uav.unit_bandwidth_hz)
    )
    noise_power_dbm = (
        -174.0
        + 10.0 * np.log10(bandwidth + 1e-3)
        + float(env.config.comm.noise_figure_db)
    )
    uav_position = np.asarray(uav_cell, dtype=float)
    ugv_position = np.asarray(ugv_cell, dtype=float)
    has_los = bool(
        env.scene.has_line_of_sight(
            uav_position=uav_position,
            ugv_position=ugv_position,
        )
    )
    horizontal_distance = float(
        np.linalg.norm(uav_position - ugv_position)
        * float(env.config.scene.grid_spacing)
    )
    height_gap = max(
        float(env.config.scene.uav_height)
        - float(env.config.scene.ugv_height),
        0.0,
    )
    distance_3d = max(float(np.hypot(horizontal_distance, height_gap)), 1.0)
    path_loss_db = (
        20.0 * np.log10(distance_3d)
        + 20.0 * np.log10(float(env.config.comm.carrier_freq))
        - 147.55
        + excess_loss_db(env.config.comm, has_los, link_blocked_length_m(
            env.scene, env.config.comm, uav_position, ugv_position, has_los))
    )
    snr_db = (
        power_dbm
        - float(path_loss_db)
        - float(noise_power_dbm)
    )
    if snr_db < float(env.config.comm.snr_outage_threshold_db):
        capacity = 0.0
    else:
        capacity = float(bandwidth * np.log2(1.0 + 10.0 ** (snr_db / 10.0)))
    if len(cache) >= 16_384:
        cache.clear()
    cache[key] = float(capacity)
    return float(capacity)


def select_two_path_support_target(
    env,
    uav_target_grid=None,
) -> Optional[GridCell]:
    """Select one road goal that robustly supports two UAV path prefixes."""
    start = _as_cell(env.uav_pos)
    if uav_target_grid is None:
        uav_target_grid = env._get_motion_target_grid()
    goal = start if uav_target_grid is None else _as_cell(uav_target_grid)
    ugv_cell = _as_cell(env.ugv_pos)
    component = int(env._ugv_component_labels[ugv_cell])
    # The forecast horizon is one map-update batch, so hold its selected road
    # goal for that same batch.  Recompute only after an episode reset, a map
    # update, a planner-target change, or a transmit-power change; calling this controller every
    # environment step must not repeat the 2B-by-2B LoS scoring work.
    cache_key = (
        int(getattr(env, "_reset_counter", 0)),
        int(getattr(env, "map_update_count", 0)),
        int(goal[0]),
        int(goal[1]),
        int(component),
        current_tx_power_dbm(env),
    )
    cached_plan = getattr(env, "_ppo_astar_2path_target_cache", None)
    if cached_plan is not None and cached_plan[0] == cache_key:
        (
            _,
            selected,
            forecast_paths,
            candidate_goals,
        ) = cached_plan
        env._ppo_astar_2path_forecast_paths = forecast_paths
        env._ppo_astar_2path_candidate_goals = candidate_goals
        env._ppo_astar_2path_support_goal = selected
        return selected

    batch_size = max(1, int(env.config.planner.ensemble_refresh_interval))
    if getattr(env.scene, "has_uav_obstacles", False):
        paths = [env.scene.uav_path_prefix(
            start, goal, macro_step=int(env.uav_step_count),
            horizon=batch_size, first_axis=axis,
        ) for axis in (0, 1)]
        forecast_paths = tuple(dict.fromkeys(path for path in paths if path))
    else:
        forecast_paths = two_axis_path_prefixes(
            start, goal, macro_step=int(env.uav_step_count), horizon=batch_size,
        )
    if not forecast_paths:
        forecast_paths = ((start,),)

    candidate_goals = {
        proxy
        for path in forecast_paths
        for reference in path
        for proxy in (_road_proxy(env, reference),)
        if proxy is not None
    }
    if not candidate_goals:
        return _as_cell(env.ugv_pos)

    expanded_horizon = max(
        1,
        int(env.config.planner.ugv_comm_expanded_path_horizon),
    )

    def _score(candidate: GridCell) -> Tuple[float, ...]:
        route_capacities: List[Tuple[float, ...]] = []
        for path in forecast_paths:
            route_capacities.append(
                tuple(_large_scale_capacity(env, point, candidate) for point in path)
            )
        route_service_ratios = tuple(
            float(np.mean(np.asarray(values, dtype=float) > 0.0))
            for values in route_capacities
        )
        route_mean_capacities = tuple(
            float(np.mean(values)) for values in route_capacities
        )
        all_capacities = tuple(
            value for values in route_capacities for value in values
        )
        distance = abs(int(candidate[0]) - ugv_cell[0]) + abs(
            int(candidate[1]) - ugv_cell[1]
        )
        return (
            1.0 if distance <= expanded_horizon else 0.0,
            min(route_service_ratios),
            min(route_mean_capacities),
            float(np.mean(all_capacities)),
            -float(distance),
            -float(candidate[0]),
            -float(candidate[1]),
        )

    selected = max(sorted(candidate_goals), key=_score)
    env._ppo_astar_2path_forecast_paths = forecast_paths
    sorted_candidate_goals = tuple(sorted(candidate_goals))
    env._ppo_astar_2path_candidate_goals = sorted_candidate_goals
    env._ppo_astar_2path_support_goal = selected
    env._ppo_astar_2path_target_cache = (
        cache_key,
        selected,
        forecast_paths,
        sorted_candidate_goals,
    )
    return selected


def select_two_path_support_ugv_action(env) -> int:
    """Choose the two-path support goal and follow its shared cached A* route."""
    target = select_two_path_support_target(env)
    if target is None:
        return 0
    return select_astar_ugv_action(env, target)


__all__ = [
    "axis_priority_path_prefix",
    "select_two_path_support_target",
    "select_two_path_support_ugv_action",
    "two_axis_path_prefixes",
]

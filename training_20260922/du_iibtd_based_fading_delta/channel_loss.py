"""Shared, deterministic obstruction-length correction for all link users.

The length is horizontal building coverage where the sampled center ray is
below the roof. It is a geometry proxy, not a calibrated penetration law.
"""
from __future__ import annotations

import math


def parse_length_loss_cap(value):
    """CLI representation of the JSON null used for an uncapped correction."""
    if value is None or str(value).strip().lower() in {"none", "null", "uncapped"}:
        return None
    cap = float(value)
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("length-loss cap must be none or a finite non-negative number")
    return cap


def length_correction_db(comm, los: bool, blocked_length_m: float) -> float:
    if los:
        return 0.0
    coefficient = float(getattr(comm, "nlos_length_loss_db_per_m", 0.0))
    cap = getattr(comm, "nlos_length_loss_cap_db", 0.0)
    correction = coefficient * max(0.0, float(blocked_length_m))
    return correction if cap is None else min(float(cap), correction)


def excess_loss_db(comm, los: bool, blocked_length_m: float = 0.0) -> float:
    base = float(comm.los_excess_db if los else comm.nlos_excess_db)
    return base + length_correction_db(comm, los, blocked_length_m)


def link_blocked_length_m(scene, comm, uav_position, ugv_position, los: bool) -> float:
    """Skip geometry work for LoS and for explicitly disabled old profiles."""
    cap = getattr(comm, "nlos_length_loss_cap_db", 0.0)
    if (los or float(getattr(comm, "nlos_length_loss_db_per_m", 0.0)) <= 0.0
            or (cap is not None and float(cap) <= 0.0)):
        return 0.0
    return scene.get_blocked_length_m(
        uav_position=uav_position, ugv_position=ugv_position)


def cached_blocked_length_m(scene, uav_position, ugv_position) -> float:
    """Reuse the production ray rasterizer and bounded, per-scene cache.

    Cell-center projection midpoints approximate horizontal coverage at the
    scene's grid spacing. Contiguous roofs are accumulated by length, never
    counted by pixel or synthetic building ID. Geometry is fixed per scene;
    a newly generated building map gets a new GridScene and an empty cache.
    """
    ux, uy = (float(v) for v in uav_position)
    gx, gy = (float(v) for v in ugv_position)
    hu, hg, spacing = float(scene.uav_height), float(scene.ugv_height), float(scene.grid_spacing)
    key = (ux, uy, gx, gy, hu, hg, spacing)
    cached = scene._blocked_length_cache.get(key)
    if cached is not None:
        return float(cached)
    if not (scene._is_within_bounds(uav_position) and scene._is_within_bounds(ugv_position)):
        raise ValueError("Blocked-length query positions must be within the scene")
    dx, dy = gx - ux, gy - uy
    length_squared = dx * dx + dy * dy
    horizontal_length = math.sqrt(length_squared) * spacing
    if length_squared <= 1e-12:
        scene._cache_store(scene._blocked_length_cache, key, 0.0)
        return 0.0
    cells = scene._get_supercover_line_cells(
        scene._grid_index(uav_position), scene._grid_index(ugv_position))
    samples = []
    for ix, iy in cells:
        t = min(1.0, max(0.0, ((ix - ux) * dx + (iy - uy) * dy) / length_squared))
        height = float(scene.building_heights[ix, iy])
        blocked = height > 0.0 and hu + t * (hg - hu) <= height + 1e-6
        samples.append((t, blocked))
    blocked_fraction = 0.0
    previous_boundary = 0.0
    for i, (t, blocked) in enumerate(samples):
        next_boundary = (t + samples[i + 1][0]) * 0.5 if i + 1 < len(samples) else 1.0
        if blocked:
            blocked_fraction += max(0.0, next_boundary - previous_boundary)
        previous_boundary = next_boundary
    length = min(horizontal_length, max(0.0, blocked_fraction * horizontal_length))
    scene._cache_store(scene._blocked_length_cache, key, length)
    return length

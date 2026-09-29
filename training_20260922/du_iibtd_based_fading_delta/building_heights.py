"""Reproducible per-footprint heights for the communication geometry."""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
from scipy.ndimage import label

_CONNECTED_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _split_connected_footprint(mask: np.ndarray) -> np.ndarray:
    """Return one side of a deterministic, connected two-way spatial partition."""
    coordinates = np.argwhere(mask)
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    axes = sorted(range(2), key=lambda axis: (-(upper[axis] - lower[axis]), axis))
    candidates = []
    for preference, axis in enumerate(axes):
        for cut in range(int(lower[axis]), int(upper[axis])):
            size = int(np.count_nonzero(coordinates[:, axis] <= cut))
            candidates.append((abs(len(coordinates) - 2 * size), preference, cut, axis))
    grid = np.indices(mask.shape)
    for _, _, cut, axis in sorted(candidates):
        first = mask & (grid[axis] <= cut)
        if label(first, _CONNECTED_4)[1] == 1 and label(mask & ~first, _CONNECTED_4)[1] == 1:
            return first

    # Irregular outlines may admit no connected straight cut. Removing an edge
    # from a BFS spanning tree always produces two connected, nonempty pieces.
    root = tuple(coordinates[0])
    parent = {root: None}
    order = []
    queue = deque([root])
    while queue:
        point = queue.popleft()
        order.append(point)
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (point[0] + dx, point[1] + dy)
            if (0 <= neighbor[0] < mask.shape[0] and 0 <= neighbor[1] < mask.shape[1]
                    and mask[neighbor] and neighbor not in parent):
                parent[neighbor] = point
                queue.append(neighbor)
    sizes = dict.fromkeys(order, 1)
    for point in reversed(order[1:]):
        sizes[parent[point]] += sizes[point]
    cut_root = min(order[1:], key=lambda point: (abs(len(order) - 2 * sizes[point]), point))
    first = np.zeros_like(mask)
    for point in order:
        first[point] = point == cut_root or (parent[point] is not None and first[parent[point]])
    return first


def partition_building_footprints(building_mask, scene):
    """Label 4-connected footprints; split each sufficiently large one once.

    A zero threshold preserves the original connected-component geometry.
    Labels 1..N retain the original component order; second halves get new IDs.
    The operation changes neither occupancy nor streets and uses no random state.
    """
    mask = np.asarray(building_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("building_mask must be two-dimensional")
    components, count = label(mask, structure=_CONNECTED_4)
    threshold = float(getattr(scene, "building_split_min_area_m2", 0.0))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("scene.building_split_min_area_m2 must be finite and non-negative")
    if threshold == 0:
        return components, count
    spacing = float(scene.grid_spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("scene.grid_spacing must be finite and positive")
    areas = np.bincount(components.ravel())
    for component_id in range(1, count + 1):
        if areas[component_id] < 2 or areas[component_id] * spacing**2 < threshold:
            continue
        footprint = components == component_id
        points = np.argwhere(footprint)
        lower, upper = points.min(axis=0), points.max(axis=0) + 1
        region = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        local = footprint[region]
        first = _split_connected_footprint(local)
        count += 1
        components[region][local & ~first] = count
    return components, count


def validate_building_height_config(scene) -> None:
    mode = str(scene.building_height_mode).strip().lower()
    if mode not in ("fixed", "uniform_per_building", "uniform_integer_per_building"):
        raise ValueError("scene.building_height_mode must be fixed, uniform_per_building or uniform_integer_per_building")
    scene.building_height_mode = mode
    for name in ("building_height_m", "building_height_min_m", "building_height_max_m"):
        value = float(getattr(scene, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"scene.{name} must be finite and non-negative")
        setattr(scene, name, value)
    if scene.building_height_min_m > scene.building_height_max_m:
        raise ValueError("scene.building_height_min_m must not exceed building_height_max_m")
    if mode == "uniform_integer_per_building":
        if not scene.building_height_min_m.is_integer() or not scene.building_height_max_m.is_integer():
            raise ValueError("Integer building heights require integer-valued lower and upper bounds")
    seed = scene.building_height_seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("scene.building_height_seed must be an integer in [0, 2**32)")
    scene.building_height_seed = int(seed)
    threshold = float(getattr(scene, "building_split_min_area_m2", 0.0))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("scene.building_split_min_area_m2 must be finite and non-negative")
    scene.building_split_min_area_m2 = threshold
    if threshold > 0 and (not np.isfinite(scene.grid_spacing) or scene.grid_spacing <= 0):
        raise ValueError("scene.grid_spacing must be finite and positive")


def build_building_height_map(building_mask, scene) -> np.ndarray:
    """Give each building block one height, independent of episode/channel RNG.

    The input dataset supplies a binary footprint mask, not building instance IDs.
    Side-touching footprints initially form one component; large components are
    split into two connected blocks when enabled. Diagonal contact does not merge
    footprints. The mask digest makes the seed geometry-specific and
    independent of dataset paths, worker order, variant and training seed.
    """
    mask = np.asarray(building_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("building_mask must be two-dimensional")
    mode = getattr(scene, "building_height_mode", "fixed")
    heights = np.zeros(mask.shape, dtype=float)
    if mode == "fixed":
        value = float(scene.building_height_m)
        if not np.isfinite(value) or value < 0:
            raise ValueError("scene.building_height_m must be finite and non-negative")
        heights[mask] = value
        return heights
    validate_building_height_config(scene)
    if not np.any(mask):
        return heights
    components, count = partition_building_footprints(mask, scene)
    digest = hashlib.sha256(np.asarray(mask.shape, dtype='<i8').tobytes() + mask.astype(np.uint8).tobytes()).digest()
    entropy = [scene.building_height_seed, *np.frombuffer(digest[:16], dtype='<u4').astype(int).tolist()]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    if mode == "uniform_integer_per_building":
        sampled = rng.integers(int(scene.building_height_min_m), int(scene.building_height_max_m) + 1, size=count)
    else:
        sampled = rng.uniform(scene.building_height_min_m, scene.building_height_max_m, size=count)
    values = np.r_[0., sampled]
    return values[components]

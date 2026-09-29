"""Cached A* controller for the UGV used by the PPO+A* baseline.

The controller selects only a legal UGV macro action.  Physical movement,
communication, energy accounting, and rewards remain owned by the common
environment.  A cell-level A* path is cached until the active UAV planner
target changes or the UGV no longer lies on that path.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np


GridCell = Tuple[int, int]
_CARDINAL_ACTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 0),
    (2, 0, 1),
    (3, -1, 0),
    (4, 0, -1),
)


@dataclass
class AStarPlanState:
    """Per-environment route cache."""

    goal: Optional[GridCell] = None
    path: Tuple[GridCell, ...] = ()
    last_step: int = -1
    replans: int = 0


def _as_cell(value) -> GridCell:
    point = np.asarray(value, dtype=float).reshape(2)
    return int(np.rint(point[0])), int(np.rint(point[1]))


def astar_grid_path(
    walkable_mask: np.ndarray,
    start: GridCell,
    goal: GridCell,
) -> List[GridCell]:
    """Return a deterministic four-neighbour shortest path, or ``[]``."""
    walkable = np.asarray(walkable_mask, dtype=bool)
    if walkable.ndim != 2:
        raise ValueError(f"walkable_mask must be two-dimensional, got {walkable.shape}")
    start = _as_cell(start)
    goal = _as_cell(goal)
    nx, ny = walkable.shape

    def valid(cell: GridCell) -> bool:
        x, y = cell
        return 0 <= x < nx and 0 <= y < ny and bool(walkable[x, y])

    if not valid(start) or not valid(goal):
        return []
    if start == goal:
        return [start]

    def heuristic(cell: GridCell) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier = [(heuristic(start), 0, start[0], start[1])]
    best_cost: Dict[GridCell, int] = {start: 0}
    parent: Dict[GridCell, GridCell] = {}
    while frontier:
        _, cost, x, y = heapq.heappop(frontier)
        cell = (int(x), int(y))
        if int(best_cost.get(cell, cost + 1)) != int(cost):
            continue
        if cell == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        next_cost = int(cost) + 1
        for _, dx, dy in _CARDINAL_ACTIONS:
            neighbour = (cell[0] + dx, cell[1] + dy)
            if not valid(neighbour):
                continue
            if next_cost >= int(best_cost.get(neighbour, next_cost + 1)):
                continue
            best_cost[neighbour] = next_cost
            parent[neighbour] = cell
            heapq.heappush(
                frontier,
                (
                    next_cost + heuristic(neighbour),
                    next_cost,
                    neighbour[0],
                    neighbour[1],
                ),
            )
    return []


def _road_goal(env, requested_goal: GridCell) -> Optional[GridCell]:
    """Map a target to the UGV's connected road component when necessary."""
    requested_goal = _as_cell(requested_goal)
    current = _as_cell(env.ugv_pos)
    if bool(env.walkable_mask[requested_goal]) and int(
        env._ugv_component_labels[requested_goal]
    ) == int(env._ugv_component_labels[current]):
        return requested_goal
    component = int(env._ugv_component_labels[current])
    fallback = env._nearest_component_road_cell_to_reference(
        np.asarray(requested_goal, dtype=float),
        component,
    )
    return None if fallback is None else _as_cell(fallback)


def _new_plan(env, state: AStarPlanState, goal: GridCell) -> None:
    start = _as_cell(env.ugv_pos)
    state.goal = goal
    state.path = tuple(astar_grid_path(env.walkable_mask, start, goal))
    state.replans += 1


def select_astar_ugv_action(env, uav_target_grid=None) -> int:
    """Follow one cached A* route segment and return a cardinal macro action."""
    state = getattr(env, "_ppo_astar_plan_state", None)
    if not isinstance(state, AStarPlanState):
        state = AStarPlanState()
        env._ppo_astar_plan_state = state

    current_step = int(getattr(env, "current_step", 0))
    if current_step <= state.last_step:
        state.goal = None
        state.path = ()
    state.last_step = current_step

    if uav_target_grid is None:
        uav_target_grid = env._get_motion_target_grid()
    if uav_target_grid is None:
        uav_target_grid = _as_cell(env.uav_pos)

    goal = _road_goal(env, _as_cell(uav_target_grid))
    if goal is None:
        return 0
    current = _as_cell(env.ugv_pos)
    path = list(state.path)
    if state.goal != goal or current not in path:
        _new_plan(env, state, goal)
        path = list(state.path)
    if not path:
        return 0

    current_index = path.index(current)
    if current_index >= len(path) - 1:
        return 0
    next_cell = path[current_index + 1]
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    for action, dx, dy in _CARDINAL_ACTIONS:
        if delta == (dx, dy):
            return int(action)

    # The cached route is corrupt or incompatible with the current grid.  A
    # fresh plan is safer than silently selecting an unrelated direction.
    _new_plan(env, state, goal)
    if len(state.path) < 2:
        return 0
    delta = (
        state.path[1][0] - state.path[0][0],
        state.path[1][1] - state.path[0][1],
    )
    for action, dx, dy in _CARDINAL_ACTIONS:
        if delta == (dx, dy):
            return int(action)
    return 0


__all__ = ["AStarPlanState", "astar_grid_path", "select_astar_ugv_action"]

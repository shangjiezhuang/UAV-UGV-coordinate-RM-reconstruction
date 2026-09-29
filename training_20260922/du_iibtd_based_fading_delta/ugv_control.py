"""Shared UGV controller helpers for fair baseline execution.

The environment owns all physical transitions.  A controller only selects a
legal macro action; it must not reimplement movement, channel, or queue logic.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


UGV_CONTROL_MODES = frozenset(
    {
        "policy",
        "fixed",
        "legacy_heuristic",
        "astar",
        "astar_target",
        "astar_support",
        "astar_2path_support",
        "astar_2path_support_recovery",
    }
)


def is_astar_control_mode(value: Any) -> bool:
    """Return whether ``value`` selects one of the shared A* UGV routers."""
    return normalize_ugv_control_mode(value).startswith("astar_")


def normalize_ugv_control_mode(value: Any) -> str:
    mode = str(value or "policy").strip().lower()
    if mode not in UGV_CONTROL_MODES:
        raise ValueError(
            f"ugv_control_mode must be one of {sorted(UGV_CONTROL_MODES)}, got {value!r}"
        )
    if mode == "astar":
        return "astar_target"
    return mode


def geometric_ugv_action_mask(env: Any) -> np.ndarray:
    """Return collision-only legal UGV actions, independent of service guidance."""
    mask = np.zeros(int(env.ugv_action_size), dtype=bool)
    for direction_idx in range(int(env.ugv_action_size)):
        mask[direction_idx] = env._can_follow_direction(
            position=env.ugv_pos,
            direction_idx=direction_idx,
            step_count=env.ugv_step_count,
            validator=env.scene.is_ugv_position_valid,
        )
    if not bool(np.any(mask)):
        raise RuntimeError("UGV geometric action mask contains no legal action")
    return mask


def select_legacy_heuristic_action(env: Any) -> int:
    """Select the legacy LOS/distance-shell UGV baseline action.

    This intentionally preserves the old baseline decision rule while using
    the current environment for movement, communication, energy, and metrics.
    """
    valid_actions = np.flatnonzero(geometric_ugv_action_mask(env))
    target_uav_pos = np.asarray(env.uav_pos, dtype=float)
    ideal_support_distance = max(float(env.ugv_step_count), 2.0)
    distance_denominator = max(ideal_support_distance, 1.0)

    best_action = int(valid_actions[0])
    best_key: Optional[Tuple[float, ...]] = None
    for action in valid_actions.tolist():
        next_position, moved_steps = env._rollout_direction(
            position=env.ugv_pos,
            direction_idx=int(action),
            step_count=env.ugv_step_count,
            validator=env.scene.is_ugv_position_valid,
            stop_at_target=False,
        )
        next_distance = float(
            np.linalg.norm(np.asarray(next_position, dtype=float) - target_uav_pos)
        )
        distance_gap = abs(next_distance - ideal_support_distance)
        distance_score = -distance_gap / distance_denominator
        los_score = 1.0 if env.scene.has_line_of_sight(
            uav_position=target_uav_pos,
            ugv_position=next_position,
        ) else 0.0
        stay_score = 1.0 if int(action) == 0 else 0.0
        score = 1.5 * distance_score + 2.0 * los_score + 0.2 * stay_score
        key = (
            float(score),
            float(los_score),
            -float(distance_gap),
            float(stay_score),
            1.0 if moved_steps > 0 else 0.0,
            -float(next_distance),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_action = int(action)
    return best_action


def resolve_ugv_action(
    env: Any,
    commanded_action: int,
    controller_target_grid=None,
    support_target_grid=None,
) -> tuple[int, str]:
    """Resolve the executed action for the configured controller mode."""
    mode = normalize_ugv_control_mode(
        getattr(env.config.planner, "ugv_control_mode", "policy")
    )
    if mode == "fixed":
        return 0, mode
    if mode == "legacy_heuristic":
        return select_legacy_heuristic_action(env), mode
    if mode == "astar_2path_support":
        from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import (
            select_two_path_support_ugv_action,
        )

        return select_two_path_support_ugv_action(env), mode
    if mode == "astar_2path_support_recovery":
        from du_iibtd_based_fading_delta.PPO_AStar_2PathSupportRecovery.controller import (
            select_two_path_support_recovery_ugv_action,
        )

        return (
            select_two_path_support_recovery_ugv_action(
                env,
                support_target_grid=support_target_grid,
            ),
            mode,
        )
    if mode in {"astar_target", "astar_support"}:
        from du_iibtd_based_fading_delta.PPO_AStar.controller import select_astar_ugv_action

        target_grid = controller_target_grid
        if mode == "astar_support":
            target_grid = support_target_grid
            if target_grid is None:
                target_grid, _, _, _ = env._get_ugv_guidance_target_grid()
        return select_astar_ugv_action(env, target_grid), mode

    action = int(commanded_action)
    if not 0 <= action < int(env.ugv_action_size):
        raise ValueError(
            f"UGV action {action} is outside [0, {int(env.ugv_action_size)})."
        )
    return action, mode


def uses_service_action_mask(config: Any) -> bool:
    """Service guidance is part of the learned-policy protocol only."""
    return bool(config.planner.ugv_service_action_mask) and normalize_ugv_control_mode(
        getattr(config.planner, "ugv_control_mode", "policy")
    ) == "policy"

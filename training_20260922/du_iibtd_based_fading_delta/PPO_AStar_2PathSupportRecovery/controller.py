"""Queue-aware recovery wrapper around the two-path support controller.

Two-path predictive support remains the normal UGV objective.  The existing
path-corridor Support goal becomes a hard safety override when the live queue
cannot be served reliably.  This keeps sensing-oriented predictive placement
while preventing rare, long outage bursts from filling the physical buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from du_iibtd_based_fading_delta.PPO_AStar.controller import (
    select_astar_ugv_action,
)
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import (
    select_two_path_support_ugv_action,
)


GridCell = Tuple[int, int]
_QUEUE_GROWTH_EPS_BITS = 1.0


@dataclass
class RecoveryState:
    """Episode-local state for the two-mode UGV controller."""

    submode: str = "two_path"
    last_step: int = -1
    entered_step: int = -1
    poor_service_steps: int = 0
    good_link_steps: int = 0
    switch_reason: str = ""
    mode_switch: str = ""
    backlog_norm: float = 0.0
    service_margin_ok: bool = True


def _planner_value(env, name: str, default):
    return getattr(env.config.planner, name, default)


def _new_episode_state(env) -> RecoveryState:
    state = RecoveryState(last_step=int(getattr(env, "current_step", 0)) - 1)
    env._ppo_astar_2path_recovery_state = state
    return state


def _get_state(env) -> RecoveryState:
    state = getattr(env, "_ppo_astar_2path_recovery_state", None)
    current_step = int(getattr(env, "current_step", 0))
    if state is None or current_step <= int(state.last_step):
        state = _new_episode_state(env)
    return state


def _current_arrival_bits(env) -> float:
    packet_bits_fn = getattr(env, "_current_sample_packet_bits", None)
    if packet_bits_fn is None:
        return 0.0
    try:
        return max(0.0, float(packet_bits_fn()))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _clear_two_path_forecast(env) -> None:
    """Force the post-recovery forecast to start at the current UAV cell."""
    for name in (
        "_ppo_astar_2path_target_cache",
        "_ppo_astar_2path_forecast_paths",
        "_ppo_astar_2path_candidate_goals",
        "_ppo_astar_2path_support_goal",
    ):
        if hasattr(env, name):
            delattr(env, name)


def update_recovery_state(env) -> RecoveryState:
    """Update and expose the deterministic recovery state for this step."""
    state = _get_state(env)
    current_step = int(getattr(env, "current_step", 0))
    queue_bits = max(0.0, float(env._queue_remaining_bits()))
    backlog_norm = float(env._queue_backlog_norm())
    outage = bool(getattr(env.ugv_channel_info, "outage", False))
    capacity_bps = max(
        0.0,
        float(getattr(env.ugv_channel_info, "capacity_bps", 0.0)),
    )
    step_duration = max(
        float(getattr(env.config.uav, "step_duration", 1.0)),
        1e-9,
    )
    service_bits = capacity_bps * step_duration
    arrival_bits = _current_arrival_bits(env)
    service_margin = max(
        1.0,
        float(_planner_value(env, "ugv_recovery_service_margin", 1.0)),
    )
    service_margin_ok = bool(
        queue_bits <= _QUEUE_GROWTH_EPS_BITS
        or service_bits >= service_margin * arrival_bits
    )

    if queue_bits > _QUEUE_GROWTH_EPS_BITS and not service_margin_ok:
        state.poor_service_steps += 1
    else:
        state.poor_service_steps = 0

    entry_backlog = float(
        _planner_value(env, "ugv_comm_backlog_threshold", 0.5)
    )
    exit_backlog = float(
        _planner_value(env, "ugv_recovery_exit_backlog_threshold", 0.2)
    )
    poor_service_required = max(
        1,
        int(_planner_value(env, "ugv_recovery_poor_service_steps", 2)),
    )
    min_hold_steps = max(
        1,
        int(_planner_value(env, "ugv_recovery_min_hold_steps", 2)),
    )
    good_link_required = max(
        1,
        int(_planner_value(env, "ugv_recovery_good_link_steps", 2)),
    )

    state.mode_switch = ""
    state.switch_reason = ""
    if state.submode == "two_path":
        reason = ""
        if outage and queue_bits > _QUEUE_GROWTH_EPS_BITS:
            reason = "outage_queue"
        elif (
            backlog_norm >= entry_backlog
            and state.poor_service_steps >= poor_service_required
        ):
            reason = "backlog_poor_service"
        if reason:
            state.submode = "support_recovery"
            state.entered_step = current_step
            state.good_link_steps = 0
            state.switch_reason = reason
            state.mode_switch = "two_path->support_recovery"
    else:
        healthy = bool(
            (not outage)
            and backlog_norm <= exit_backlog
            and service_margin_ok
        )
        state.good_link_steps = state.good_link_steps + 1 if healthy else 0
        held_long_enough = (
            current_step - int(state.entered_step) + 1 >= min_hold_steps
        )
        if held_long_enough and state.good_link_steps >= good_link_required:
            state.submode = "two_path"
            state.entered_step = -1
            state.poor_service_steps = 0
            state.switch_reason = "recovered"
            state.mode_switch = "support_recovery->two_path"
            _clear_two_path_forecast(env)

    state.last_step = current_step
    state.backlog_norm = backlog_norm
    state.service_margin_ok = service_margin_ok

    env._ppo_astar_2path_recovery_submode = str(state.submode)
    env._ppo_astar_2path_recovery_switch = str(state.mode_switch)
    env._ppo_astar_2path_recovery_reason = str(state.switch_reason)
    env._ppo_astar_2path_recovery_poor_service_steps = int(
        state.poor_service_steps
    )
    env._ppo_astar_2path_recovery_good_link_steps = int(state.good_link_steps)
    env._ppo_astar_2path_recovery_service_margin_ok = int(
        state.service_margin_ok
    )
    return state


def select_two_path_support_recovery_ugv_action(
    env,
    support_target_grid: Optional[GridCell] = None,
) -> int:
    """Select one UGV action from the active predictive/recovery submode."""
    state = update_recovery_state(env)
    if state.submode == "support_recovery":
        target = support_target_grid
        if target is None:
            target, _, _, _ = env._get_ugv_guidance_target_grid()
        env._ppo_astar_2path_recovery_goal = target
        if target is None:
            return 0
        return select_astar_ugv_action(env, target)

    action = select_two_path_support_ugv_action(env)
    env._ppo_astar_2path_recovery_goal = getattr(
        env,
        "_ppo_astar_2path_support_goal",
        None,
    )
    return int(action)


__all__ = [
    "RecoveryState",
    "select_two_path_support_recovery_ugv_action",
    "update_recovery_state",
]

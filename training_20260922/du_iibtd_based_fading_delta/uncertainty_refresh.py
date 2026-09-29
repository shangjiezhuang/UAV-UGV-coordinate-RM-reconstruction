"""One uncertainty decision couples local/global exploration and full reconstruction."""
from typing import Tuple
import numpy as np


def compute_uncertainty_update_metrics(previous_uncertainty: np.ndarray,
                                       current_uncertainty: np.ndarray) -> Tuple[float, float]:
    """Return relative map change and signed reduction in Frobenius norm.

    This preserves the existing local-to-global metric exactly. Both callers
    use the entire ensemble variance tensor on the same fixed map domain.
    """
    previous = np.asarray(previous_uncertainty, dtype=float)
    current = np.asarray(current_uncertainty, dtype=float)
    if previous.shape != current.shape:
        raise ValueError(f"Uncertainty maps must have matching shapes, got {previous.shape} and {current.shape}.")
    previous_norm = float(np.linalg.norm(previous.ravel()))
    denominator = previous_norm + 1e-12
    relative_delta = float(np.linalg.norm((current - previous).ravel()) / denominator)
    relative_improvement = float((previous_norm - float(np.linalg.norm(current.ravel()))) / denominator)
    return relative_delta, relative_improvement


def uncertainty_norm(var_map: np.ndarray) -> float:
    values = np.asarray(var_map, dtype=float)
    if values.ndim != 3 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Reconstruction refresh requires a finite nonempty 3D uncertainty map")
    return float(np.linalg.norm(values.ravel()))


def append_uncertainty_window(history, previous_norm, current_norm, threshold=0.03):
    """Sum the last two signed relative improvements, preserving their signs.

    Exactly meeting the threshold is sufficient improvement. Only machine
    roundoff is absorbed at this boundary. One valid update cannot trigger.
    """
    values = list(history)
    if not values:
        values.append(float(previous_norm))
    elif not np.isclose(values[-1], previous_norm, rtol=1e-9, atol=1e-12):
        raise ValueError("Uncertainty window baseline does not match the previous map")
    values.append(float(current_norm))
    values = values[-3:]
    if len(values) < 3:
        return values, float("nan"), False
    improvement = sum((a-b)/(a+1e-12) for a,b in zip(values[:-1], values[1:]))
    stalled = improvement < threshold and not np.isclose(improvement, threshold, rtol=1e-9, atol=1e-12)
    return values, float(improvement), bool(stalled)


def update_hybrid_planner_submode(env, planner_submode_before_step, map_updated):
    """Shared production implementation for both observation representations."""
    if not env.hybrid_enabled or not env.planner_initialized:
        return
    improvement = float("nan")
    previous = env.previous_switch_uncertainty_map
    if map_updated:
        current = np.asarray(env.latest_var_map, dtype=float)
        delta = float("nan")
        if previous is not None and previous.shape == current.shape:
            delta, improvement = compute_uncertainty_update_metrics(previous, current)
        env.previous_switch_uncertainty_map = current.copy()
        env.last_uncertainty_map_delta = delta
        env.last_uncertainty_map_improvement = improvement

    if planner_submode_before_step == "local" and env.planner_submode == "local":
        if map_updated and np.isfinite(improvement):
            values, cumulative, stalled = append_uncertainty_window(
                env._local_uncertainty_norm_window,
                float(np.linalg.norm(previous.ravel())), float(np.linalg.norm(current.ravel())),
                env.hybrid_uncertainty_improvement_threshold,
            )
            env._local_uncertainty_norm_window = values
            env.last_uncertainty_improvement_sum = cumulative
            env.planner_stall_count = min(len(values)-1, 2)
            if stalled:
                switched = env._switch_planner_submode("global", hold_steps=env._get_hybrid_global_hold_steps())
                if not switched:
                    raise RuntimeError("A local-to-global transition was expected")
                env.radio_map_state = env.td.full_refit_for_mode_switch()
                env._sync_cached_ensemble_state()
                env.previous_switch_uncertainty_map = env.latest_var_map.copy()
                env._local_uncertainty_norm_window = []
                env.last_uncertainty_improvement_sum = cumulative
                if not env.ensemble_events or env.ensemble_events[-1]['step'] != env.current_step:
                    raise RuntimeError("A mode-switch refit must amend the current data-update event")
                event = env.ensemble_events[-1]
                delta_nmse = float(env.radio_map_state.nmse) - float(event['nmse'])
                event.update(recon_mode="ensemble_mode_switch_refresh", mode_switch_refresh_triggered=True,
                             nmse=float(env.radio_map_state.nmse), nmse_delta=float(event['nmse_delta'])+delta_nmse,
                             uncertainty_improvement_sum=cumulative,
                             uncertainty_switch_threshold=env.hybrid_uncertainty_improvement_threshold)
                env._snapshot_global_switch_top_targets(top_k=3)
                env._start_new_grid_plan()
                return

    if planner_submode_before_step == "global" and env.planner_submode == "global":
        env._local_uncertainty_norm_window = []
        env.global_steps_remaining = max(env.global_steps_remaining-1, 0)
        if (env.global_steps_remaining <= 0
                and env._count_local_reentry_candidates() >= env._get_hybrid_local_reentry_min_targets()):
            if env._switch_planner_submode("local"):
                env._local_uncertainty_norm_window = []
                env._start_new_grid_plan()

"""Greedy UAV action selection for the UAV-UGV environment.

The policy plans over the same macro-actions that the environment executes:
one action is one cardinal direction plus the configured UAV step length.  The
selected path must be able to reach the current planner/bootstrap target, and
among bounded reachable paths it prefers the largest accumulated uncertainty.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from environment import DIRECTION_OFFSETS


GridCell = Tuple[int, int]


@dataclass
class GreedyPolicyConfig:
    beam_width: int = 512
    max_extra_actions: int = 2
    revisit_penalty: float = 0.25
    sampled_revisit_penalty: float = 0.15
    progress_weight: float = 0.05
    arrival_bonus: float = 1.0
    building_uncertainty_scale: float = 0.0
    score_traversed_cells: bool = True


class GreedyPathPolicy:
    """Plan a high-uncertainty path and execute its first legal UAV action."""

    def __init__(self, config: Optional[GreedyPolicyConfig] = None, env=None):
        self.config = config if config is not None else GreedyPolicyConfig()
        self.env = env
        self.last_plan: Dict[str, object] = {}

    def bind_env(self, env) -> "GreedyPathPolicy":
        self.env = env
        return self

    def get_single_action(
        self,
        uav_obs=None,
        critic_state=None,
        uav_action_mask=None,
        deterministic: bool = True,
    ) -> Dict[str, object]:
        if self.env is None:
            raise RuntimeError("GreedyPathPolicy requires bind_env(env) before inference.")
        action = int(self.select_action(self.env, uav_action_mask=uav_action_mask))
        return {
            "uav_action": action,
            "uav_log_prob": 0.0,
            "value": 0.0,
        }

    def select_action(self, env, uav_action_mask=None) -> int:
        self.bind_env(env)
        self._ensure_active_env_plan(env)
        target = self._get_target_cell(env)
        uncertainty = self._build_spatial_uncertainty(env)
        action_mask = self._resolve_action_mask(env, uav_action_mask)

        if target is None:
            direction = self._select_one_step_uncertain_direction(
                env=env,
                uncertainty=uncertainty,
                target=None,
                action_mask=action_mask,
            )
            path_meta = {
                "mode": "one_step_no_target",
                "first_action_uncertainty": float("nan"),
            }
        else:
            direction, path_meta = self._select_path_direction(
                env=env,
                target=target,
                uncertainty=uncertainty,
                action_mask=action_mask,
            )

        landing_unc = float(path_meta.get("first_action_uncertainty", np.nan))
        bw_idx = self._select_bandwidth_choice(
            env,
            landing_uncertainty=landing_unc,
            uncertainty=uncertainty,
            target=target,
            action_mask=action_mask,
        )
        quant_idx = self._select_quant_choice(
            env,
            landing_uncertainty=landing_unc,
            uncertainty=uncertainty,
            target=target,
            action_mask=action_mask,
        )
        action = self._encode_action(env, direction_idx=direction, bw_idx=bw_idx, quant_idx=quant_idx)
        action = self._repair_action_if_masked(env, action, direction, bw_idx, quant_idx, action_mask)

        self.last_plan = {
            **path_meta,
            "selected_direction": int(direction),
            "selected_bw_choice_idx": int(bw_idx),
            "selected_quant_choice_idx": int(quant_idx) if quant_idx is not None else -1,
            "selected_action": int(action),
        }
        return int(action)

    def _ensure_active_env_plan(self, env) -> None:
        start_plan = getattr(env, "_start_new_grid_plan", None)
        if callable(start_plan):
            start_plan()

    def _get_target_cell(self, env) -> Optional[GridCell]:
        target_grid = None
        get_motion_target = getattr(env, "_get_motion_target_grid", None)
        if callable(get_motion_target):
            target_grid = get_motion_target()
        if target_grid is None:
            get_obs_target = getattr(env, "_get_current_observation_target", None)
            obs_target = get_obs_target() if callable(get_obs_target) else None
            if obs_target is not None:
                target_grid = (int(obs_target.gx), int(obs_target.gy))
        if target_grid is None:
            return None
        return self._clip_cell(env, target_grid)

    def _build_spatial_uncertainty(self, env) -> np.ndarray:
        if hasattr(env, "uncertainty") and hasattr(env.uncertainty, "spatial_uncertainty"):
            spatial = np.asarray(env.uncertainty.spatial_uncertainty, dtype=float)
        elif hasattr(env, "latest_var_map"):
            spatial = np.mean(np.asarray(env.latest_var_map, dtype=float), axis=2)
        else:
            spatial = np.ones((int(env.Nx), int(env.Ny)), dtype=float)

        spatial = np.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.ones_like(spatial, dtype=bool)
        if hasattr(env, "sampling_valid_mask"):
            mask = np.asarray(env.sampling_valid_mask, dtype=bool)
        valid_values = spatial[mask & np.isfinite(spatial)]
        if valid_values.size > 0:
            lo = float(np.min(valid_values))
            hi = float(np.max(valid_values))
        else:
            lo = float(np.min(spatial))
            hi = float(np.max(spatial))
        if hi > lo + 1e-12:
            spatial = (spatial - lo) / (hi - lo)
        else:
            spatial = np.zeros_like(spatial, dtype=float)

        if mask.shape == spatial.shape:
            scale = float(self.config.building_uncertainty_scale)
            spatial = np.where(mask, spatial, spatial * scale)
        return np.clip(spatial, 0.0, 1.0)

    def _resolve_action_mask(self, env, uav_action_mask=None) -> np.ndarray:
        if uav_action_mask is None:
            build_mask = getattr(env, "_build_uav_action_mask", None)
            if callable(build_mask):
                uav_action_mask = build_mask()
            else:
                uav_action_mask = np.ones(int(env.uav_action_size), dtype=bool)
        mask = np.asarray(uav_action_mask, dtype=bool).reshape(-1)
        if mask.size != int(env.uav_action_size):
            mask = np.ones(int(env.uav_action_size), dtype=bool)
        return mask

    def _select_path_direction(
        self,
        env,
        target: GridCell,
        uncertainty: np.ndarray,
        action_mask: np.ndarray,
    ) -> Tuple[int, Dict[str, object]]:
        current = self._current_cell(env)
        if current == target:
            if self._env_preserves_global_target(env):
                max_actions = max(2, int(self.config.max_extra_actions) + 1)
                candidates = self._beam_search_paths(
                    env=env,
                    current=current,
                    target=target,
                    uncertainty=uncertainty,
                    action_mask=action_mask,
                    max_actions=max_actions,
                )
                if candidates:
                    best = self._best_path_candidate(candidates)
                    first_direction = int(best["actions"][0])
                    return first_direction, {
                        "mode": "global_anchor_loop",
                        "target": target,
                        "max_macro_actions": int(max_actions),
                        "planned_actions": [int(a) for a in best["actions"]],
                        "planned_score": float(best["score"]),
                        "planned_cells": [list(cell) for cell in best["cells"]],
                        "first_action_uncertainty": float(best["first_action_uncertainty"]),
                    }
                direction = self._select_one_step_uncertain_direction(
                    env=env,
                    uncertainty=uncertainty,
                    target=None,
                    action_mask=action_mask,
                )
                first_unc = self._direction_mean_uncertainty(env, direction, None, uncertainty)
                return direction, {
                    "mode": "global_anchor_explore",
                    "target": target,
                    "planned_actions": [int(direction)],
                    "planned_score": float("nan"),
                    "first_action_uncertainty": float(first_unc),
                }
            stay_dir = 0 if self._direction_is_available(env, 0, action_mask) else self._first_valid_direction(env, action_mask)
            return stay_dir, {
                "mode": "already_at_target",
                "target": target,
                "planned_actions": [int(stay_dir)],
                "planned_score": 0.0,
                "first_action_uncertainty": float(uncertainty[current]),
            }

        shortest = self._shortest_macro_action_count(env, current, target, action_mask)
        if shortest is None:
            direction = self._select_one_step_uncertain_direction(
                env=env,
                uncertainty=uncertainty,
                target=target,
                action_mask=action_mask,
            )
            first_unc = self._direction_mean_uncertainty(env, direction, target, uncertainty)
            return direction, {
                "mode": "fallback_no_reachable_target_path",
                "target": target,
                "planned_actions": [int(direction)],
                "planned_score": float("nan"),
                "first_action_uncertainty": float(first_unc),
            }

        max_actions = max(1, int(shortest) + max(0, int(self.config.max_extra_actions)))
        candidates = self._beam_search_paths(
            env=env,
            current=current,
            target=target,
            uncertainty=uncertainty,
            action_mask=action_mask,
            max_actions=max_actions,
        )
        if not candidates:
            direction = self._select_one_step_uncertain_direction(
                env=env,
                uncertainty=uncertainty,
                target=target,
                action_mask=action_mask,
            )
            first_unc = self._direction_mean_uncertainty(env, direction, target, uncertainty)
            return direction, {
                "mode": "fallback_beam_miss",
                "target": target,
                "shortest_macro_actions": int(shortest),
                "planned_actions": [int(direction)],
                "planned_score": float("nan"),
                "first_action_uncertainty": float(first_unc),
            }

        best = self._best_path_candidate(candidates)
        first_direction = int(best["actions"][0])
        return first_direction, {
            "mode": "beam_path_to_target",
            "target": target,
            "shortest_macro_actions": int(shortest),
            "max_macro_actions": int(max_actions),
            "planned_actions": [int(a) for a in best["actions"]],
            "planned_score": float(best["score"]),
            "planned_cells": [list(cell) for cell in best["cells"]],
            "first_action_uncertainty": float(best["first_action_uncertainty"]),
        }

    @staticmethod
    def _best_path_candidate(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
        return max(
            candidates,
            key=lambda item: (
                float(item["score"]),
                -len(item["actions"]),
                float(item["first_action_uncertainty"]),
            ),
        )

    @staticmethod
    def _env_preserves_global_target(env) -> bool:
        preserve_fn = getattr(env, "_should_preserve_global_target", None)
        if not callable(preserve_fn):
            return False
        try:
            return bool(preserve_fn())
        except Exception:
            return False

    def _beam_search_paths(
        self,
        env,
        current: GridCell,
        target: GridCell,
        uncertainty: np.ndarray,
        action_mask: np.ndarray,
        max_actions: int,
    ) -> List[Dict[str, object]]:
        beam = [
            {
                "pos": current,
                "actions": [],
                "cells": [current],
                "visited": frozenset([current]),
                "score": 0.0,
                "first_action_uncertainty": float("nan"),
            }
        ]
        candidates: List[Dict[str, object]] = []
        beam_width = max(1, int(self.config.beam_width))

        for _depth in range(max_actions):
            expanded: List[Dict[str, object]] = []
            for item in beam:
                for direction in self._available_directions(env, action_mask, include_stay=False):
                    rollout = self._rollout_macro_action(env, item["pos"], int(direction), target)
                    next_cell = rollout["next_cell"]
                    if next_cell == item["pos"]:
                        continue
                    if next_cell in item["visited"] and next_cell != target:
                        continue

                    step_score = self._score_rollout_cells(
                        env=env,
                        cells=rollout["cells"],
                        uncertainty=uncertainty,
                        visited=item["visited"],
                    )
                    next_actions = [*item["actions"], int(direction)]
                    next_cells = [*item["cells"], *rollout["cells"]]
                    first_unc = item["first_action_uncertainty"]
                    if not np.isfinite(float(first_unc)):
                        first_unc = self._mean_uncertainty_for_cells(rollout["cells"], uncertainty)

                    next_item = {
                        "pos": next_cell,
                        "actions": next_actions,
                        "cells": next_cells,
                        "visited": frozenset(set(item["visited"]).union(rollout["cells"])),
                        "score": float(item["score"]) + float(step_score),
                        "first_action_uncertainty": float(first_unc),
                    }
                    if next_cell == target:
                        next_item["score"] = float(next_item["score"]) + float(self.config.arrival_bonus)
                        candidates.append(next_item)
                    else:
                        expanded.append(next_item)

            if not expanded:
                break

            expanded.sort(
                key=lambda item: (
                    float(item["score"]) + self._progress_rank(env, item["pos"], target),
                    -len(item["actions"]),
                ),
                reverse=True,
            )
            beam = expanded[:beam_width]

        return candidates

    def _shortest_macro_action_count(
        self,
        env,
        current: GridCell,
        target: GridCell,
        action_mask: np.ndarray,
    ) -> Optional[int]:
        queue = deque([(current, 0)])
        seen = {current}
        while queue:
            cell, depth = queue.popleft()
            if cell == target:
                return int(depth)
            for direction in self._available_directions(env, action_mask, include_stay=False):
                rollout = self._rollout_macro_action(env, cell, int(direction), target)
                next_cell = rollout["next_cell"]
                if next_cell in seen or next_cell == cell:
                    continue
                if next_cell == target:
                    return int(depth + 1)
                seen.add(next_cell)
                queue.append((next_cell, depth + 1))
        return None

    def _select_one_step_uncertain_direction(
        self,
        env,
        uncertainty: np.ndarray,
        target: Optional[GridCell],
        action_mask: np.ndarray,
    ) -> int:
        current = self._current_cell(env)
        current_dist = self._manhattan(current, target) if target is not None else 0
        best_direction = self._first_valid_direction(env, action_mask)
        best_key: Optional[Tuple[float, ...]] = None
        for direction in self._available_directions(env, action_mask, include_stay=False):
            rollout = self._rollout_macro_action(env, current, int(direction), target)
            next_cell = rollout["next_cell"]
            if next_cell == current:
                continue
            unc_score = self._score_rollout_cells(
                env=env,
                cells=rollout["cells"],
                uncertainty=uncertainty,
                visited=frozenset([current]),
            )
            progress = 0.0
            arrival = 0.0
            if target is not None:
                next_dist = self._manhattan(next_cell, target)
                progress = float(current_dist - next_dist)
                arrival = 1.0 if next_cell == target else 0.0
            key = (
                float(unc_score) + float(self.config.progress_weight) * progress,
                arrival,
                progress,
                -float(self._distance_from_uav(env, next_cell)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_direction = int(direction)
        return int(best_direction)

    def _rollout_macro_action(
        self,
        env,
        position: GridCell,
        direction_idx: int,
        target: Optional[GridCell],
    ) -> Dict[str, object]:
        offset = np.asarray(DIRECTION_OFFSETS.get(int(direction_idx), np.array([0, 0])), dtype=float)
        pos = np.asarray(position, dtype=float)
        if np.allclose(offset, 0.0):
            cell = self._clip_cell(env, pos)
            return {"next_cell": cell, "cells": [cell], "moved_steps": 0}

        cells: List[GridCell] = []
        new_pos = pos.copy()
        moved_steps = 0
        step_count = int(getattr(env, "uav_step_count", int(env.config.uav.step_size)))
        for _ in range(max(1, step_count)):
            proposed = new_pos + offset
            if not env.scene.is_uav_position_valid(proposed):
                break
            new_pos = proposed
            moved_steps += 1
            cell = self._clip_cell(env, new_pos)
            cells.append(cell)
            if target is not None and cell == target:
                break

        next_cell = self._clip_cell(env, new_pos)
        if not cells:
            cells = [next_cell]
        return {"next_cell": next_cell, "cells": cells, "moved_steps": int(moved_steps)}

    def _score_rollout_cells(
        self,
        env,
        cells: Sequence[GridCell],
        uncertainty: np.ndarray,
        visited: Iterable[GridCell],
    ) -> float:
        visited_set = set(visited)
        score = 0.0
        scored_cells = cells if bool(self.config.score_traversed_cells) else cells[-1:]
        for cell in scored_cells:
            gx, gy = int(cell[0]), int(cell[1])
            cell_score = float(uncertainty[gx, gy])
            if cell in visited_set:
                cell_score -= float(self.config.revisit_penalty)
            if hasattr(env, "local_spatial_visit"):
                visits = float(env.local_spatial_visit[gx, gy])
                if visits > 0.0:
                    cell_score -= float(self.config.sampled_revisit_penalty) * np.log1p(visits)
            score += cell_score
        return float(score)

    def _direction_mean_uncertainty(
        self,
        env,
        direction_idx: int,
        target: Optional[GridCell],
        uncertainty: np.ndarray,
    ) -> float:
        rollout = self._rollout_macro_action(env, self._current_cell(env), int(direction_idx), target)
        return self._mean_uncertainty_for_cells(rollout["cells"], uncertainty)

    def _mean_uncertainty_for_cells(
        self,
        cells: Sequence[GridCell],
        uncertainty: np.ndarray,
    ) -> float:
        if not cells:
            return float("nan")
        values = [float(uncertainty[int(gx), int(gy)]) for gx, gy in cells]
        return float(np.mean(values)) if values else float("nan")

    def _select_bandwidth_choice(
        self,
        env,
        landing_uncertainty: float,
        uncertainty: Optional[np.ndarray] = None,
        target: Optional[GridCell] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        ratios = np.asarray(getattr(env, "bandwidth_ratios", [getattr(env, "current_bw_ratio", 0.5)]), dtype=float)
        if ratios.size <= 0:
            return 0
        uav_config = getattr(getattr(env, "config", None), "uav", None)
        default_ratio = (
            float(uav_config.default_bw_ratio)
            if hasattr(uav_config, "default_bw_ratio")
            else float(getattr(env, "current_bw_ratio", np.max(ratios)))
        )
        if not np.isfinite(float(landing_uncertainty)):
            return int(np.argmin(np.abs(ratios - default_ratio)))
        uncertainty_rank = self._adaptive_uncertainty_rank(
            env=env,
            landing_uncertainty=landing_uncertainty,
            uncertainty=uncertainty,
            target=target,
            action_mask=action_mask,
        )
        return self._select_uncertainty_ranked_index(ratios, uncertainty_rank)

    def _select_quant_choice(
        self,
        env,
        landing_uncertainty: float,
        uncertainty: Optional[np.ndarray] = None,
        target: Optional[GridCell] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        if not hasattr(env, "num_quant_choices") or not hasattr(env, "quant_bits"):
            return None
        quant_bits = np.asarray(env.quant_bits, dtype=int)
        if quant_bits.size <= 0:
            return 0
        if not np.isfinite(float(landing_uncertainty)):
            target_bits = int(getattr(env, "default_quant_bits", int(np.max(quant_bits))))
            return int(np.argmin(np.abs(quant_bits.astype(float) - float(target_bits))))
        uncertainty_rank = self._adaptive_uncertainty_rank(
            env=env,
            landing_uncertainty=landing_uncertainty,
            uncertainty=uncertainty,
            target=target,
            action_mask=action_mask,
        )
        return self._select_uncertainty_ranked_index(quant_bits.astype(float), uncertainty_rank)

    def _adaptive_uncertainty_rank(
        self,
        env,
        landing_uncertainty: float,
        uncertainty: Optional[np.ndarray],
        target: Optional[GridCell],
        action_mask: Optional[np.ndarray],
    ) -> float:
        """Map local uncertainty to a relative rank before choosing resource levels."""
        landing_uncertainty = float(landing_uncertainty)
        if not np.isfinite(landing_uncertainty):
            return float("nan")

        local_values: List[float] = []
        if uncertainty is not None and action_mask is not None:
            for direction in self._available_directions(env, action_mask, include_stay=False):
                value = self._direction_mean_uncertainty(env, int(direction), target, uncertainty)
                if np.isfinite(float(value)):
                    local_values.append(float(value))
        local_rank = self._rank_against_values(landing_uncertainty, local_values)
        if np.isfinite(local_rank):
            return float(local_rank)

        if uncertainty is not None:
            values = np.asarray(uncertainty, dtype=float)
        else:
            values = self._build_spatial_uncertainty(env)
        mask = np.ones(values.shape, dtype=bool)
        if hasattr(env, "sampling_valid_mask") and np.asarray(env.sampling_valid_mask).shape == values.shape:
            mask = np.asarray(env.sampling_valid_mask, dtype=bool)
        global_rank = self._rank_against_values(
            landing_uncertainty,
            values[mask & np.isfinite(values)].reshape(-1),
        )
        if np.isfinite(global_rank):
            return float(global_rank)
        return float(np.clip(landing_uncertainty, 0.0, 1.0))

    @staticmethod
    def _rank_against_values(value: float, values: Iterable[float]) -> float:
        values_arr = np.asarray(list(values), dtype=float).reshape(-1)
        values_arr = values_arr[np.isfinite(values_arr)]
        if values_arr.size < 2:
            return float("nan")
        values_arr.sort()
        if float(values_arr[-1] - values_arr[0]) <= 1e-12:
            return float("nan")
        value = float(value)
        if value <= float(values_arr[0]):
            return 0.0
        if value >= float(values_arr[-1]):
            return 1.0
        return float(np.searchsorted(values_arr, value, side="left") / max(values_arr.size - 1, 1))

    @staticmethod
    def _select_uncertainty_ranked_index(values: np.ndarray, uncertainty: float) -> int:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size <= 0:
            return 0
        order = np.argsort(values)
        uncertainty_norm = float(np.clip(float(uncertainty), 0.0, 1.0))
        rank = min(int(np.floor(uncertainty_norm * values.size)), values.size - 1)
        return int(order[rank])

    def _encode_action(self, env, direction_idx: int, bw_idx: int, quant_idx: Optional[int]) -> int:
        direction_ids = [int(v) for v in getattr(env, "uav_direction_ids", [1, 2, 3, 4, 0])]
        if int(direction_idx) not in direction_ids:
            direction_idx = direction_ids[0]
        direction_choice_idx = int(direction_ids.index(int(direction_idx)))
        num_bw = int(getattr(env, "num_bw_choices", 1))
        bw_idx = int(np.clip(bw_idx, 0, max(0, num_bw - 1)))
        if hasattr(env, "num_quant_choices"):
            num_quant = int(getattr(env, "num_quant_choices", 1))
            q_idx = 0 if quant_idx is None else int(np.clip(quant_idx, 0, max(0, num_quant - 1)))
            return int((direction_choice_idx * num_bw + bw_idx) * num_quant + q_idx)
        return int(direction_choice_idx * num_bw + bw_idx)

    def _repair_action_if_masked(
        self,
        env,
        action: int,
        direction_idx: int,
        bw_idx: int,
        quant_idx: Optional[int],
        action_mask: np.ndarray,
    ) -> int:
        action = int(action)
        if 0 <= action < action_mask.size and bool(action_mask[action]):
            return action

        for candidate_bw in self._preferred_bandwidth_order(env, bw_idx):
            if hasattr(env, "num_quant_choices"):
                for candidate_quant in self._preferred_quant_order(env, quant_idx):
                    candidate = self._encode_action(env, direction_idx, candidate_bw, candidate_quant)
                    if 0 <= candidate < action_mask.size and bool(action_mask[candidate]):
                        return int(candidate)
            else:
                candidate = self._encode_action(env, direction_idx, candidate_bw, None)
                if 0 <= candidate < action_mask.size and bool(action_mask[candidate]):
                    return int(candidate)

        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size:
            return int(valid_actions[0])
        return 0

    def _preferred_bandwidth_order(self, env, first_idx: int) -> List[int]:
        count = int(getattr(env, "num_bw_choices", 1))
        first_idx = int(np.clip(first_idx, 0, max(0, count - 1)))
        rest = [idx for idx in range(count) if idx != first_idx]
        return [first_idx, *rest]

    def _preferred_quant_order(self, env, first_idx: Optional[int]) -> List[int]:
        count = int(getattr(env, "num_quant_choices", 1))
        if first_idx is None:
            first_idx = 0
        first_idx = int(np.clip(first_idx, 0, max(0, count - 1)))
        rest = [idx for idx in range(count) if idx != first_idx]
        return [first_idx, *rest]

    def _available_directions(self, env, action_mask: np.ndarray, include_stay: bool) -> List[int]:
        directions = []
        for direction in getattr(env, "uav_direction_ids", [1, 2, 3, 4, 0]):
            direction = int(direction)
            if not include_stay and direction == 0:
                continue
            if self._direction_is_available(env, direction, action_mask):
                directions.append(direction)
        if not directions and include_stay and self._direction_is_available(env, 0, action_mask):
            directions.append(0)
        return directions

    def _direction_is_available(self, env, direction_idx: int, action_mask: np.ndarray) -> bool:
        num_bw = int(getattr(env, "num_bw_choices", 1))
        num_quant = int(getattr(env, "num_quant_choices", 1))
        direction_ids = [int(v) for v in getattr(env, "uav_direction_ids", [1, 2, 3, 4, 0])]
        if int(direction_idx) not in direction_ids:
            return False
        d_idx = direction_ids.index(int(direction_idx))
        start = d_idx * num_bw * num_quant
        end = start + num_bw * num_quant
        return bool(np.any(action_mask[start:end]))

    def _first_valid_direction(self, env, action_mask: np.ndarray) -> int:
        for direction in getattr(env, "uav_direction_ids", [1, 2, 3, 4, 0]):
            if self._direction_is_available(env, int(direction), action_mask):
                return int(direction)
        return 0

    def _current_cell(self, env) -> GridCell:
        return self._clip_cell(env, np.asarray(env.uav_pos, dtype=float))

    def _clip_cell(self, env, cell) -> GridCell:
        arr = np.asarray(cell, dtype=float).reshape(2)
        gx = int(np.clip(np.rint(arr[0]), 0, int(env.Nx) - 1))
        gy = int(np.clip(np.rint(arr[1]), 0, int(env.Ny) - 1))
        return gx, gy

    def _manhattan(self, a: GridCell, b: Optional[GridCell]) -> int:
        if b is None:
            return 0
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    def _progress_rank(self, env, cell: GridCell, target: GridCell) -> float:
        denom = float(max(int(env.Nx) + int(env.Ny) - 2, 1))
        return -float(self.config.progress_weight) * float(self._manhattan(cell, target)) / denom

    def _distance_from_uav(self, env, cell: GridCell) -> float:
        current = self._current_cell(env)
        return float(self._manhattan(current, cell))

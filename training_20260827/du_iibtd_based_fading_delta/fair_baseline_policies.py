"""Non-learning baselines that execute inside the frozen based environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from du_iibtd_based_fading_delta.Greedy_heuristic_quant.greedy_policy import GreedyPathPolicy
from du_iibtd_based_fading_delta.ugv_control import uses_service_action_mask


BASELINE_CONTROLLER_ALIASES = {
    "random": "total_random",
    "greedy": "greedy_legacy",
    "greedy_2path": "greedy_astar_2path_support",
}
BASELINE_UGV_CONTROL_MODES = {
    "total_random": "policy",
    "greedy_guidance": "policy",
    "greedy_legacy": "legacy_heuristic",
    "greedy_astar_2path_support": "astar_2path_support",
}
BASELINE_CONTROLLERS = (
    *BASELINE_UGV_CONTROL_MODES,
    *BASELINE_CONTROLLER_ALIASES,
)


def normalize_baseline_controller(value: str) -> str:
    controller = BASELINE_CONTROLLER_ALIASES.get(str(value), str(value))
    if controller not in BASELINE_UGV_CONTROL_MODES:
        raise ValueError(f"Unsupported baseline controller: {value!r}")
    return controller


def ugv_control_mode_for_baseline(value: str) -> str:
    return BASELINE_UGV_CONTROL_MODES[normalize_baseline_controller(value)]


def _valid_actions(mask: Any, expected_size: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size != int(expected_size):
        raise ValueError(f"Expected action-mask width {expected_size}, got {values.size}")
    valid = np.flatnonzero(values)
    if valid.size == 0:
        raise RuntimeError("Action mask has no valid actions")
    return valid


def _guidance_greedy_ugv_action(env: Any, ugv_action_mask: Any) -> int:
    valid = _valid_actions(ugv_action_mask, int(env.ugv_action_size))
    target_grid, _, _, _ = env._get_ugv_guidance_target_grid()
    if target_grid is None:
        return int(valid[0])

    target = np.asarray(target_grid, dtype=float)
    candidates = []
    for action in valid.tolist():
        if hasattr(env, "config") and uses_service_action_mask(env.config):
            next_position, _, distance = env._rollout_ugv_toward_target(
                int(action),
                target_grid,
            )
        else:
            next_position, _ = env._rollout_direction(
                position=env.ugv_pos,
                direction_idx=int(action),
                step_count=env.ugv_step_count,
                validator=env.scene.is_ugv_position_valid,
            )
            distance = env._occupancy_shortest_path_distance(next_position, target)
        candidates.append(
            (
                float(distance),
                0 if int(action) == 0 else 1,
                int(action),
            )
        )
    return int(min(candidates)[2])


class RandomValidPolicy:
    """Uniformly sample both agents from the exact environment action masks."""

    def __init__(self, seed: int = 0):
        self.base_seed = int(seed)
        self.rng = np.random.RandomState(self.base_seed)
        self.env: Optional[Any] = None

    def on_episode_reset(self, reset_seed: Optional[int], env: Any) -> None:
        self.env = env
        seed = self.base_seed if reset_seed is None else int(reset_seed)
        self.rng = np.random.RandomState(seed)

    def get_single_action(
        self,
        *,
        uav_action_mask: Any,
        ugv_action_mask: Any,
        **_: Any,
    ) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("RandomValidPolicy must receive on_episode_reset first")
        uav_valid = _valid_actions(uav_action_mask, int(self.env.uav_action_size))
        ugv_valid = _valid_actions(ugv_action_mask, int(self.env.ugv_action_size))
        return {
            "uav_action": int(self.rng.choice(uav_valid)),
            "ugv_action": int(self.rng.choice(ugv_valid)),
            "value": 0.0,
        }


class GreedyUAVPolicy:
    """Uncertainty-path greedy UAV with a dummy UGV command."""

    def __init__(self):
        self.env: Optional[Any] = None
        self.uav_policy = GreedyPathPolicy()

    def on_episode_reset(self, reset_seed: Optional[int], env: Any) -> None:
        del reset_seed
        self.env = env
        self.uav_policy.bind_env(env)

    def get_single_action(
        self,
        *,
        uav_action_mask: Any,
        ugv_action_mask: Any,
        **_: Any,
    ) -> Dict[str, Any]:
        del ugv_action_mask
        if self.env is None:
            raise RuntimeError("GreedyUAVPolicy must receive on_episode_reset first")
        return {
            "uav_action": int(
                self.uav_policy.select_action(
                    self.env,
                    uav_action_mask=uav_action_mask,
                )
            ),
            "ugv_action": 0,
            "value": 0.0,
        }


class GreedyGuidancePolicy(GreedyUAVPolicy):
    """Uncertainty-path greedy UAV plus communication-guidance greedy UGV."""

    def get_single_action(
        self,
        *,
        uav_action_mask: Any,
        ugv_action_mask: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        action_data = super().get_single_action(
            uav_action_mask=uav_action_mask,
            ugv_action_mask=ugv_action_mask,
            **kwargs,
        )
        action_data["ugv_action"] = _guidance_greedy_ugv_action(
            self.env,
            ugv_action_mask,
        )
        return action_data


class GuidanceGreedyUGVOverridePolicy:
    """Keep a learned UAV action and replace only its UGV command."""

    def __init__(self, policy: Any, env: Any):
        self.policy = policy
        self.env = env

    def get_single_action(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        action_data = dict(self.policy.get_single_action(*args, **kwargs))
        target_grid, _, _, _ = self.env._get_ugv_guidance_target_grid()
        if target_grid is None:
            return action_data
        ugv_action_mask = kwargs.get("ugv_action_mask")
        if ugv_action_mask is None and len(args) >= 5:
            ugv_action_mask = args[4]
        if ugv_action_mask is None:
            ugv_action_mask = np.ones(int(self.env.ugv_action_size), dtype=bool)
        action_data["ugv_action"] = _guidance_greedy_ugv_action(
            self.env,
            ugv_action_mask,
        )
        return action_data


def make_baseline_policy(controller: str, seed: int = 0):
    """Build one canonical non-learning policy for the shared evaluators."""
    controller = normalize_baseline_controller(controller)
    if controller == "total_random":
        return RandomValidPolicy(seed=seed)
    if controller in {"greedy_legacy", "greedy_astar_2path_support"}:
        return GreedyUAVPolicy()
    if controller == "greedy_guidance":
        return GreedyGuidancePolicy()
    raise ValueError(f"Unsupported baseline controller: {controller!r}")


__all__ = [
    "BASELINE_CONTROLLERS",
    "GreedyGuidancePolicy",
    "GreedyUAVPolicy",
    "GuidanceGreedyUGVOverridePolicy",
    "RandomValidPolicy",
    "make_baseline_policy",
    "normalize_baseline_controller",
    "ugv_control_mode_for_baseline",
]

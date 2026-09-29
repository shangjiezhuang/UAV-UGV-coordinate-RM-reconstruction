"""Shared trainer/controller mapping for the fair-baseline protocol."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


LEARNED_TRAINERS = (
    "mappo",
    "happo",
    "mappo_cf",
    "ippo",
    "ppo_fixed",
    "ppo_heuristic",
    "ppo_astar",
    "ppo_astar_support",
    "ppo_astar_2path_support",
    "ppo_astar_2path_support_recovery",
)
_UGV_CONTROL_BY_TRAINER = {
    "mappo": "policy",
    "happo": "policy",
    "mappo_cf": "policy",
    "ippo": "policy",
    "ppo_fixed": "fixed",
    "ppo_heuristic": "legacy_heuristic",
    # Keep the historical target-following PPO+A* trainer available as the
    # explicit support-disabled control.  The two support variants below use
    # the same UAV-only PPO architecture and differ only in how the UGV goal
    # is selected before the shared A* router is called.
    "ppo_astar": "astar_target",
    "ppo_astar_support": "astar_support",
    "ppo_astar_2path_support": "astar_2path_support",
    "ppo_astar_2path_support_recovery": "astar_2path_support_recovery",
}


def normalize_trainer(value: Any) -> str:
    trainer = str(value).strip().lower()
    if trainer not in _UGV_CONTROL_BY_TRAINER:
        raise ValueError(
            f"trainer must be one of {list(LEARNED_TRAINERS)}, got {value!r}"
        )
    return trainer


def ugv_control_mode_for_trainer(value: Any) -> str:
    return _UGV_CONTROL_BY_TRAINER[normalize_trainer(value)]


def is_independent_ppo(value: Any) -> bool:
    return normalize_trainer(value) == "ippo"


def is_uav_only_ppo(value: Any) -> bool:
    return normalize_trainer(value) in {
        "ppo_fixed",
        "ppo_heuristic",
        "ppo_astar",
        "ppo_astar_support",
        "ppo_astar_2path_support",
        "ppo_astar_2path_support_recovery",
    }


def training_reward_for_trainer(
    value: Any,
    team_reward: np.ndarray,
    infos: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return the rollout reward controlled by the selected trainer.

    The legacy-heuristic UGV is owned by the environment, so its progress reward
    is not controllable by the UAV-only PPO actor.  Other trainers retain the
    original cooperative team reward, including the fixed-UGV ablation.
    """
    if ugv_control_mode_for_trainer(value) not in {
        "legacy_heuristic",
        "astar_target",
        "astar_support",
        "astar_2path_support",
        "astar_2path_support_recovery",
    }:
        return team_reward

    reward_array = np.asarray(team_reward)
    ugv_progress = np.asarray(
        [float(info["r_ugv_progress"]) for info in infos],
        dtype=reward_array.dtype,
    )
    if reward_array.shape != ugv_progress.shape:
        raise ValueError(
            "team_reward and r_ugv_progress must have matching vector shapes, "
            f"got {reward_array.shape} and {ugv_progress.shape}"
        )
    return reward_array - ugv_progress


__all__ = [
    "LEARNED_TRAINERS",
    "is_independent_ppo",
    "is_uav_only_ppo",
    "normalize_trainer",
    "training_reward_for_trainer",
    "ugv_control_mode_for_trainer",
]

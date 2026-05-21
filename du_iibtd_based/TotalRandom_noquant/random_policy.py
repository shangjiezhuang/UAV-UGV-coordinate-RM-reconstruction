"""Uniform random legal-action policy for the total-random baseline."""

from __future__ import annotations

from typing import Optional

import numpy as np


class RandomPolicy:
    """Sample UAV/UGV actions uniformly from the legal entries in each mask."""

    def __init__(self, action_dims: dict, seed: Optional[int] = None):
        self.uav_action_dim = int(action_dims["uav_action"])
        self.ugv_action_dim = int(action_dims["ugv_action"])
        self.initial_seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.initial_seed)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        next_seed = self.initial_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(next_seed)

    @staticmethod
    def _as_batched_mask(
        action_mask: Optional[np.ndarray],
        action_dim: int,
        batch_size: int,
    ) -> np.ndarray:
        if action_mask is None:
            return np.ones((batch_size, action_dim), dtype=bool)

        mask = np.asarray(action_mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask.reshape(1, -1)
        if mask.shape[-1] != action_dim:
            raise ValueError(f"Expected mask width {action_dim}, got {mask.shape[-1]}")
        if mask.shape[0] == 1 and batch_size > 1:
            mask = np.repeat(mask, batch_size, axis=0)
        if mask.shape[0] != batch_size:
            raise ValueError(f"Expected mask batch {batch_size}, got {mask.shape[0]}")
        return mask

    @staticmethod
    def _infer_batch_size(*arrays: Optional[np.ndarray]) -> int:
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim >= 2:
                return int(arr.shape[0])
        return 1

    def _sample_from_mask(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.zeros(mask.shape[0], dtype=np.int64)
        log_probs = np.zeros(mask.shape[0], dtype=np.float32)
        entropies = np.zeros(mask.shape[0], dtype=np.float32)

        for row_idx, row in enumerate(mask):
            valid = np.flatnonzero(row)
            if valid.size == 0:
                raise ValueError(
                    f"Action mask row {row_idx} has no legal actions for dimension {row.shape[0]}"
                )
            choice = int(self.rng.choice(valid))
            actions[row_idx] = choice
            log_probs[row_idx] = -float(np.log(valid.size))
            entropies[row_idx] = float(np.log(valid.size))

        return actions, log_probs, entropies

    def get_actions(
        self,
        uav_obs: Optional[np.ndarray] = None,
        ugv_obs: Optional[np.ndarray] = None,
        uav_action_mask: Optional[np.ndarray] = None,
        ugv_action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        **_: object,
    ) -> dict:
        del deterministic
        batch_size = self._infer_batch_size(uav_action_mask, ugv_action_mask, uav_obs, ugv_obs)
        uav_mask = self._as_batched_mask(uav_action_mask, self.uav_action_dim, batch_size)
        ugv_mask = self._as_batched_mask(ugv_action_mask, self.ugv_action_dim, batch_size)

        uav_action, uav_log_prob, uav_entropy = self._sample_from_mask(uav_mask)
        ugv_action, ugv_log_prob, ugv_entropy = self._sample_from_mask(ugv_mask)
        return {
            "uav_action": uav_action,
            "ugv_action": ugv_action,
            "uav_log_prob": uav_log_prob,
            "ugv_log_prob": ugv_log_prob,
            "uav_entropy": uav_entropy,
            "ugv_entropy": ugv_entropy,
        }

    def get_single_action(
        self,
        uav_obs: Optional[np.ndarray] = None,
        ugv_obs: Optional[np.ndarray] = None,
        uav_action_mask: Optional[np.ndarray] = None,
        ugv_action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        **kwargs: object,
    ) -> dict:
        action_data = self.get_actions(
            uav_obs=None if uav_obs is None else np.asarray(uav_obs)[np.newaxis, ...],
            ugv_obs=None if ugv_obs is None else np.asarray(ugv_obs)[np.newaxis, ...],
            uav_action_mask=(
                None if uav_action_mask is None else np.asarray(uav_action_mask)[np.newaxis, ...]
            ),
            ugv_action_mask=(
                None if ugv_action_mask is None else np.asarray(ugv_action_mask)[np.newaxis, ...]
            ),
            deterministic=deterministic,
            **kwargs,
        )
        return {
            "uav_action": int(action_data["uav_action"][0]),
            "ugv_action": int(action_data["ugv_action"][0]),
            "uav_log_prob": float(action_data["uav_log_prob"][0]),
            "ugv_log_prob": float(action_data["ugv_log_prob"][0]),
            "uav_entropy": float(action_data["uav_entropy"][0]),
            "ugv_entropy": float(action_data["ugv_entropy"][0]),
        }

"""
Rollout buffer for IPPO training.

Stores transitions during rollout and computes per-agent GAE advantages.
"""

import numpy as np
import torch
from typing import Dict, Generator


class RolloutBuffer:
    """
    Buffer for independent PPO updates for UAV and UGV.

    Stores separate observations, rewards, value estimates, returns, and
    advantages for each agent.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_dims: Dict[str, int],
        action_dims: Dict[str, int],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_size = rollout_length * num_envs

        self.uav_obs = np.zeros(
            (rollout_length, num_envs, obs_dims["uav_obs"]), dtype=np.float32
        )
        self.ugv_obs = np.zeros(
            (rollout_length, num_envs, obs_dims["ugv_obs"]), dtype=np.float32
        )

        self.uav_actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.ugv_actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.uav_log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.uav_action_masks = np.zeros(
            (rollout_length, num_envs, action_dims["uav_action"]), dtype=bool
        )
        self.ugv_action_masks = np.zeros(
            (rollout_length, num_envs, action_dims["ugv_action"]), dtype=bool
        )

        self.uav_rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.uav_values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.terminated = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.truncated = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.uav_timeout_values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_timeout_values = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.uav_advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.uav_returns = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.step = 0
        self._flat_tensors: dict | None = None

    def add(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        uav_action: np.ndarray,
        ugv_action: np.ndarray,
        uav_log_prob: np.ndarray,
        ugv_log_prob: np.ndarray,
        uav_action_mask: np.ndarray,
        ugv_action_mask: np.ndarray,
        uav_reward: np.ndarray,
        ugv_reward: np.ndarray,
        uav_value: np.ndarray,
        ugv_value: np.ndarray,
        done: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        uav_timeout_value: np.ndarray,
        ugv_timeout_value: np.ndarray,
    ):
        """Add a transition to the buffer."""
        assert self.step < self.rollout_length, "Buffer is full"

        self.uav_obs[self.step] = uav_obs
        self.ugv_obs[self.step] = ugv_obs
        self.uav_actions[self.step] = uav_action
        self.ugv_actions[self.step] = ugv_action
        self.uav_log_probs[self.step] = uav_log_prob
        self.ugv_log_probs[self.step] = ugv_log_prob
        self.uav_action_masks[self.step] = uav_action_mask
        self.ugv_action_masks[self.step] = ugv_action_mask
        self.uav_rewards[self.step] = uav_reward
        self.ugv_rewards[self.step] = ugv_reward
        self.uav_values[self.step] = uav_value
        self.ugv_values[self.step] = ugv_value
        self.dones[self.step] = done
        self.terminated[self.step] = terminated
        self.truncated[self.step] = truncated
        self.uav_timeout_values[self.step] = uav_timeout_value
        self.ugv_timeout_values[self.step] = ugv_timeout_value

        self.step += 1
        self._flat_tensors = None

    def _compute_agent_returns_and_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        timeout_values: np.ndarray,
        last_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            timeout_mask = (self.truncated[t] > 0.5) & (self.terminated[t] <= 0.5)
            bootstrap_value = np.where(timeout_mask, timeout_values[t], next_value)
            delta_non_terminal = 1.0 - self.terminated[t]
            # Vec envs reset immediately after truncation, so bootstrap the delta
            # with terminal_obs values but stop the GAE trace at the episode boundary.
            gae_non_terminal = 1.0 - self.dones[t]
            delta = rewards[t] + self.gamma * bootstrap_value * delta_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * gae_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def compute_returns_and_advantages(
        self,
        last_uav_value: np.ndarray,
        last_ugv_value: np.ndarray,
    ):
        """Compute per-agent GAE advantages and returns."""
        self.uav_advantages, self.uav_returns = self._compute_agent_returns_and_advantages(
            rewards=self.uav_rewards,
            values=self.uav_values,
            timeout_values=self.uav_timeout_values,
            last_value=last_uav_value,
        )
        self.ugv_advantages, self.ugv_returns = self._compute_agent_returns_and_advantages(
            rewards=self.ugv_rewards,
            values=self.ugv_values,
            timeout_values=self.ugv_timeout_values,
            last_value=last_ugv_value,
        )
        self._flat_tensors = None

    def _prepare_flat_tensors(self) -> dict:
        if self._flat_tensors is not None:
            return self._flat_tensors

        flat_np = {
            "uav_obs": self.uav_obs.reshape(-1, self.uav_obs.shape[-1]),
            "ugv_obs": self.ugv_obs.reshape(-1, self.ugv_obs.shape[-1]),
            "uav_actions": self.uav_actions.reshape(-1),
            "ugv_actions": self.ugv_actions.reshape(-1),
            "uav_log_probs": self.uav_log_probs.reshape(-1),
            "ugv_log_probs": self.ugv_log_probs.reshape(-1),
            "uav_action_masks": self.uav_action_masks.reshape(-1, self.uav_action_masks.shape[-1]),
            "ugv_action_masks": self.ugv_action_masks.reshape(-1, self.ugv_action_masks.shape[-1]),
            "uav_advantages": self.uav_advantages.reshape(-1),
            "ugv_advantages": self.ugv_advantages.reshape(-1),
            "uav_returns": self.uav_returns.reshape(-1),
            "ugv_returns": self.ugv_returns.reshape(-1),
            "uav_values": self.uav_values.reshape(-1),
            "ugv_values": self.ugv_values.reshape(-1),
        }
        flat_tensors = {}
        float_keys = {
            "uav_obs",
            "ugv_obs",
            "uav_log_probs",
            "ugv_log_probs",
            "uav_advantages",
            "ugv_advantages",
            "uav_returns",
            "ugv_returns",
            "uav_values",
            "ugv_values",
        }
        for key, arr in flat_np.items():
            tensor = torch.from_numpy(arr)
            if key in ("uav_actions", "ugv_actions"):
                tensor = tensor.to(dtype=torch.long)
            elif key in ("uav_action_masks", "ugv_action_masks"):
                tensor = tensor.to(dtype=torch.bool)
            elif key in float_keys:
                tensor = tensor.to(dtype=torch.float32)
            else:
                tensor = tensor.contiguous()
            flat_tensors[key] = tensor

        self._flat_tensors = flat_tensors
        return flat_tensors

    def release_cached_tensors(self) -> None:
        """Drop cached flattened tensors once an update no longer needs them."""
        self._flat_tensors = None

    def get_batches(
        self,
        num_minibatches: int,
        device: torch.device,
    ) -> Generator[dict, None, None]:
        """
        Yield minibatches for PPO update.
        """
        batch_size = self.total_size
        if num_minibatches <= 0:
            raise ValueError(f"num_minibatches must be positive, got {num_minibatches}")
        effective_minibatches = min(int(num_minibatches), batch_size)
        minibatch_size = batch_size // effective_minibatches
        flat = self._prepare_flat_tensors()
        indices = torch.randperm(batch_size)
        use_non_blocking = device.type == "cuda"

        for mb in range(effective_minibatches):
            start = mb * minibatch_size
            end = batch_size if mb == effective_minibatches - 1 else start + minibatch_size
            mb_indices = indices[start:end]
            cpu_batch = {key: tensor.index_select(0, mb_indices) for key, tensor in flat.items()}
            if device.type == "cpu":
                yield cpu_batch
                continue
            yield {
                key: tensor.to(device=device, non_blocking=use_non_blocking)
                for key, tensor in cpu_batch.items()
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.step = 0
        self.release_cached_tensors()

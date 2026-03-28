"""
Rollout buffer for MAPPO training.

Stores transitions during rollout and computes GAE advantages.
Supports multi-agent data with shared rewards.
"""

import numpy as np
import torch
from typing import Dict, Generator


class RolloutBuffer:
    """
    Buffer for storing rollout data from parallel environments.
    
    Stores separate observations for each agent but shared rewards and values.
    Computes Generalized Advantage Estimation (GAE) for policy optimization.
    
    Args:
        rollout_length: Number of steps per rollout.
        num_envs: Number of parallel environments.
        obs_dims: Dict mapping observation keys to dimensions.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
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

        # Observations (per agent type + critic)
        self.uav_obs = np.zeros(
            (rollout_length, num_envs, obs_dims["uav_obs"]), dtype=np.float32
        )
        self.ugv_obs = np.zeros(
            (rollout_length, num_envs, obs_dims["ugv_obs"]), dtype=np.float32
        )
        self.critic_states = np.zeros(
            (rollout_length, num_envs, obs_dims["critic_state"]), dtype=np.float32
        )

        # Actions
        self.uav_move_actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.uav_bw_actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.ugv_actions = np.zeros((rollout_length, num_envs), dtype=np.int64)

        # Log probabilities
        self.uav_log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.ugv_log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.uav_move_action_masks = np.zeros(
            (rollout_length, num_envs, action_dims["uav_move_action"]), dtype=bool
        )
        self.uav_bw_action_masks = np.zeros(
            (rollout_length, num_envs, action_dims["uav_bw_action"]), dtype=bool
        )
        self.ugv_action_masks = np.zeros(
            (rollout_length, num_envs, action_dims["ugv_action"]), dtype=bool
        )

        # Shared reward and value (team reward)
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.terminated = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.truncated = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.timeout_values = np.zeros((rollout_length, num_envs), dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.step = 0

    def add(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray,
        uav_move_action: np.ndarray,
        uav_bw_action: np.ndarray,
        ugv_action: np.ndarray,
        uav_log_prob: np.ndarray,
        ugv_log_prob: np.ndarray,
        uav_move_action_mask: np.ndarray,
        uav_bw_action_mask: np.ndarray,
        ugv_action_mask: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        timeout_value: np.ndarray,
    ):
        """Add a transition to the buffer."""
        assert self.step < self.rollout_length, "Buffer is full"

        self.uav_obs[self.step] = uav_obs
        self.ugv_obs[self.step] = ugv_obs
        self.critic_states[self.step] = critic_state
        self.uav_move_actions[self.step] = uav_move_action
        self.uav_bw_actions[self.step] = uav_bw_action
        self.ugv_actions[self.step] = ugv_action
        self.uav_log_probs[self.step] = uav_log_prob
        self.ugv_log_probs[self.step] = ugv_log_prob
        self.uav_move_action_masks[self.step] = uav_move_action_mask
        self.uav_bw_action_masks[self.step] = uav_bw_action_mask
        self.ugv_action_masks[self.step] = ugv_action_mask
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.terminated[self.step] = terminated
        self.truncated[self.step] = truncated
        self.timeout_values[self.step] = timeout_value

        self.step += 1

    def compute_returns_and_advantages(self, last_value: np.ndarray):
        """
        Compute GAE advantages and returns.
        
        Args:
            last_value: (num_envs,) value estimate for the state after the last step.
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            timeout_mask = (self.truncated[t] > 0.5) & (self.terminated[t] <= 0.5)
            bootstrap_value = np.where(timeout_mask, self.timeout_values[t], next_value)
            delta_non_terminal = 1.0 - self.terminated[t]
            gae_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * bootstrap_value * delta_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * gae_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        num_minibatches: int,
        device: torch.device,
    ) -> Generator[dict, None, None]:
        """
        Yield minibatches for PPO update.
        
        Flattens (rollout_length, num_envs) into (total,) and splits into minibatches.
        
        Args:
            num_minibatches: Number of minibatches to split data into.
            device: Torch device for tensors.
            
        Yields:
            Dict with all batch data as torch tensors on device.
        """
        batch_size = self.total_size
        if num_minibatches <= 0:
            raise ValueError(f"num_minibatches must be positive, got {num_minibatches}")
        effective_minibatches = min(int(num_minibatches), batch_size)
        minibatch_size = batch_size // effective_minibatches
        indices = np.random.permutation(batch_size)

        # Flatten all data: (T, N, ...) → (T*N, ...)
        flat = {
            "uav_obs": self.uav_obs.reshape(-1, self.uav_obs.shape[-1]),
            "ugv_obs": self.ugv_obs.reshape(-1, self.ugv_obs.shape[-1]),
            "critic_states": self.critic_states.reshape(-1, self.critic_states.shape[-1]),
            "uav_move_actions": self.uav_move_actions.reshape(-1),
            "uav_bw_actions": self.uav_bw_actions.reshape(-1),
            "ugv_actions": self.ugv_actions.reshape(-1),
            "uav_log_probs": self.uav_log_probs.reshape(-1),
            "ugv_log_probs": self.ugv_log_probs.reshape(-1),
            "uav_move_action_masks": self.uav_move_action_masks.reshape(
                -1,
                self.uav_move_action_masks.shape[-1],
            ),
            "uav_bw_action_masks": self.uav_bw_action_masks.reshape(
                -1,
                self.uav_bw_action_masks.shape[-1],
            ),
            "ugv_action_masks": self.ugv_action_masks.reshape(-1, self.ugv_action_masks.shape[-1]),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }

        for mb in range(effective_minibatches):
            start = mb * minibatch_size
            end = batch_size if mb == effective_minibatches - 1 else start + minibatch_size
            mb_indices = indices[start:end]

            batch = {}
            for key, arr in flat.items():
                tensor = torch.from_numpy(arr[mb_indices])
                if key in ("uav_move_actions", "uav_bw_actions", "ugv_actions"):
                    batch[key] = tensor.long().to(device)
                elif key in ("uav_move_action_masks", "uav_bw_action_masks", "ugv_action_masks"):
                    batch[key] = tensor.bool().to(device)
                else:
                    batch[key] = tensor.float().to(device)

            yield batch

    def reset(self):
        """Reset buffer for next rollout."""
        self.step = 0

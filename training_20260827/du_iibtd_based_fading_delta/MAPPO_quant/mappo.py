"""
Multi-Agent PPO (MAPPO) Algorithm.

Implements Centralized Training with Decentralized Execution (CTDE):
- Actor networks use local observations (decentralized execution)
- Critic network uses global state (centralized training)
- Shared reward for cooperative agents
"""

import torch
from typing import Dict

from du_iibtd_based_fading_delta.MAPPO_quant.networks import MAPPOPolicy
from du_iibtd_based_fading_delta.shared.joint_buffer import RolloutBuffer
from du_iibtd_based_fading_delta.shared.ppo_base import PPOTrainerBase


class MAPPO(PPOTrainerBase):
    """
    MAPPO trainer for two cooperative agents (UAV + UGV).
    
    Training loop:
    1. Collect rollout using current policy
    2. Compute GAE advantages
    3. Update actors and critic with PPO clipped objective
    
    Args:
        policy: MAPPOPolicy containing all networks and optimizers.
        config: Shared on-policy hyperparameters.
    """

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        
        Args:
            buffer: Filled rollout buffer with computed advantages.
            
        Returns:
            Dict of training metrics.
        """
        self.policy.prepare_for_training()
        self._normalize_advantages(buffer)

        # Tracking metrics
        total_uav_policy_loss = 0
        total_ugv_policy_loss = 0
        total_value_loss = 0
        total_uav_entropy = 0
        total_ugv_entropy = 0
        total_uav_clip_frac = 0
        total_ugv_clip_frac = 0
        num_updates = 0

        for _ in range(self.config.num_epochs):
            for batch in buffer.get_batches(self.config.num_minibatches, self.device):
                # 计算新动作的概率和熵
                uav_new_logp, uav_entropy = self.policy.uav_actor.evaluate_action(
                    batch["uav_obs"], batch["uav_actions"], batch["uav_action_masks"]
                )
                ugv_new_logp, ugv_entropy = self.policy.ugv_actor.evaluate_action(
                    batch["ugv_obs"], batch["ugv_actions"], batch["ugv_action_masks"]
                )

                new_values = self.policy.critic(batch["critic_states"]).squeeze(-1)
                # 计算 agent动作的loss
                uav_policy_loss, uav_clip_frac = self._compute_policy_loss(
                    new_log_prob=uav_new_logp,
                    old_log_prob=batch["uav_log_probs"],
                    advantages=batch["advantages"],
                )
                ugv_policy_loss, ugv_clip_frac = self._compute_policy_loss(
                    new_log_prob=ugv_new_logp,
                    old_log_prob=batch["ugv_log_probs"],
                    advantages=batch["advantages"],
                )
                uav_total_loss = uav_policy_loss - self.config.entropy_coef * uav_entropy.mean()
                ugv_total_loss = ugv_policy_loss - self.config.entropy_coef * ugv_entropy.mean()

                value_targets = batch["returns"]
                value_baseline = batch["values"]

                value_pred_clipped = value_baseline + torch.clamp(
                    new_values - value_baseline,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                value_loss_unclipped = (new_values - value_targets) ** 2
                value_loss_clipped = (value_pred_clipped - value_targets) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                critic_loss = self.config.value_loss_coef * value_loss

                self._apply_optimizer_step(
                    optimizer=self.policy.uav_actor_optimizer,
                    network=self.policy.uav_actor,
                    loss=uav_total_loss,
                )
                self._apply_optimizer_step(
                    optimizer=self.policy.ugv_actor_optimizer,
                    network=self.policy.ugv_actor,
                    loss=ugv_total_loss,
                )
                self._apply_optimizer_step(
                    optimizer=self.policy.critic_optimizer,
                    network=self.policy.critic,
                    loss=critic_loss,
                )

                # Accumulate metrics
                total_uav_policy_loss += uav_policy_loss.item()
                total_ugv_policy_loss += ugv_policy_loss.item()
                total_value_loss += value_loss.item()
                total_uav_entropy += uav_entropy.mean().item()
                total_ugv_entropy += ugv_entropy.mean().item()
                total_uav_clip_frac += uav_clip_frac
                total_ugv_clip_frac += ugv_clip_frac
                num_updates += 1

        if num_updates <= 0:
            raise ValueError(
                "MAPPO.update() produced zero minibatch updates. "
                "Check that num_epochs and rollout settings are positive."
            )

        # Average metrics
        metrics = {
            "uav_policy_loss": total_uav_policy_loss / num_updates,
            "ugv_policy_loss": total_ugv_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "uav_entropy": total_uav_entropy / num_updates,
            "ugv_entropy": total_ugv_entropy / num_updates,
            "uav_clip_fraction": total_uav_clip_frac / num_updates,
            "ugv_clip_fraction": total_ugv_clip_frac / num_updates,
        }

        return metrics

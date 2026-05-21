"""
Multi-Agent PPO (MAPPO) Algorithm.

Implements Centralized Training with Decentralized Execution (CTDE):
- Actor networks use local observations (decentralized execution)
- Critic network uses global state (centralized training)
- Shared reward for cooperative agents
"""

import torch
import torch.nn as nn
from typing import Dict

from config import MAPPOConfig
from networks import MAPPOPolicy
from buffer import RolloutBuffer


class MAPPO:
    """
    MAPPO trainer for two cooperative agents (UAV + UGV).
    
    Training loop:
    1. Collect rollout using current policy
    2. Compute GAE advantages
    3. Update actors and critic with PPO clipped objective
    
    Args:
        policy: MAPPOPolicy containing all networks and optimizers.
        config: MAPPOConfig with hyperparameters.
    """

    def __init__(self, policy: MAPPOPolicy, config: MAPPOConfig):
        self.policy = policy
        self.config = config
        self.device = policy.device

    def _normalize_advantages(self, buffer: RolloutBuffer) -> None:
        flat_adv = buffer.advantages.reshape(-1)
        mean = flat_adv.mean()
        std = flat_adv.std()
        normalized = (flat_adv - mean) / (std + 1e-8)
        buffer.advantages = normalized.reshape(buffer.advantages.shape)

    def _compute_policy_loss(
        self,
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        # 计算概率比率
        ratio = torch.exp(new_log_prob - old_log_prob)
        # 限制前的advantage
        unclipped = ratio * advantages
        # 概率比限制区间
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        # 限制后的advantage
        clipped = clipped_ratio * advantages
        # 计算policy loss
        policy_loss = -torch.min(unclipped, clipped).mean()
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
        return policy_loss, float(clip_fraction)

    def _apply_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        network: torch.nn.Module,
        loss: torch.Tensor,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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

        self.policy.mark_rollout_stale()
        return metrics

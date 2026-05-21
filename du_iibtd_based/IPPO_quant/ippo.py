"""
Independent PPO (IPPO) trainer for UAV-UGV cooperation.

Each agent has its own actor and local critic. No centralized critic or global
critic state is used during training.
"""

from typing import Dict

import torch
import torch.nn as nn

from buffer import RolloutBuffer
from config import IPPOConfig
from networks import IPPOPolicy


class IPPO:
    """
    IPPO trainer for two cooperative agents (UAV + UGV).

    Policy/value updates are computed independently for each agent from local
    observations and per-agent reward streams.
    """

    def __init__(self, policy: IPPOPolicy, config: IPPOConfig):
        self.policy = policy
        self.config = config
        self.device = policy.device

    def _normalize_agent_advantages(self, advantages) -> None:
        flat_adv = advantages.reshape(-1)
        mean = flat_adv.mean()
        std = flat_adv.std()
        flat_adv[...] = (flat_adv - mean) / (std + 1e-8)

    def _normalize_advantages(self, buffer: RolloutBuffer) -> None:
        self._normalize_agent_advantages(buffer.uav_advantages)
        self._normalize_agent_advantages(buffer.ugv_advantages)

    def _compute_policy_loss(
        self,
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * advantages
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        clipped = clipped_ratio * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
        return policy_loss, float(clip_fraction)

    def _compute_value_loss(
        self,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.config.clip_epsilon,
            self.config.clip_epsilon,
        )
        value_loss_unclipped = (new_values - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        return 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

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
        """Perform independent PPO updates using collected rollout data."""
        self.policy.prepare_for_training()
        self._normalize_advantages(buffer)

        total_uav_policy_loss = 0.0
        total_ugv_policy_loss = 0.0
        total_uav_value_loss = 0.0
        total_ugv_value_loss = 0.0
        total_uav_entropy = 0.0
        total_ugv_entropy = 0.0
        total_uav_clip_frac = 0.0
        total_ugv_clip_frac = 0.0
        num_updates = 0

        for _ in range(self.config.num_epochs):
            for batch in buffer.get_batches(self.config.num_minibatches, self.device):
                uav_new_logp, uav_entropy = self.policy.uav_actor.evaluate_action(
                    batch["uav_obs"], batch["uav_actions"], batch["uav_action_masks"]
                )
                ugv_new_logp, ugv_entropy = self.policy.ugv_actor.evaluate_action(
                    batch["ugv_obs"], batch["ugv_actions"], batch["ugv_action_masks"]
                )

                uav_new_values = self.policy.uav_critic(batch["uav_obs"]).squeeze(-1)
                ugv_new_values = self.policy.ugv_critic(batch["ugv_obs"]).squeeze(-1)

                uav_policy_loss, uav_clip_frac = self._compute_policy_loss(
                    new_log_prob=uav_new_logp,
                    old_log_prob=batch["uav_log_probs"],
                    advantages=batch["uav_advantages"],
                )
                ugv_policy_loss, ugv_clip_frac = self._compute_policy_loss(
                    new_log_prob=ugv_new_logp,
                    old_log_prob=batch["ugv_log_probs"],
                    advantages=batch["ugv_advantages"],
                )
                uav_total_loss = uav_policy_loss - self.config.entropy_coef * uav_entropy.mean()
                ugv_total_loss = ugv_policy_loss - self.config.entropy_coef * ugv_entropy.mean()

                uav_value_loss = self._compute_value_loss(
                    new_values=uav_new_values,
                    old_values=batch["uav_values"],
                    returns=batch["uav_returns"],
                )
                ugv_value_loss = self._compute_value_loss(
                    new_values=ugv_new_values,
                    old_values=batch["ugv_values"],
                    returns=batch["ugv_returns"],
                )

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
                    optimizer=self.policy.uav_critic_optimizer,
                    network=self.policy.uav_critic,
                    loss=self.config.value_loss_coef * uav_value_loss,
                )
                self._apply_optimizer_step(
                    optimizer=self.policy.ugv_critic_optimizer,
                    network=self.policy.ugv_critic,
                    loss=self.config.value_loss_coef * ugv_value_loss,
                )

                total_uav_policy_loss += uav_policy_loss.item()
                total_ugv_policy_loss += ugv_policy_loss.item()
                total_uav_value_loss += uav_value_loss.item()
                total_ugv_value_loss += ugv_value_loss.item()
                total_uav_entropy += uav_entropy.mean().item()
                total_ugv_entropy += ugv_entropy.mean().item()
                total_uav_clip_frac += uav_clip_frac
                total_ugv_clip_frac += ugv_clip_frac
                num_updates += 1

        if num_updates <= 0:
            raise ValueError(
                "IPPO.update() produced zero minibatch updates. "
                "Check that num_epochs and rollout settings are positive."
            )

        uav_value_loss = total_uav_value_loss / num_updates
        ugv_value_loss = total_ugv_value_loss / num_updates
        metrics = {
            "uav_policy_loss": total_uav_policy_loss / num_updates,
            "ugv_policy_loss": total_ugv_policy_loss / num_updates,
            "uav_value_loss": uav_value_loss,
            "ugv_value_loss": ugv_value_loss,
            "value_loss": 0.5 * (uav_value_loss + ugv_value_loss),
            "uav_entropy": total_uav_entropy / num_updates,
            "ugv_entropy": total_ugv_entropy / num_updates,
            "uav_clip_fraction": total_uav_clip_frac / num_updates,
            "ugv_clip_fraction": total_ugv_clip_frac / num_updates,
        }

        self.policy.mark_rollout_stale()
        return metrics

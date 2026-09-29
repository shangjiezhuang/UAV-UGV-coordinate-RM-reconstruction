"""Algorithm-neutral PPO optimization primitives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PPOTrainerBase:
    """Common clipped-loss and optimizer helpers for PPO-family algorithms."""

    def __init__(self, policy: Any, config: Any):
        self.policy = policy
        self.config = config
        self.device = policy.device

    def _normalize_advantages(self, buffer: Any) -> None:
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
            clip_fraction = (
                (ratio - 1.0).abs() > self.config.clip_epsilon
            ).float().mean().item()
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


__all__ = ["PPOTrainerBase"]

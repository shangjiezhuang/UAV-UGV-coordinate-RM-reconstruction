"""Standard two-agent HAPPO update with a shared joint GAE estimator.

For a sampled agent order, each actor is optimized to completion before the
next actor.  The joint advantage used by later actors is multiplied by the
new/old likelihood ratio of every actor already updated in that sequence.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from du_iibtd_based_fading_delta.on_policy_update import (
    clipped_value_loss,
    indexed_batch,
    minibatch_indices,
)
from du_iibtd_based_fading_delta.shared.ppo_base import PPOTrainerBase


_AGENTS = ("uav", "ugv")


def update_compound_factor(
    compound_factor: torch.Tensor,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Apply one already-updated agent's exact importance ratio."""
    updated = compound_factor * torch.exp(new_log_prob - old_log_prob)
    if not bool(torch.isfinite(updated).all()):
        raise FloatingPointError("HAPPO compound importance factor became non-finite")
    return updated


class HAPPO(PPOTrainerBase):
    """Sequential HAPPO optimizer for one UAV actor and one UGV actor."""

    def _sample_agent_order(self) -> Tuple[str, str]:
        order = torch.randperm(len(_AGENTS)).tolist()
        return _AGENTS[int(order[0])], _AGENTS[int(order[1])]

    def _batch(self, flat: dict, indices: torch.Tensor) -> dict:
        return indexed_batch(flat, indices, self.device)

    def _evaluate_agent_log_probs(self, agent: str, flat: dict) -> torch.Tensor:
        values = []
        actor = getattr(self.policy, f"{agent}_actor")
        with torch.no_grad():
            for indices in minibatch_indices(
                int(flat["advantages"].numel()),
                int(self.config.num_minibatches),
                shuffle=False,
            ):
                batch = self._batch(flat, indices)
                new_log_prob, _ = actor.evaluate_action(
                    batch[f"{agent}_obs"],
                    batch[f"{agent}_actions"],
                    batch[f"{agent}_action_masks"],
                )
                values.append((indices, new_log_prob.detach().cpu()))
        ordered = torch.empty_like(flat[f"{agent}_log_probs"])
        for indices, log_prob in values:
            ordered.index_copy_(0, indices, log_prob)
        return ordered

    def _update_actor(
        self,
        agent: str,
        flat: dict,
        compound_factor: torch.Tensor,
    ) -> Dict[str, float]:
        actor = getattr(self.policy, f"{agent}_actor")
        optimizer = getattr(self.policy, f"{agent}_actor_optimizer")
        totals = {"policy_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
        count = 0
        for _ in range(int(self.config.num_epochs)):
            for indices in minibatch_indices(
                int(flat["advantages"].numel()),
                int(self.config.num_minibatches),
            ):
                batch = self._batch(flat, indices)
                factor = compound_factor.index_select(0, indices).to(
                    device=self.device,
                    non_blocking=self.device.type == "cuda",
                )
                modified_advantage = factor * batch["advantages"]
                new_log_prob, entropy = actor.evaluate_action(
                    batch[f"{agent}_obs"],
                    batch[f"{agent}_actions"],
                    batch[f"{agent}_action_masks"],
                )
                policy_loss, clip_fraction = self._compute_policy_loss(
                    new_log_prob=new_log_prob,
                    old_log_prob=batch[f"{agent}_log_probs"],
                    advantages=modified_advantage,
                )
                total_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                self._apply_optimizer_step(optimizer, actor, total_loss)
                totals["policy_loss"] += float(policy_loss.item())
                totals["entropy"] += float(entropy.mean().item())
                totals["clip_fraction"] += float(clip_fraction)
                count += 1
        if count <= 0:
            raise ValueError(f"HAPPO produced zero {agent} actor updates")
        return {key: value / count for key, value in totals.items()}

    def _update_critic(self, flat: dict) -> float:
        total_value_loss = 0.0
        count = 0
        for _ in range(int(self.config.num_epochs)):
            for indices in minibatch_indices(
                int(flat["advantages"].numel()),
                int(self.config.num_minibatches),
            ):
                batch = self._batch(flat, indices)
                new_values = self.policy.critic(batch["critic_states"]).squeeze(-1)
                baseline = batch["values"]
                value_loss = clipped_value_loss(
                    new_values,
                    baseline,
                    batch["returns"],
                    self.config.clip_epsilon,
                )
                self._apply_optimizer_step(
                    self.policy.critic_optimizer,
                    self.policy.critic,
                    self.config.value_loss_coef * value_loss,
                )
                total_value_loss += float(value_loss.item())
                count += 1
        if count <= 0:
            raise ValueError("HAPPO produced zero critic updates")
        return total_value_loss / count

    def update(self, buffer) -> Dict[str, float]:
        self.policy.prepare_for_training()
        self._normalize_advantages(buffer)
        buffer.release_cached_tensors()
        flat = buffer._prepare_flat_tensors()
        compound_factor = torch.ones_like(flat["advantages"])
        order = self._sample_agent_order()
        actor_metrics: Dict[str, Dict[str, float]] = {}

        for position, agent in enumerate(order):
            actor_metrics[agent] = self._update_actor(agent, flat, compound_factor)
            if position < len(order) - 1:
                new_log_prob = self._evaluate_agent_log_probs(agent, flat)
                compound_factor = update_compound_factor(
                    compound_factor,
                    new_log_prob,
                    flat[f"{agent}_log_probs"],
                )

        value_loss = self._update_critic(flat)
        return {
            "uav_policy_loss": actor_metrics["uav"]["policy_loss"],
            "ugv_policy_loss": actor_metrics["ugv"]["policy_loss"],
            "value_loss": value_loss,
            "uav_entropy": actor_metrics["uav"]["entropy"],
            "ugv_entropy": actor_metrics["ugv"]["entropy"],
            "uav_clip_fraction": actor_metrics["uav"]["clip_fraction"],
            "ugv_clip_fraction": actor_metrics["ugv"]["clip_fraction"],
            "happo_uav_first": float(order[0] == "uav"),
            "happo_compound_factor_mean": float(compound_factor.mean().item()),
        }


__all__ = ["HAPPO", "update_compound_factor"]

"""MAPPO with COMA-style counterfactual credit assignment.

The common value critic still supplies temporal GAE/return targets.  A separate
centralized joint-action Q critic supplies agent credit: for each actor, the
other actor's sampled action is fixed while the current actor's legal actions
are marginalized under its old policy.
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


def counterfactual_advantage(
    q_taken: torch.Tensor,
    all_action_q: torch.Tensor,
    action_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Compute ``Q(taken) - E_pi[Q(counterfactual action)]``."""
    if all_action_q.shape != action_probabilities.shape:
        raise ValueError(
            "all_action_q and action_probabilities must have identical shapes, "
            f"got {all_action_q.shape} and {action_probabilities.shape}"
        )
    if q_taken.shape != all_action_q.shape[:-1]:
        raise ValueError(
            f"q_taken shape {q_taken.shape} does not match {all_action_q.shape[:-1]}"
        )
    baseline = (all_action_q * action_probabilities).sum(dim=-1)
    return q_taken - baseline


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


class MAPPOCF(PPOTrainerBase):
    """PPO actor updates driven by separate UAV and UGV counterfactual advantages."""

    def _batch(self, flat: dict, indices: torch.Tensor) -> dict:
        return indexed_batch(flat, indices, self.device)

    def _train_critics(self, flat: dict) -> Tuple[float, float]:
        value_total = 0.0
        q_total = 0.0
        count = 0
        for _ in range(int(self.config.num_epochs)):
            for indices in minibatch_indices(
                int(flat["returns"].numel()),
                int(self.config.num_minibatches),
            ):
                batch = self._batch(flat, indices)
                returns = batch["returns"]
                values = self.policy.critic(batch["critic_states"]).squeeze(-1)
                baseline = batch["values"]
                value_loss = clipped_value_loss(
                    values,
                    baseline,
                    returns,
                    self.config.clip_epsilon,
                )
                q_values = self.policy.q_critic(
                    batch["critic_states"],
                    batch["uav_actions"],
                    batch["ugv_actions"],
                )
                q_loss = 0.5 * ((q_values - returns) ** 2).mean()
                self._apply_optimizer_step(
                    self.policy.critic_optimizer,
                    self.policy.critic,
                    self.config.value_loss_coef * value_loss,
                )
                self._apply_optimizer_step(
                    self.policy.q_critic_optimizer,
                    self.policy.q_critic,
                    self.config.value_loss_coef * q_loss,
                )
                value_total += float(value_loss.item())
                q_total += float(q_loss.item())
                count += 1
        if count <= 0:
            raise ValueError("MAPPO-CF produced zero critic updates")
        return value_total / count, q_total / count

    def _all_action_q(self, batch: dict, agent: str) -> torch.Tensor:
        batch_size = int(batch["critic_states"].shape[0])
        action_dim = int(getattr(self.policy, f"{agent}_action_dim"))
        alternatives = torch.arange(action_dim, device=self.device).view(1, -1)
        alternatives = alternatives.expand(batch_size, -1)
        states = batch["critic_states"].unsqueeze(1).expand(-1, action_dim, -1)
        if agent == "uav":
            uav_actions = alternatives
            ugv_actions = batch["ugv_actions"].unsqueeze(1).expand(-1, action_dim)
        else:
            uav_actions = batch["uav_actions"].unsqueeze(1).expand(-1, action_dim)
            ugv_actions = alternatives
        return self.policy.q_critic(
            states.reshape(batch_size * action_dim, -1),
            uav_actions.reshape(-1),
            ugv_actions.reshape(-1),
        ).reshape(batch_size, action_dim)

    def _compute_counterfactual_advantages(self, flat: dict) -> Tuple[dict, dict]:
        total_size = int(flat["returns"].numel())
        uav_values = torch.empty(total_size, dtype=torch.float32)
        ugv_values = torch.empty(total_size, dtype=torch.float32)
        with torch.no_grad():
            for indices in minibatch_indices(
                total_size,
                int(self.config.num_minibatches),
                shuffle=False,
            ):
                batch = self._batch(flat, indices)
                q_taken = self.policy.q_critic(
                    batch["critic_states"],
                    batch["uav_actions"],
                    batch["ugv_actions"],
                )
                uav_probabilities = self.policy.uav_actor(
                    batch["uav_obs"], batch["uav_action_masks"]
                ).probs
                ugv_probabilities = self.policy.ugv_actor(
                    batch["ugv_obs"], batch["ugv_action_masks"]
                ).probs
                uav_advantage = counterfactual_advantage(
                    q_taken,
                    self._all_action_q(batch, "uav"),
                    uav_probabilities,
                )
                ugv_advantage = counterfactual_advantage(
                    q_taken,
                    self._all_action_q(batch, "ugv"),
                    ugv_probabilities,
                )
                uav_values.index_copy_(0, indices, uav_advantage.cpu())
                ugv_values.index_copy_(0, indices, ugv_advantage.cpu())
        diagnostics = {
            "cf_uav_advantage_mean": float(uav_values.mean().item()),
            "cf_uav_advantage_std": float(uav_values.std(unbiased=False).item()),
            "cf_ugv_advantage_mean": float(ugv_values.mean().item()),
            "cf_ugv_advantage_std": float(ugv_values.std(unbiased=False).item()),
        }
        return {
            "uav": _normalize(uav_values),
            "ugv": _normalize(ugv_values),
        }, diagnostics

    def _train_actors(self, flat: dict, advantages: dict) -> Dict[str, float]:
        totals = {
            "uav_policy_loss": 0.0,
            "ugv_policy_loss": 0.0,
            "uav_entropy": 0.0,
            "ugv_entropy": 0.0,
            "uav_clip_fraction": 0.0,
            "ugv_clip_fraction": 0.0,
        }
        count = 0
        for _ in range(int(self.config.num_epochs)):
            for indices in minibatch_indices(
                int(flat["returns"].numel()),
                int(self.config.num_minibatches),
            ):
                batch = self._batch(flat, indices)
                agent_losses = {}
                for agent in ("uav", "ugv"):
                    actor = getattr(self.policy, f"{agent}_actor")
                    new_log_prob, entropy = actor.evaluate_action(
                        batch[f"{agent}_obs"],
                        batch[f"{agent}_actions"],
                        batch[f"{agent}_action_masks"],
                    )
                    agent_advantage = advantages[agent].index_select(0, indices).to(
                        device=self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    policy_loss, clip_fraction = self._compute_policy_loss(
                        new_log_prob,
                        batch[f"{agent}_log_probs"],
                        agent_advantage,
                    )
                    agent_losses[agent] = (
                        policy_loss - self.config.entropy_coef * entropy.mean()
                    )
                    totals[f"{agent}_policy_loss"] += float(policy_loss.item())
                    totals[f"{agent}_entropy"] += float(entropy.mean().item())
                    totals[f"{agent}_clip_fraction"] += float(clip_fraction)
                self._apply_optimizer_step(
                    self.policy.uav_actor_optimizer,
                    self.policy.uav_actor,
                    agent_losses["uav"],
                )
                self._apply_optimizer_step(
                    self.policy.ugv_actor_optimizer,
                    self.policy.ugv_actor,
                    agent_losses["ugv"],
                )
                count += 1
        if count <= 0:
            raise ValueError("MAPPO-CF produced zero actor updates")
        return {key: value / count for key, value in totals.items()}

    def update(self, buffer) -> Dict[str, float]:
        self.policy.prepare_for_training()
        buffer.release_cached_tensors()
        flat = buffer._prepare_flat_tensors()
        value_loss, q_loss = self._train_critics(flat)
        advantages, diagnostics = self._compute_counterfactual_advantages(flat)
        metrics = self._train_actors(flat, advantages)
        metrics["value_loss"] = value_loss
        metrics["q_value_loss"] = q_loss
        metrics.update(diagnostics)
        return metrics


__all__ = ["MAPPOCF", "counterfactual_advantage"]

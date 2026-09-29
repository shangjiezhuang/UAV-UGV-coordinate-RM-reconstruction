"""Single-UAV PPO used with environment-owned UGV baseline controllers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from du_iibtd_based_fading_delta.shared.on_policy_networks import ActorNetwork, CriticNetwork
from du_iibtd_based_fading_delta.shared.ppo_base import PPOTrainerBase


class UAVPPOPolicy:
    """One UAV actor and one critic with the common rollout-policy API."""

    def __init__(self, obs_dims: dict, action_dims: dict, config):
        self.config = config
        requested_device = torch.device(config.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"PPO device {config.device!r} was explicitly requested, but "
                "CUDA is unavailable. Refusing to fall back to CPU."
            )
        self.device = requested_device
        self.rollout_device = self.device
        self.uav_actor = ActorNetwork(
            obs_dim=obs_dims["uav_obs"],
            action_dim=action_dims["uav_action"],
            hidden_dims=config.actor_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)
        self.critic = CriticNetwork(
            state_dim=obs_dims["critic_state"],
            hidden_dims=config.critic_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)
        self.uav_actor_optimizer = torch.optim.Adam(
            self.uav_actor.parameters(), lr=config.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic, eps=1e-5
        )
        self.rollout_uav_actor = self.uav_actor
        self.rollout_critic = self.critic
        self.prepare_for_training()

    def prepare_for_training(self) -> None:
        self.uav_actor.train()
        self.critic.train()

    def _enter_rollout_mode(self) -> tuple[bool, bool]:
        previous_modes = (
            bool(self.rollout_uav_actor.training),
            bool(self.rollout_critic.training),
        )
        self.rollout_uav_actor.eval()
        self.rollout_critic.eval()
        return previous_modes

    def _exit_rollout_mode(self, previous_modes: tuple[bool, bool]) -> None:
        self.rollout_uav_actor.train(previous_modes[0])
        self.rollout_critic.train(previous_modes[1])

    @torch.no_grad()
    def get_actions(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        del ugv_obs, ugv_action_mask
        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs, dtype=torch.float32, device=self.rollout_device
            )
            state_t = torch.as_tensor(
                critic_state, dtype=torch.float32, device=self.rollout_device
            )
            uav_mask_t = (
                None
                if uav_action_mask is None
                else torch.as_tensor(
                    uav_action_mask, dtype=torch.bool, device=self.rollout_device
                )
            )
            uav_action, uav_logp, uav_entropy = self.rollout_uav_actor.get_action(
                uav_obs_t,
                uav_mask_t,
                deterministic,
            )
            value = self.rollout_critic(state_t).squeeze(-1)
            dummy = torch.zeros_like(uav_action)
            dummy_float = torch.zeros_like(uav_logp)
            return {
                "uav_action": uav_action.cpu().numpy(),
                "ugv_action": dummy.cpu().numpy(),
                "uav_log_prob": uav_logp.cpu().numpy(),
                "ugv_log_prob": dummy_float.cpu().numpy(),
                "value": value.cpu().numpy(),
                "uav_entropy": uav_entropy.cpu().numpy(),
                "ugv_entropy": dummy_float.cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    @torch.no_grad()
    def get_single_action(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        action_data = self.get_actions(
            uav_obs=np.asarray(uav_obs)[np.newaxis, ...],
            ugv_obs=np.asarray(ugv_obs)[np.newaxis, ...],
            critic_state=np.asarray(critic_state)[np.newaxis, ...],
            uav_action_mask=(
                None
                if uav_action_mask is None
                else np.asarray(uav_action_mask)[np.newaxis, ...]
            ),
            ugv_action_mask=(
                None
                if ugv_action_mask is None
                else np.asarray(ugv_action_mask)[np.newaxis, ...]
            ),
            deterministic=deterministic,
        )
        return {
            key: int(value[0]) if key.endswith("action") else float(value[0])
            for key, value in action_data.items()
        }

    def get_value(self, critic_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            previous_modes = self._enter_rollout_mode()
            try:
                state_t = torch.as_tensor(
                    critic_state, dtype=torch.float32, device=self.rollout_device
                )
                return self.rollout_critic(state_t).squeeze(-1).cpu().numpy()
            finally:
                self._exit_rollout_mode(previous_modes)

    def save(self, path: str) -> None:
        torch.save(
            {
                "uav_actor": self.uav_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.uav_actor.load_state_dict(checkpoint["uav_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "uav_actor_optimizer" in checkpoint:
            self.uav_actor_optimizer.load_state_dict(checkpoint["uav_actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.prepare_for_training()


class UAVPPO(PPOTrainerBase):
    """PPO update for only the UAV actor and its critic."""

    policy: UAVPPOPolicy

    def update(self, buffer) -> Dict[str, float]:
        self.policy.prepare_for_training()
        self._normalize_advantages(buffer)
        totals = {
            "uav_policy_loss": 0.0,
            "value_loss": 0.0,
            "uav_entropy": 0.0,
            "uav_clip_fraction": 0.0,
        }
        num_updates = 0
        for _ in range(self.config.num_epochs):
            for batch in buffer.get_batches(self.config.num_minibatches, self.device):
                new_logp, entropy = self.policy.uav_actor.evaluate_action(
                    batch["uav_obs"],
                    batch["uav_actions"],
                    batch["uav_action_masks"],
                )
                new_values = self.policy.critic(batch["critic_states"]).squeeze(-1)
                policy_loss, clip_fraction = self._compute_policy_loss(
                    new_log_prob=new_logp,
                    old_log_prob=batch["uav_log_probs"],
                    advantages=batch["advantages"],
                )
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                value_baseline = batch["values"]
                value_pred_clipped = value_baseline + torch.clamp(
                    new_values - value_baseline,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                value_loss = 0.5 * torch.max(
                    (new_values - batch["returns"]) ** 2,
                    (value_pred_clipped - batch["returns"]) ** 2,
                ).mean()
                self._apply_optimizer_step(
                    self.policy.uav_actor_optimizer,
                    self.policy.uav_actor,
                    actor_loss,
                )
                self._apply_optimizer_step(
                    self.policy.critic_optimizer,
                    self.policy.critic,
                    self.config.value_loss_coef * value_loss,
                )
                totals["uav_policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["uav_entropy"] += float(entropy.mean().item())
                totals["uav_clip_fraction"] += float(clip_fraction)
                num_updates += 1
        if num_updates <= 0:
            raise ValueError("UAVPPO.update() produced zero minibatch updates")
        return {
            "uav_policy_loss": totals["uav_policy_loss"] / num_updates,
            "ugv_policy_loss": 0.0,
            "value_loss": totals["value_loss"] / num_updates,
            "uav_entropy": totals["uav_entropy"] / num_updates,
            "ugv_entropy": 0.0,
            "uav_clip_fraction": totals["uav_clip_fraction"] / num_updates,
            "ugv_clip_fraction": 0.0,
        }


__all__ = ["UAVPPO", "UAVPPOPolicy"]

"""IPPO policy: shared actor architecture with two local critics."""

from __future__ import annotations

import numpy as np
import torch

from du_iibtd_based_fading_delta.shared.on_policy_networks import (
    ActorNetwork,
    CriticNetwork,
    init_weights,
)


class IPPOPolicy:
    """Two actors and two local-observation critics on one explicit device."""

    def __init__(self, obs_dims: dict, action_dims: dict, config):
        self.config = config
        requested_device = torch.device(config.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"IPPO device {config.device!r} was explicitly requested, but "
                "CUDA is unavailable. Refusing to fall back to CPU."
            )
        self.device = requested_device
        self.rollout_device = self.device

        actor_kwargs = {
            "hidden_dims": config.actor_hidden_dims,
            "use_feature_norm": config.use_feature_norm,
            "use_orthogonal_init": config.use_orthogonal_init,
        }
        critic_kwargs = {
            "hidden_dims": config.critic_hidden_dims,
            "use_feature_norm": config.use_feature_norm,
            "use_orthogonal_init": config.use_orthogonal_init,
        }
        self.uav_actor = ActorNetwork(
            obs_dim=obs_dims["uav_obs"],
            action_dim=action_dims["uav_action"],
            **actor_kwargs,
        ).to(self.device)
        self.ugv_actor = ActorNetwork(
            obs_dim=obs_dims["ugv_obs"],
            action_dim=action_dims["ugv_action"],
            **actor_kwargs,
        ).to(self.device)
        self.uav_critic = CriticNetwork(
            state_dim=obs_dims["uav_obs"],
            **critic_kwargs,
        ).to(self.device)
        self.ugv_critic = CriticNetwork(
            state_dim=obs_dims["ugv_obs"],
            **critic_kwargs,
        ).to(self.device)

        self.uav_actor_optimizer = torch.optim.Adam(
            self.uav_actor.parameters(), lr=config.lr_actor, eps=1e-5
        )
        self.ugv_actor_optimizer = torch.optim.Adam(
            self.ugv_actor.parameters(), lr=config.lr_actor, eps=1e-5
        )
        self.uav_critic_optimizer = torch.optim.Adam(
            self.uav_critic.parameters(), lr=config.lr_critic, eps=1e-5
        )
        self.ugv_critic_optimizer = torch.optim.Adam(
            self.ugv_critic.parameters(), lr=config.lr_critic, eps=1e-5
        )
        self.rollout_uav_actor = self.uav_actor
        self.rollout_ugv_actor = self.ugv_actor
        self.rollout_uav_critic = self.uav_critic
        self.rollout_ugv_critic = self.ugv_critic
        self.prepare_for_training()

    def prepare_for_training(self) -> None:
        self.uav_actor.train()
        self.ugv_actor.train()
        self.uav_critic.train()
        self.ugv_critic.train()

    def _enter_rollout_mode(self) -> tuple[bool, bool, bool, bool]:
        previous_modes = (
            bool(self.uav_actor.training),
            bool(self.ugv_actor.training),
            bool(self.uav_critic.training),
            bool(self.ugv_critic.training),
        )
        self.uav_actor.eval()
        self.ugv_actor.eval()
        self.uav_critic.eval()
        self.ugv_critic.eval()
        return previous_modes

    def _exit_rollout_mode(self, previous_modes: tuple[bool, bool, bool, bool]) -> None:
        self.uav_actor.train(previous_modes[0])
        self.ugv_actor.train(previous_modes[1])
        self.uav_critic.train(previous_modes[2])
        self.ugv_critic.train(previous_modes[3])

    @torch.no_grad()
    def get_actions(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray = None,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        del critic_state
        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs, dtype=torch.float32, device=self.rollout_device
            )
            ugv_obs_t = torch.as_tensor(
                ugv_obs, dtype=torch.float32, device=self.rollout_device
            )
            uav_mask_t = (
                None
                if uav_action_mask is None
                else torch.as_tensor(
                    uav_action_mask, dtype=torch.bool, device=self.rollout_device
                )
            )
            ugv_mask_t = (
                None
                if ugv_action_mask is None
                else torch.as_tensor(
                    ugv_action_mask, dtype=torch.bool, device=self.rollout_device
                )
            )
            uav_action, uav_logp, uav_entropy = self.uav_actor.get_action(
                uav_obs_t, uav_mask_t, deterministic
            )
            ugv_action, ugv_logp, ugv_entropy = self.ugv_actor.get_action(
                ugv_obs_t, ugv_mask_t, deterministic
            )
            uav_value = self.uav_critic(uav_obs_t).squeeze(-1)
            ugv_value = self.ugv_critic(ugv_obs_t).squeeze(-1)
            return {
                "uav_action": uav_action.cpu().numpy(),
                "ugv_action": ugv_action.cpu().numpy(),
                "uav_log_prob": uav_logp.cpu().numpy(),
                "ugv_log_prob": ugv_logp.cpu().numpy(),
                "uav_value": uav_value.cpu().numpy(),
                "ugv_value": ugv_value.cpu().numpy(),
                "value": (0.5 * (uav_value + ugv_value)).cpu().numpy(),
                "uav_entropy": uav_entropy.cpu().numpy(),
                "ugv_entropy": ugv_entropy.cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    @torch.no_grad()
    def get_single_action(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray = None,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        action_data = self.get_actions(
            uav_obs=np.asarray(uav_obs)[np.newaxis, ...],
            ugv_obs=np.asarray(ugv_obs)[np.newaxis, ...],
            critic_state=critic_state,
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

    @torch.no_grad()
    def get_values(self, uav_obs: np.ndarray, ugv_obs: np.ndarray) -> dict:
        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs, dtype=torch.float32, device=self.rollout_device
            )
            ugv_obs_t = torch.as_tensor(
                ugv_obs, dtype=torch.float32, device=self.rollout_device
            )
            return {
                "uav_value": self.uav_critic(uav_obs_t).squeeze(-1).cpu().numpy(),
                "ugv_value": self.ugv_critic(ugv_obs_t).squeeze(-1).cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    def save(self, path: str) -> None:
        torch.save(
            {
                "uav_actor": self.uav_actor.state_dict(),
                "ugv_actor": self.ugv_actor.state_dict(),
                "uav_critic": self.uav_critic.state_dict(),
                "ugv_critic": self.ugv_critic.state_dict(),
                "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
                "ugv_actor_optimizer": self.ugv_actor_optimizer.state_dict(),
                "uav_critic_optimizer": self.uav_critic_optimizer.state_dict(),
                "ugv_critic_optimizer": self.ugv_critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        for name in ("uav_actor", "ugv_actor", "uav_critic", "ugv_critic"):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in (
            "uav_actor_optimizer",
            "ugv_actor_optimizer",
            "uav_critic_optimizer",
            "ugv_critic_optimizer",
        ):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])
        self.prepare_for_training()


__all__ = ["ActorNetwork", "CriticNetwork", "IPPOPolicy", "init_weights"]

"""Actors, value critic, and centralized joint-action Q critic for MAPPO-CF."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from du_iibtd_based_fading_delta.shared.on_policy_networks import (
    JointActorCriticPolicy,
    init_weights,
)


class CentralizedQNetwork(nn.Module):
    """Estimate ``Q(s, a_uav, a_ugv)`` from global state and joint action."""

    def __init__(
        self,
        state_dim: int,
        uav_action_dim: int,
        ugv_action_dim: int,
        hidden_dims: List[int],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.uav_action_dim = int(uav_action_dim)
        self.ugv_action_dim = int(ugv_action_dim)
        self.state_norm = (
            nn.LayerNorm(self.state_dim) if use_feature_norm else nn.Identity()
        )
        layers = []
        input_dim = self.state_dim + self.uav_action_dim + self.ugv_action_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
            input_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*layers)
        self.q_head = nn.Linear(input_dim, 1)
        if use_orthogonal_init:
            self.mlp.apply(lambda module: init_weights(module, gain=np.sqrt(2)))
            init_weights(self.q_head, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        uav_action: torch.Tensor,
        ugv_action: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected critic state width {self.state_dim}, got {state.shape[-1]}"
            )
        uav_action = uav_action.to(dtype=torch.long)
        ugv_action = ugv_action.to(dtype=torch.long)
        if ((uav_action < 0) | (uav_action >= self.uav_action_dim)).any():
            raise ValueError("UAV action is outside the Q critic action range")
        if ((ugv_action < 0) | (ugv_action >= self.ugv_action_dim)).any():
            raise ValueError("UGV action is outside the Q critic action range")
        features = torch.cat(
            (
                self.state_norm(state),
                F.one_hot(uav_action, self.uav_action_dim).to(dtype=state.dtype),
                F.one_hot(ugv_action, self.ugv_action_dim).to(dtype=state.dtype),
            ),
            dim=-1,
        )
        return self.q_head(self.mlp(features)).squeeze(-1)


class MAPPOCFPolicy(JointActorCriticPolicy):
    """MAPPO policy plus a centralized joint-action Q critic."""

    def __init__(self, obs_dims: dict, action_dims: dict, config):
        super().__init__(obs_dims, action_dims, config)
        self.uav_action_dim = int(action_dims["uav_action"])
        self.ugv_action_dim = int(action_dims["ugv_action"])
        self.q_critic = CentralizedQNetwork(
            state_dim=obs_dims["critic_state"],
            uav_action_dim=self.uav_action_dim,
            ugv_action_dim=self.ugv_action_dim,
            hidden_dims=config.critic_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=config.lr_critic, eps=1e-5
        )
        self.prepare_for_training()

    def prepare_for_training(self) -> None:
        super().prepare_for_training()
        if hasattr(self, "q_critic"):
            self.q_critic.train()

    def save(self, path: str) -> None:
        torch.save(
            {
                "uav_actor": self.uav_actor.state_dict(),
                "ugv_actor": self.ugv_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "q_critic": self.q_critic.state_dict(),
                "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
                "ugv_actor_optimizer": self.ugv_actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "q_critic_optimizer": self.q_critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.uav_actor.load_state_dict(checkpoint["uav_actor"])
        self.ugv_actor.load_state_dict(checkpoint["ugv_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.q_critic.load_state_dict(checkpoint["q_critic"])
        for key, optimizer in (
            ("uav_actor_optimizer", self.uav_actor_optimizer),
            ("ugv_actor_optimizer", self.ugv_actor_optimizer),
            ("critic_optimizer", self.critic_optimizer),
            ("q_critic_optimizer", self.q_critic_optimizer),
        ):
            if key in checkpoint:
                optimizer.load_state_dict(checkpoint[key])
        self.prepare_for_training()


__all__ = ["CentralizedQNetwork", "MAPPOCFPolicy"]

"""
Neural network architectures for MAPPO.

- Actor networks (one per agent): map local observations → action logits
- Critic network (shared): maps global state → value estimate

Design choices:
- Actor uses local observation (different per agent type)
- Critic uses global state (same for both agents)
- Orthogonal initialization for stable training
- Layer normalization on input features
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple


def init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal initialization for stable PPO training."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class UAVFactorizedActorNetwork(nn.Module):
    """UAV actor with a shared backbone and separate direction/bandwidth heads.

    The environment still consumes one joint action index. This module keeps
    the external interface identical by sampling/evaluating the two categorical
    factors independently and combining them back into a single joint action.
    """

    def __init__(
        self,
        obs_dim: int,
        num_directions: int,
        num_bandwidths: int,
        hidden_dims: List[int] = [256, 128, 64],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.num_directions = int(num_directions)
        self.num_bandwidths = int(num_bandwidths)
        self.action_dim = self.num_directions * self.num_bandwidths

        self.feature_norm = nn.LayerNorm(obs_dim) if use_feature_norm else nn.Identity()

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        self.direction_head = nn.Linear(in_dim, self.num_directions)
        self.bandwidth_head = nn.Linear(in_dim, self.num_bandwidths)

        if use_orthogonal_init:
            self.mlp.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
            init_weights(self.direction_head, gain=0.01)
            init_weights(self.bandwidth_head, gain=0.01)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.feature_norm(obs)
        return self.mlp(x)

    def _masked_categorical(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Categorical:
        if action_mask is not None:
            mask = action_mask.bool()
            if (~mask.any(dim=-1)).any():
                raise RuntimeError("Action mask contains a row with no valid actions.")
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def _joint_mask_from_action_mask(
        self,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor | None:
        if action_mask is None:
            return None

        mask = action_mask.bool()
        if mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected UAV joint action mask width {self.action_dim}, got {mask.shape[-1]}"
            )

        leading_shape = mask.shape[:-1]
        return mask.reshape(*leading_shape, self.num_directions, self.num_bandwidths)

    def _direction_mask_from_joint_mask(
        self,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor | None:
        joint_mask = self._joint_mask_from_action_mask(action_mask)
        if joint_mask is None:
            return None
        return joint_mask.any(dim=-1)

    def _bandwidth_mask_from_joint_mask(
        self,
        action_mask: torch.Tensor = None,
        direction_action: torch.Tensor = None,
    ) -> torch.Tensor | None:
        joint_mask = self._joint_mask_from_action_mask(action_mask)
        if joint_mask is None or direction_action is None:
            return None

        direction_action = direction_action.to(
            device=joint_mask.device,
            dtype=torch.long,
        ).clamp(0, self.num_directions - 1)
        gather_index = direction_action.unsqueeze(-1).unsqueeze(-1).expand(
            *direction_action.shape,
            1,
            self.num_bandwidths,
        )
        return joint_mask.gather(dim=-2, index=gather_index).squeeze(-2)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
        direction_action: torch.Tensor = None,
    ) -> tuple[Categorical, Categorical]:
        x = self._encode(obs)
        direction_logits = self.direction_head(x)
        bandwidth_logits = self.bandwidth_head(x)
        direction_mask = self._direction_mask_from_joint_mask(action_mask)
        bandwidth_mask = self._bandwidth_mask_from_joint_mask(action_mask, direction_action)
        direction_dist = self._masked_categorical(direction_logits, direction_mask)
        bandwidth_dist = self._masked_categorical(bandwidth_logits, bandwidth_mask)
        return direction_dist, bandwidth_dist

    def _join_actions(
        self,
        direction_action: torch.Tensor,
        bandwidth_action: torch.Tensor,
    ) -> torch.Tensor:
        return (
            direction_action.to(dtype=torch.long) * int(self.num_bandwidths)
            + bandwidth_action.to(dtype=torch.long)
        )

    def _split_actions(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint_action = action.to(dtype=torch.long).clamp(0, self.action_dim - 1)
        direction_action = torch.div(
            joint_action,
            int(self.num_bandwidths),
            rounding_mode="floor",
        )
        bandwidth_action = torch.remainder(joint_action, int(self.num_bandwidths))
        return direction_action, bandwidth_action

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._encode(obs)
        direction_logits = self.direction_head(x)
        bandwidth_logits = self.bandwidth_head(x)

        direction_mask = self._direction_mask_from_joint_mask(action_mask)
        direction_dist = self._masked_categorical(direction_logits, direction_mask)

        if deterministic:
            direction_action = direction_dist.probs.argmax(dim=-1)
        else:
            direction_action = direction_dist.sample()

        bandwidth_mask = self._bandwidth_mask_from_joint_mask(action_mask, direction_action)
        bandwidth_dist = self._masked_categorical(bandwidth_logits, bandwidth_mask)

        if deterministic:
            bandwidth_action = bandwidth_dist.probs.argmax(dim=-1)
        else:
            bandwidth_action = bandwidth_dist.sample()

        action = self._join_actions(direction_action, bandwidth_action)
        log_prob = direction_dist.log_prob(direction_action) + bandwidth_dist.log_prob(
            bandwidth_action
        )
        entropy = direction_dist.entropy() + bandwidth_dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        direction_action, bandwidth_action = self._split_actions(action)
        direction_dist, bandwidth_dist = self.forward(
            obs,
            action_mask,
            direction_action=direction_action,
        )
        log_prob = direction_dist.log_prob(direction_action) + bandwidth_dist.log_prob(
            bandwidth_action
        )
        entropy = direction_dist.entropy() + bandwidth_dist.entropy()
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized critic network that estimates state value.
    
    Architecture:
        global_state → LayerNorm → MLP → V(s)
        
    Uses global state (all agents' info + environment state) for training.
    
    Args:
        state_dim: Dimension of the global state.
        hidden_dims: List of hidden layer dimensions.
        use_feature_norm: Whether to apply layer normalization to input.
        use_orthogonal_init: Whether to use orthogonal weight initialization.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim

        # Input normalization
        self.feature_norm = nn.LayerNorm(state_dim) if use_feature_norm else nn.Identity()

        # Build MLP
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Value head
        self.value_head = nn.Linear(in_dim, 1)

        # Initialize weights
        if use_orthogonal_init:
            self.mlp.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
            init_weights(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim) global state tensor.
            
        Returns:
            value: (batch, 1) estimated state value.
        """
        x = self.feature_norm(state)
        x = self.mlp(x)
        value = self.value_head(x)
        return value


class MAPPOPolicy:
    """
    Container for the UAV PPO actor and shared critic.
    
    Handles device management, action sampling, and model saving/loading.
    """

    def __init__(self, obs_dims: dict, action_dims: dict, config):
        """
        Args:
            obs_dims: Dict with at least 'uav_obs' and 'critic_state' dimensions.
            action_dims: Dict with at least 'uav_action' size.
            config: MAPPOConfig with hyperparameters.
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.rollout_device = (
            torch.device("cpu")
            if self.device.type == "cuda"
            else self.device
        )
        self._use_separate_rollout_policy = self.rollout_device != self.device

        # UAV Actor: factorized direction/bandwidth policy with a shared trunk.
        self.uav_actor = UAVFactorizedActorNetwork(
            obs_dim=obs_dims["uav_obs"],
            num_directions=action_dims["uav_direction"],
            num_bandwidths=action_dims["uav_bandwidth"],
            hidden_dims=config.actor_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)

        # Shared Critic (uses global state)
        self.critic = CriticNetwork(
            state_dim=obs_dims["critic_state"],
            hidden_dims=config.critic_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)

        # Optimizers
        self.uav_actor_optimizer = torch.optim.Adam(
            self.uav_actor.parameters(), lr=config.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic, eps=1e-5
        )

        if self._use_separate_rollout_policy:
            self.rollout_uav_actor = copy.deepcopy(self.uav_actor).to(self.rollout_device)
            self.rollout_critic = copy.deepcopy(self.critic).to(self.rollout_device)
        else:
            self.rollout_uav_actor = self.uav_actor
            self.rollout_critic = self.critic

        self._rollout_params_dirty = True
        self.sync_rollout_policy()
        self.prepare_for_training()

    def mark_rollout_stale(self) -> None:
        self._rollout_params_dirty = True

    def prepare_for_training(self) -> None:
        self.uav_actor.train()
        self.critic.train()

    def sync_rollout_policy(self) -> None:
        if self._use_separate_rollout_policy:
            self.rollout_uav_actor.load_state_dict(self.uav_actor.state_dict())
            self.rollout_critic.load_state_dict(self.critic.state_dict())
            self.rollout_uav_actor.eval()
            self.rollout_critic.eval()
        self._rollout_params_dirty = False

    def _ensure_rollout_policy(self) -> None:
        if self._rollout_params_dirty:
            self.sync_rollout_policy()

    def _enter_rollout_mode(self) -> tuple[bool, bool] | None:
        self._ensure_rollout_policy()
        if self._use_separate_rollout_policy:
            return None

        previous_modes = (
            bool(self.rollout_uav_actor.training),
            bool(self.rollout_critic.training),
        )
        self.rollout_uav_actor.eval()
        self.rollout_critic.eval()
        return previous_modes

    def _exit_rollout_mode(self, previous_modes: tuple[bool, bool] | None) -> None:
        if previous_modes is None:
            return
        self.rollout_uav_actor.train(previous_modes[0])
        self.rollout_critic.train(previous_modes[1])

    @torch.no_grad()
    def get_actions(
        self,
        uav_obs: np.ndarray,
        critic_state: np.ndarray = None,
        uav_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        """
        Get UAV actions during rollout collection.
        
        Args:
            uav_obs: (num_envs, uav_obs_dim) UAV observations.
            critic_state: (num_envs, critic_state_dim) global state.
            uav_action_mask: (num_envs, uav_action_dim) bool mask for valid UAV actions.
            deterministic: Whether to use greedy actions.
            
        Returns:
            Dict with UAV actions, log_probs, value, entropies.
        """
        if critic_state is None:
            raise ValueError("critic_state must be provided for value estimation.")

        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs,
                dtype=torch.float32,
                device=self.rollout_device,
            )
            state_t = torch.as_tensor(
                critic_state,
                dtype=torch.float32,
                device=self.rollout_device,
            )
            if uav_action_mask is not None:
                uav_mask_t = torch.as_tensor(
                    uav_action_mask,
                    dtype=torch.bool,
                    device=self.rollout_device,
                )
            else:
                uav_mask_t = None

            uav_action, uav_logp, uav_ent = self.rollout_uav_actor.get_action(
                uav_obs_t, uav_mask_t, deterministic
            )
            value = self.rollout_critic(state_t).squeeze(-1)

            return {
                "uav_action": uav_action.cpu().numpy(),
                "uav_log_prob": uav_logp.cpu().numpy(),
                "value": value.cpu().numpy(),
                "uav_entropy": uav_ent.cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    @torch.no_grad()
    def get_single_action(
        self,
        uav_obs: np.ndarray,
        critic_state: np.ndarray = None,
        uav_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        """Convenience wrapper for one unbatched environment step."""
        action_data = self.get_actions(
            uav_obs=np.asarray(uav_obs)[np.newaxis, ...],
            critic_state=np.asarray(critic_state)[np.newaxis, ...],
            uav_action_mask=(
                None if uav_action_mask is None else np.asarray(uav_action_mask)[np.newaxis, ...]
            ),
            deterministic=deterministic,
        )
        return {
            "uav_action": int(action_data["uav_action"][0]),
            "uav_log_prob": float(action_data["uav_log_prob"][0]),
            "value": float(action_data["value"][0]),
            "uav_entropy": float(action_data["uav_entropy"][0]),
        }

    def get_value(self, critic_state: np.ndarray) -> np.ndarray:
        """Get value estimate for given global states."""
        with torch.no_grad():
            previous_modes = self._enter_rollout_mode()
            try:
                state_t = torch.as_tensor(
                    critic_state,
                    dtype=torch.float32,
                    device=self.rollout_device,
                )
                value = self.rollout_critic(state_t).squeeze(-1)
                return value.cpu().numpy()
            finally:
                self._exit_rollout_mode(previous_modes)

    def save(self, path: str):
        """Save the UAV actor + critic parameters."""
        torch.save({
            "uav_actor": self.uav_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load UAV actor + critic parameters; tolerate older dual-actor checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.uav_actor.load_state_dict(checkpoint["uav_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "uav_actor_optimizer" in checkpoint:
            self.uav_actor_optimizer.load_state_dict(checkpoint["uav_actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.prepare_for_training()
        self.mark_rollout_stale()

"""
Neural network architectures for IPPO.

- Actor networks (one per agent): map local observations → action logits
- Critic networks (one per agent): map local observations → value estimates

Design choices:
- Actor uses local observation (different per agent type)
- Each critic uses only its own agent's local observation
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


class ActorNetwork(nn.Module):
    """
    Actor network that outputs a categorical distribution over discrete actions.
    
    Architecture:
        obs → LayerNorm → MLP → action logits → Categorical distribution
        
    Args:
        obs_dim: Dimension of the agent's local observation.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer dimensions.
        use_feature_norm: Whether to apply layer normalization to input.
        use_orthogonal_init: Whether to use orthogonal weight initialization.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Input normalization
        self.feature_norm = nn.LayerNorm(obs_dim) if use_feature_norm else nn.Identity()

        # Build MLP
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Action head (smaller init gain for output layer)
        self.action_head = nn.Linear(in_dim, action_dim)

        # Initialize weights
        if use_orthogonal_init:
            self.mlp.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
            init_weights(self.action_head, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Categorical:
        """
        Forward pass.
        
        Args:
            obs: (batch, obs_dim) observation tensor.
            action_mask: (batch, action_dim) bool tensor, True for valid actions.
            
        Returns:
            Categorical distribution over actions.
        """
        x = self.feature_norm(obs)
        x = self.mlp(x)
        logits = self.action_head(x)

        if action_mask is not None:
            mask = action_mask.bool()
            if (~mask.any(dim=-1)).any():
                raise RuntimeError("Action mask contains a row with no valid actions.")
            logits = logits.masked_fill(~mask, -1e9)

        return Categorical(logits=logits)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability + entropy.
        
        Args:
            obs: (batch, obs_dim) observation tensor.
            action_mask: (batch, action_dim) bool tensor, True for valid actions.
            deterministic: If True, return argmax action.
            
        Returns:
            action: (batch,) sampled or argmax action indices.
            log_prob: (batch,) log probability of the action.
            entropy: (batch,) entropy of the distribution.
        """
        dist = self.forward(obs, action_mask)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            obs: (batch, obs_dim) observation tensor.
            action: (batch,) action indices.
            action_mask: (batch, action_dim) bool tensor, True for valid actions.
            
        Returns:
            log_prob: (batch,) log probability.
            entropy: (batch,) entropy.
        """
        dist = self.forward(obs, action_mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class UAVFactorizedActorNetwork(nn.Module):
    """UAV actor with shared features and separate direction/bandwidth/bit heads.

    The environment still consumes one joint action index. This module keeps
    the external interface identical by sampling/evaluating the categorical
    factors independently and combining them back into a single joint action.
    """

    def __init__(
        self,
        obs_dim: int,
        num_directions: int,
        num_bandwidths: int,
        num_quant_bits: int,
        hidden_dims: List[int] = [256, 128, 64],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.num_directions = int(num_directions)
        self.num_bandwidths = int(num_bandwidths)
        self.num_quant_bits = int(num_quant_bits)
        self.action_dim = self.num_directions * self.num_bandwidths * self.num_quant_bits

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
        self.quant_bit_head = nn.Linear(in_dim, self.num_quant_bits)

        if use_orthogonal_init:
            self.mlp.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
            init_weights(self.direction_head, gain=0.01)
            init_weights(self.bandwidth_head, gain=0.01)
            init_weights(self.quant_bit_head, gain=0.01)

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

    def _direction_mask_from_joint_mask(
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

        # The environment exposes a block-structured mask:
        # each valid direction enables every bandwidth/quantization choice.
        leading_shape = mask.shape[:-1]
        reshaped = mask.reshape(
            *leading_shape,
            self.num_directions,
            self.num_bandwidths,
            self.num_quant_bits,
        )
        return reshaped.any(dim=-1).any(dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> tuple[Categorical, Categorical, Categorical]:
        x = self._encode(obs)
        direction_logits = self.direction_head(x)
        bandwidth_logits = self.bandwidth_head(x)
        quant_bit_logits = self.quant_bit_head(x)
        direction_mask = self._direction_mask_from_joint_mask(action_mask)
        direction_dist = self._masked_categorical(direction_logits, direction_mask)
        bandwidth_dist = self._masked_categorical(bandwidth_logits, None)
        quant_bit_dist = self._masked_categorical(quant_bit_logits, None)
        return direction_dist, bandwidth_dist, quant_bit_dist

    def _join_actions(
        self,
        direction_action: torch.Tensor,
        bandwidth_action: torch.Tensor,
        quant_bit_action: torch.Tensor,
    ) -> torch.Tensor:
        return (
            direction_action.to(dtype=torch.long)
            * int(self.num_bandwidths)
            * int(self.num_quant_bits)
            + bandwidth_action.to(dtype=torch.long) * int(self.num_quant_bits)
            + quant_bit_action.to(dtype=torch.long)
        )

    def _split_actions(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_action = action.to(dtype=torch.long).clamp(0, self.action_dim - 1)
        quant_bit_action = torch.remainder(joint_action, int(self.num_quant_bits))
        action_without_bit = torch.div(
            joint_action,
            int(self.num_quant_bits),
            rounding_mode="floor",
        )
        bandwidth_action = torch.remainder(action_without_bit, int(self.num_bandwidths))
        direction_action = torch.div(
            action_without_bit,
            int(self.num_bandwidths),
            rounding_mode="floor",
        )
        return direction_action, bandwidth_action, quant_bit_action

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction_dist, bandwidth_dist, quant_bit_dist = self.forward(obs, action_mask)

        if deterministic:
            direction_action = direction_dist.probs.argmax(dim=-1)
            bandwidth_action = bandwidth_dist.probs.argmax(dim=-1)
            quant_bit_action = quant_bit_dist.probs.argmax(dim=-1)
        else:
            direction_action = direction_dist.sample()
            bandwidth_action = bandwidth_dist.sample()
            quant_bit_action = quant_bit_dist.sample()

        action = self._join_actions(direction_action, bandwidth_action, quant_bit_action)
        log_prob = direction_dist.log_prob(direction_action) + bandwidth_dist.log_prob(
            bandwidth_action
        ) + quant_bit_dist.log_prob(quant_bit_action)
        entropy = direction_dist.entropy() + bandwidth_dist.entropy() + quant_bit_dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        direction_dist, bandwidth_dist, quant_bit_dist = self.forward(obs, action_mask)
        direction_action, bandwidth_action, quant_bit_action = self._split_actions(action)
        log_prob = direction_dist.log_prob(direction_action) + bandwidth_dist.log_prob(
            bandwidth_action
        ) + quant_bit_dist.log_prob(quant_bit_action)
        entropy = direction_dist.entropy() + bandwidth_dist.entropy() + quant_bit_dist.entropy()
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Local critic network that estimates state value.
    
    Architecture:
        local_obs → LayerNorm → MLP → V(o_i)
    
    Args:
        state_dim: Dimension of the local observation.
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
            state: (batch, state_dim) local observation tensor.
            
        Returns:
            value: (batch, 1) estimated state value.
        """
        x = self.feature_norm(state)
        x = self.mlp(x)
        value = self.value_head(x)
        return value


class IPPOPolicy:
    """
    Container for IPPO networks (2 actors + 2 local critics).

    Handles device management, action sampling, and model saving/loading.
    """

    def __init__(self, obs_dims: dict, action_dims: dict, config):
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

        self.uav_actor = UAVFactorizedActorNetwork(
            obs_dim=obs_dims["uav_obs"],
            num_directions=action_dims["uav_direction"],
            num_bandwidths=action_dims["uav_bandwidth"],
            num_quant_bits=action_dims["uav_quant_bits"],
            hidden_dims=config.actor_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)
        self.ugv_actor = ActorNetwork(
            obs_dim=obs_dims["ugv_obs"],
            action_dim=action_dims["ugv_action"],
            hidden_dims=config.actor_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)

        self.uav_critic = CriticNetwork(
            state_dim=obs_dims["uav_obs"],
            hidden_dims=config.critic_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)
        self.ugv_critic = CriticNetwork(
            state_dim=obs_dims["ugv_obs"],
            hidden_dims=config.critic_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
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

        if self._use_separate_rollout_policy:
            self.rollout_uav_actor = copy.deepcopy(self.uav_actor).to(self.rollout_device)
            self.rollout_ugv_actor = copy.deepcopy(self.ugv_actor).to(self.rollout_device)
            self.rollout_uav_critic = copy.deepcopy(self.uav_critic).to(self.rollout_device)
            self.rollout_ugv_critic = copy.deepcopy(self.ugv_critic).to(self.rollout_device)
        else:
            self.rollout_uav_actor = self.uav_actor
            self.rollout_ugv_actor = self.ugv_actor
            self.rollout_uav_critic = self.uav_critic
            self.rollout_ugv_critic = self.ugv_critic

        self._rollout_params_dirty = True
        self.sync_rollout_policy()
        self.prepare_for_training()

    def mark_rollout_stale(self) -> None:
        self._rollout_params_dirty = True

    def prepare_for_training(self) -> None:
        self.uav_actor.train()
        self.ugv_actor.train()
        self.uav_critic.train()
        self.ugv_critic.train()

    def sync_rollout_policy(self) -> None:
        if self._use_separate_rollout_policy:
            self.rollout_uav_actor.load_state_dict(self.uav_actor.state_dict())
            self.rollout_ugv_actor.load_state_dict(self.ugv_actor.state_dict())
            self.rollout_uav_critic.load_state_dict(self.uav_critic.state_dict())
            self.rollout_ugv_critic.load_state_dict(self.ugv_critic.state_dict())
            self.rollout_uav_actor.eval()
            self.rollout_ugv_actor.eval()
            self.rollout_uav_critic.eval()
            self.rollout_ugv_critic.eval()
        self._rollout_params_dirty = False

    def _ensure_rollout_policy(self) -> None:
        if self._rollout_params_dirty:
            self.sync_rollout_policy()

    def _enter_rollout_mode(self) -> tuple[bool, bool, bool, bool] | None:
        self._ensure_rollout_policy()
        if self._use_separate_rollout_policy:
            return None

        previous_modes = (
            bool(self.rollout_uav_actor.training),
            bool(self.rollout_ugv_actor.training),
            bool(self.rollout_uav_critic.training),
            bool(self.rollout_ugv_critic.training),
        )
        self.rollout_uav_actor.eval()
        self.rollout_ugv_actor.eval()
        self.rollout_uav_critic.eval()
        self.rollout_ugv_critic.eval()
        return previous_modes

    def _exit_rollout_mode(
        self,
        previous_modes: tuple[bool, bool, bool, bool] | None,
    ) -> None:
        if previous_modes is None:
            return
        self.rollout_uav_actor.train(previous_modes[0])
        self.rollout_ugv_actor.train(previous_modes[1])
        self.rollout_uav_critic.train(previous_modes[2])
        self.rollout_ugv_critic.train(previous_modes[3])

    @torch.no_grad()
    def get_actions(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        """Get actions and local value estimates for both agents."""
        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs,
                dtype=torch.float32,
                device=self.rollout_device,
            )
            ugv_obs_t = torch.as_tensor(
                ugv_obs,
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
            if ugv_action_mask is not None:
                ugv_mask_t = torch.as_tensor(
                    ugv_action_mask,
                    dtype=torch.bool,
                    device=self.rollout_device,
                )
            else:
                ugv_mask_t = None

            uav_action, uav_logp, uav_ent = self.rollout_uav_actor.get_action(
                uav_obs_t, uav_mask_t, deterministic
            )
            ugv_action, ugv_logp, ugv_ent = self.rollout_ugv_actor.get_action(
                ugv_obs_t, ugv_mask_t, deterministic
            )
            uav_value = self.rollout_uav_critic(uav_obs_t).squeeze(-1)
            ugv_value = self.rollout_ugv_critic(ugv_obs_t).squeeze(-1)
            value = 0.5 * (uav_value + ugv_value)

            return {
                "uav_action": uav_action.cpu().numpy(),
                "ugv_action": ugv_action.cpu().numpy(),
                "uav_log_prob": uav_logp.cpu().numpy(),
                "ugv_log_prob": ugv_logp.cpu().numpy(),
                "uav_value": uav_value.cpu().numpy(),
                "ugv_value": ugv_value.cpu().numpy(),
                "value": value.cpu().numpy(),
                "uav_entropy": uav_ent.cpu().numpy(),
                "ugv_entropy": ugv_ent.cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    @torch.no_grad()
    def get_single_action(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        uav_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        """Convenience wrapper for one unbatched environment step."""
        action_data = self.get_actions(
            uav_obs=np.asarray(uav_obs)[np.newaxis, ...],
            ugv_obs=np.asarray(ugv_obs)[np.newaxis, ...],
            uav_action_mask=(
                None if uav_action_mask is None else np.asarray(uav_action_mask)[np.newaxis, ...]
            ),
            ugv_action_mask=(
                None if ugv_action_mask is None else np.asarray(ugv_action_mask)[np.newaxis, ...]
            ),
            deterministic=deterministic,
        )
        return {
            "uav_action": int(action_data["uav_action"][0]),
            "ugv_action": int(action_data["ugv_action"][0]),
            "uav_log_prob": float(action_data["uav_log_prob"][0]),
            "ugv_log_prob": float(action_data["ugv_log_prob"][0]),
            "uav_value": float(action_data["uav_value"][0]),
            "ugv_value": float(action_data["ugv_value"][0]),
            "value": float(action_data["value"][0]),
            "uav_entropy": float(action_data["uav_entropy"][0]),
            "ugv_entropy": float(action_data["ugv_entropy"][0]),
        }

    @torch.no_grad()
    def get_values(self, uav_obs: np.ndarray, ugv_obs: np.ndarray) -> dict:
        """Get local value estimates for both agents."""
        previous_modes = self._enter_rollout_mode()
        try:
            uav_obs_t = torch.as_tensor(
                uav_obs,
                dtype=torch.float32,
                device=self.rollout_device,
            )
            ugv_obs_t = torch.as_tensor(
                ugv_obs,
                dtype=torch.float32,
                device=self.rollout_device,
            )
            uav_value = self.rollout_uav_critic(uav_obs_t).squeeze(-1)
            ugv_value = self.rollout_ugv_critic(ugv_obs_t).squeeze(-1)
            return {
                "uav_value": uav_value.cpu().numpy(),
                "ugv_value": ugv_value.cpu().numpy(),
            }
        finally:
            self._exit_rollout_mode(previous_modes)

    def save(self, path: str):
        """Save all model parameters."""
        torch.save({
            "uav_actor": self.uav_actor.state_dict(),
            "ugv_actor": self.ugv_actor.state_dict(),
            "uav_critic": self.uav_critic.state_dict(),
            "ugv_critic": self.ugv_critic.state_dict(),
            "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
            "ugv_actor_optimizer": self.ugv_actor_optimizer.state_dict(),
            "uav_critic_optimizer": self.uav_critic_optimizer.state_dict(),
            "ugv_critic_optimizer": self.ugv_critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load all model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.uav_actor.load_state_dict(checkpoint["uav_actor"])
        self.ugv_actor.load_state_dict(checkpoint["ugv_actor"])
        self.uav_critic.load_state_dict(checkpoint["uav_critic"])
        self.ugv_critic.load_state_dict(checkpoint["ugv_critic"])
        self.uav_actor_optimizer.load_state_dict(checkpoint["uav_actor_optimizer"])
        self.ugv_actor_optimizer.load_state_dict(checkpoint["ugv_actor_optimizer"])
        self.uav_critic_optimizer.load_state_dict(checkpoint["uav_critic_optimizer"])
        self.ugv_critic_optimizer.load_state_dict(checkpoint["ugv_critic_optimizer"])
        self.prepare_for_training()
        self.mark_rollout_stale()

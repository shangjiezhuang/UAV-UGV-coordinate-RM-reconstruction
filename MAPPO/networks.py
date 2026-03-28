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
            # Safety fallback: if a row has no valid action, allow all actions.
            no_valid = ~mask.any(dim=-1, keepdim=True)
            if no_valid.any():
                mask = torch.where(no_valid, torch.ones_like(mask), mask)
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


class UAVActorNetwork(nn.Module):
    """UAV actor with independent movement and bandwidth heads."""

    def __init__(
        self,
        obs_dim: int,
        move_action_dim: int,
        bw_action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        use_feature_norm: bool = True,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.move_action_dim = move_action_dim
        self.bw_action_dim = bw_action_dim
        self.feature_norm = nn.LayerNorm(obs_dim) if use_feature_norm else nn.Identity()

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.move_head = nn.Linear(in_dim, move_action_dim)
        self.bw_head = nn.Linear(in_dim, bw_action_dim)

        if use_orthogonal_init:
            self.mlp.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
            init_weights(self.move_head, gain=0.01)
            init_weights(self.bw_head, gain=0.01)

    def _build_dist(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Categorical:
        if action_mask is not None:
            mask = action_mask.bool()
            no_valid = ~mask.any(dim=-1, keepdim=True)
            if no_valid.any():
                mask = torch.where(no_valid, torch.ones_like(mask), mask)
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def forward(
        self,
        obs: torch.Tensor,
        move_action_mask: torch.Tensor = None,
        bw_action_mask: torch.Tensor = None,
    ) -> Tuple[Categorical, Categorical]:
        x = self.feature_norm(obs)
        x = self.mlp(x)
        move_logits = self.move_head(x)
        bw_logits = self.bw_head(x)
        return (
            self._build_dist(move_logits, move_action_mask),
            self._build_dist(bw_logits, bw_action_mask),
        )

    def get_action(
        self,
        obs: torch.Tensor,
        move_action_mask: torch.Tensor = None,
        bw_action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        move_dist, bw_dist = self.forward(obs, move_action_mask, bw_action_mask)
        if deterministic:
            move_action = move_dist.probs.argmax(dim=-1)
            bw_action = bw_dist.probs.argmax(dim=-1)
        else:
            move_action = move_dist.sample()
            bw_action = bw_dist.sample()

        move_log_prob = move_dist.log_prob(move_action)
        bw_log_prob = bw_dist.log_prob(bw_action)
        move_entropy = move_dist.entropy()
        bw_entropy = bw_dist.entropy()
        return (
            move_action,
            bw_action,
            move_log_prob,
            bw_log_prob,
            move_entropy,
            bw_entropy,
        )

    def evaluate_action(
        self,
        obs: torch.Tensor,
        move_action: torch.Tensor,
        bw_action: torch.Tensor,
        move_action_mask: torch.Tensor = None,
        bw_action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        move_dist, bw_dist = self.forward(obs, move_action_mask, bw_action_mask)
        move_log_prob = move_dist.log_prob(move_action)
        bw_log_prob = bw_dist.log_prob(bw_action)
        move_entropy = move_dist.entropy()
        bw_entropy = bw_dist.entropy()
        return move_log_prob, bw_log_prob, move_entropy, bw_entropy


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
    Container for all MAPPO networks (2 actors + 1 shared critic).
    
    Handles device management, action sampling, and model saving/loading.
    """

    def __init__(self, obs_dims: dict, action_dims: dict, config):
        """
        Args:
            obs_dims: Dict with 'uav_obs', 'ugv_obs', 'critic_state' dimensions.
            action_dims: Dict with 'uav_action', 'ugv_action' sizes.
            config: MAPPOConfig with hyperparameters.
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # UAV Actor
        self.uav_actor = UAVActorNetwork(
            obs_dim=obs_dims["uav_obs"],
            move_action_dim=action_dims["uav_move_action"],
            bw_action_dim=action_dims["uav_bw_action"],
            hidden_dims=config.actor_hidden_dims,
            use_feature_norm=config.use_feature_norm,
            use_orthogonal_init=config.use_orthogonal_init,
        ).to(self.device)

        # UGV Actor
        self.ugv_actor = ActorNetwork(
            obs_dim=obs_dims["ugv_obs"],
            action_dim=action_dims["ugv_action"],
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
        self.ugv_actor_optimizer = torch.optim.Adam(
            self.ugv_actor.parameters(), lr=config.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic, eps=1e-5
        )

    @torch.no_grad()
    def get_actions(
        self,
        uav_obs: np.ndarray,
        ugv_obs: np.ndarray,
        critic_state: np.ndarray,
        uav_move_action_mask: np.ndarray = None,
        uav_bw_action_mask: np.ndarray = None,
        ugv_action_mask: np.ndarray = None,
        deterministic: bool = False,
    ) -> dict:
        """
        Get actions for both agents during rollout collection.
        
        Args:
            uav_obs: (num_envs, uav_obs_dim) UAV observations.
            ugv_obs: (num_envs, ugv_obs_dim) UGV observations.
            critic_state: (num_envs, critic_state_dim) global state.
            uav_move_action_mask: (num_envs, move_action_dim) bool mask for valid UAV moves.
            uav_bw_action_mask: (num_envs, bw_action_dim) bool mask for valid UAV bandwidth picks.
            ugv_action_mask: (num_envs, ugv_action_dim) bool mask for valid UGV actions.
            deterministic: Whether to use greedy actions.
            
        Returns:
            Dict with actions, log_probs, value, entropies.
        """
        uav_obs_t = torch.as_tensor(uav_obs, dtype=torch.float32, device=self.device)
        ugv_obs_t = torch.as_tensor(ugv_obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(critic_state, dtype=torch.float32, device=self.device)
        if uav_move_action_mask is not None:
            uav_move_mask_t = torch.as_tensor(
                uav_move_action_mask,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            uav_move_mask_t = None
        if uav_bw_action_mask is not None:
            uav_bw_mask_t = torch.as_tensor(
                uav_bw_action_mask,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            uav_bw_mask_t = None
        if ugv_action_mask is not None:
            ugv_mask_t = torch.as_tensor(ugv_action_mask, dtype=torch.bool, device=self.device)
        else:
            ugv_mask_t = None

        # UAV action heads
        (
            uav_move_action,
            uav_bw_action,
            uav_move_logp,
            uav_bw_logp,
            uav_move_ent,
            uav_bw_ent,
        ) = self.uav_actor.get_action(
            uav_obs_t,
            uav_move_mask_t,
            uav_bw_mask_t,
            deterministic,
        )
        uav_logp = uav_move_logp + uav_bw_logp
        uav_ent = uav_move_ent + uav_bw_ent
        uav_action = (
            uav_move_action * int(self.uav_actor.bw_action_dim) + uav_bw_action
        )

        # UGV action
        ugv_action, ugv_logp, ugv_ent = self.ugv_actor.get_action(
            ugv_obs_t, ugv_mask_t, deterministic
        )

        # Value estimate
        value = self.critic(state_t).squeeze(-1)

        return {
            "uav_move_action": uav_move_action.cpu().numpy(),
            "uav_bw_action": uav_bw_action.cpu().numpy(),
            "uav_action": uav_action.cpu().numpy(),
            "ugv_action": ugv_action.cpu().numpy(),
            "uav_log_prob": uav_logp.cpu().numpy(),
            "ugv_log_prob": ugv_logp.cpu().numpy(),
            "value": value.cpu().numpy(),
            "uav_entropy": uav_ent.cpu().numpy(),
            "uav_move_entropy": uav_move_ent.cpu().numpy(),
            "uav_bw_entropy": uav_bw_ent.cpu().numpy(),
            "ugv_entropy": ugv_ent.cpu().numpy(),
        }

    def get_value(self, critic_state: np.ndarray) -> np.ndarray:
        """Get value estimate for given global states."""
        with torch.no_grad():
            state_t = torch.as_tensor(critic_state, dtype=torch.float32, device=self.device)
            value = self.critic(state_t).squeeze(-1)
            return value.cpu().numpy()

    def save(self, path: str):
        """Save all model parameters."""
        torch.save({
            "uav_actor": self.uav_actor.state_dict(),
            "ugv_actor": self.ugv_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "uav_actor_optimizer": self.uav_actor_optimizer.state_dict(),
            "ugv_actor_optimizer": self.ugv_actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load all model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.uav_actor.load_state_dict(checkpoint["uav_actor"])
        self.ugv_actor.load_state_dict(checkpoint["ugv_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.uav_actor_optimizer.load_state_dict(checkpoint["uav_actor_optimizer"])
        self.ugv_actor_optimizer.load_state_dict(checkpoint["ugv_actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

"""Compatibility exports for the shared MAPPO network implementation."""

from du_iibtd_based_fading_delta.MAPPO_quant.networks import (
    ActorNetwork,
    CriticNetwork,
    MAPPOPolicy,
    init_weights,
)

__all__ = ["ActorNetwork", "CriticNetwork", "MAPPOPolicy", "init_weights"]

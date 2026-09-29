"""MAPPO-specific policy name backed by shared actor/critic networks."""

from du_iibtd_based_fading_delta.shared.on_policy_networks import (
    ActorNetwork,
    CriticNetwork,
    JointActorCriticPolicy,
    init_weights,
)


class MAPPOPolicy(JointActorCriticPolicy):
    """Two-actor centralized-critic policy used by MAPPO."""


__all__ = ["ActorNetwork", "CriticNetwork", "MAPPOPolicy", "init_weights"]

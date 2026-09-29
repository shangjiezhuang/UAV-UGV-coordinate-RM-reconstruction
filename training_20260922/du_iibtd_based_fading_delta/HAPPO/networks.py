"""Network container for standard HAPPO.

HAPPO uses the shared two-actor centralized-critic architecture. Its defining
difference is the sequential actor update implemented by the trainer.
"""

from du_iibtd_based_fading_delta.shared.on_policy_networks import JointActorCriticPolicy


class HAPPOPolicy(JointActorCriticPolicy):
    """Two-actor centralized-critic network container for HAPPO."""


__all__ = ["HAPPOPolicy"]

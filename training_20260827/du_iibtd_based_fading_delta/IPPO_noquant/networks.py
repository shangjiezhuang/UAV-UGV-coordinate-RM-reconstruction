"""Compatibility export for the shared based IPPO policy networks."""

from du_iibtd_based_fading_delta.IPPO_quant.networks import (
    ActorNetwork,
    CriticNetwork,
    IPPOPolicy,
    init_weights,
)

__all__ = ["ActorNetwork", "CriticNetwork", "IPPOPolicy", "init_weights"]

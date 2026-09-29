"""Compatibility shim for the algorithm-neutral shared training launcher."""

from du_iibtd_based_fading_delta.shared.train_runner import (  # noqa: F401
    run_algorithm_entrypoint,
    run_training_entrypoint,
)

__all__ = ["run_algorithm_entrypoint", "run_training_entrypoint"]

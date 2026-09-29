# PPO+A* Current Support

The UAV uses the shared single-agent PPO implementation.  The UGV uses the
environment's existing path-corridor communication-support target and passes
that road target to the shared cached A* router.

The standalone training entrypoint was removed from this version on
2026-09-20 because this method is not part of the current paper comparison.
Use the frozen `dense_uncertainty_len022_20260917` source to reproduce the
historical standalone experiment. Shared controller support is retained.

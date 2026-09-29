# PPO+A* Two-Path Support

The UAV uses the shared single-agent PPO implementation.  The UGV predicts
the x-first and y-first shortest UAV route prefixes, taking at most one
ensemble refresh batch of future sampling positions from each.  Those
positions produce at most twice-the-batch-size road support candidates.  The
candidate that robustly supports both route prefixes is passed to the shared
cached A* router.

The standalone training entrypoint was removed from this version on
2026-09-20 because this method is not trained separately in the current paper
comparison. Use the frozen `dense_uncertainty_len022_20260917` source to
reproduce that historical experiment. `controller.py` and package exports
are retained: the Recovery controller, power control, and UGV control still
depend on them.

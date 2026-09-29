# PPO+A* Two-Path Support with Recovery

This method keeps the existing two-path predictive-support controller as its
normal mode and gives the current-link Support controller priority when real
queue service is at risk.

The controller enters recovery immediately when queued data is in outage.  A
non-outage link triggers recovery only when the normalized backlog is above the
urgent threshold and service has remained below the current arrival rate for
two consecutive steps.  Recovery targets the UAV's current path-corridor
support point.  Hysteresis keeps recovery active until the queue is low and the
link has been healthy for several steps.  Returning to normal mode invalidates
the old two-path forecast so it is rebuilt from the current UAV position.

The controller never reads the episode length, remaining steps, or termination
time.  Deadline-based recovery and standalone queue-growth recovery are not
part of this method.  The clean-observation comparison removes episode time,
link SNR, and previous-action quantization/packet context from every A* method.
LOS remains observable.  The UAV action mask checks only current-step geometry
and energy feasibility, never the number of steps left in the episode.

Defaults:

- recovery entry backlog: `0.50` (the shared Support threshold)
- recovery exit backlog: `0.20`
- consecutive poor-service trigger: `2` steps (only above entry backlog)
- minimum recovery hold: `2` steps
- healthy-link exit requirement: `2` steps
- service margin: `1.00`

The uncertainty-map planner is unchanged.  Set its improvement threshold in
the usual way, for example:

```bash
python -m du_iibtd_based_fading_delta.PPO_AStar_2PathSupportRecovery.train \
  --variant quant \
  --hybrid_uncertainty_improvement_threshold 0.08
```

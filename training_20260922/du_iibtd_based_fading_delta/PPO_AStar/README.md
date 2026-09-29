# PPO+A* Target-Only Control

The UAV is trained with the existing single-agent PPO implementation.  The UGV
follows a cached four-neighbour A* route toward the current UAV planner target.
This is the explicit support-disabled control used in the three-way A*
comparison.
The route is recomputed when that target changes, after an episode reset, or
when the UGV deviates from the cached route.  Before a planner target exists,
the UGV routes toward the UAV's current grid cell.

Run a short quant training job with:

```bash
python -m du_iibtd_based_fading_delta.PPO_AStar.train --variant quant [common training options]
```

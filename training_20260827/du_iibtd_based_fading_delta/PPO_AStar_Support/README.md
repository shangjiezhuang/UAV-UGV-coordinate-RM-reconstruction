# PPO+A* Current Support

The UAV uses the shared single-agent PPO implementation.  The UGV uses the
environment's existing path-corridor communication-support target and passes
that road target to the shared cached A* router.

```bash
python -m du_iibtd_based_fading_delta.PPO_AStar_Support.train --variant noquant
```

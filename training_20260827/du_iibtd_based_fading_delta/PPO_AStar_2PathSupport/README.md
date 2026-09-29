# PPO+A* Two-Path Support

The UAV uses the shared single-agent PPO implementation.  The UGV predicts
the x-first and y-first shortest UAV route prefixes, taking at most one
ensemble refresh batch of future sampling positions from each.  Those
positions produce at most twice-the-batch-size road support candidates.  The
candidate that robustly supports both route prefixes is passed to the shared
cached A* router.

```bash
python -m du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.train --variant noquant
```

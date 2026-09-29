# Greedy + A* Two-Path Support

This non-learning baseline replaces the paper method's UAV PPO actor with the
shared uncertainty-path greedy UAV policy. The UGV still uses exactly the same
two-path predictive support target selector and cached A* router as
`PPO_AStar_2PathSupport`.

Both communication variants are evaluated through the frozen shared
environment:

```bash
python -m du_iibtd_based_fading_delta.Greedy_AStar_2PathSupport.train \
  --variant quant \
  --config /path/to/config.json \
  --output /path/to/greedy_astar_2path_quant.json
```

Use `--variant noquant` for the unquantized source-data baseline. No checkpoint
is needed because this method has no learned policy.

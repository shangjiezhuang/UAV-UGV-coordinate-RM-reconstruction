# Shared Runtime

This package owns algorithm-neutral configuration, physical environment,
spectrum data, reconstruction, evaluation, visualization, and multi-algorithm
training orchestration.

- `ppo_base.py`, `on_policy_networks.py`, and `joint_buffer.py` provide
  algorithm-neutral PPO primitives. MAPPO, HAPPO, IPPO, MAPPO-CF, and UAV-PPO
  build on these components without treating MAPPO as the shared layer.
- `train_runner.py` selects an algorithm and one of the two physical-runtime
  variants; `evaluate_checkpoint.py` provides the matching shared evaluator.
- `noquant` samples the floating-point source data without numerical
  quantization. Its 16-bit setting is only the uncompressed payload-width
  reference.
- `quant` applies exactly one policy-selected 10/8/6-bit quantization stage
  after source sampling.

Algorithm folders such as `MAPPO_*`, `IPPO_*`, `HAPPO`, and `PPO_AStar_*`
should contain only algorithm or controller implementations and thin
entrypoints. Legacy imports from shared modules under `MAPPO_*` are retained
as compatibility shims.

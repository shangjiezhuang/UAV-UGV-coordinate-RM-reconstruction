"""Small shared tensor helpers for non-standard on-policy trainers."""

from __future__ import annotations

from typing import Iterable

import torch


def minibatch_indices(
    total_size: int,
    num_minibatches: int,
    *,
    shuffle: bool = True,
) -> Iterable[torch.Tensor]:
    """Split every sample exactly once, optionally in randomized order."""
    total_size = int(total_size)
    num_minibatches = int(num_minibatches)
    if total_size <= 0:
        raise ValueError(f"total_size must be positive, got {total_size}")
    if not 1 <= num_minibatches <= total_size:
        raise ValueError(
            f"num_minibatches must be in [1, {total_size}], got {num_minibatches}"
        )
    indices = (
        torch.randperm(total_size)
        if shuffle
        else torch.arange(total_size, dtype=torch.long)
    )
    base_size = total_size // num_minibatches
    for index in range(num_minibatches):
        start = index * base_size
        stop = total_size if index == num_minibatches - 1 else start + base_size
        yield indices[start:stop]


def indexed_batch(flat: dict, indices: torch.Tensor, device: torch.device) -> dict:
    """Gather one CPU minibatch and move it to the trainer device once."""
    batch = {key: value.index_select(0, indices) for key, value in flat.items()}
    if device.type == "cpu":
        return batch
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Return PPO's clipped scalar value loss."""
    clipped_values = old_values + torch.clamp(
        new_values - old_values,
        -float(clip_epsilon),
        float(clip_epsilon),
    )
    return 0.5 * torch.max(
        (new_values - returns) ** 2,
        (clipped_values - returns) ** 2,
    ).mean()


__all__ = ["clipped_value_loss", "indexed_batch", "minibatch_indices"]

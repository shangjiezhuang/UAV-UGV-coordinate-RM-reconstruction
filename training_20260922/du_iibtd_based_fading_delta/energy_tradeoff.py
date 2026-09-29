"""Energy/NMSE/transmission reporting with episode-preserving pairs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


_PAIR_FIELDS = ("energy_budget_j", "nmse", "link_transmitted_bits")


def _validated_record(record: Mapping[str, Any], index: int) -> Dict[str, Any]:
    missing = [field for field in _PAIR_FIELDS if field not in record]
    if missing:
        raise ValueError(f"episode record {index} is missing required fields: {missing}")
    clean = dict(record)
    for field in _PAIR_FIELDS:
        value = float(clean[field])
        if not np.isfinite(value):
            raise ValueError(
                f"episode record {index} has non-finite {field}: {clean[field]!r}"
            )
        clean[field] = value
    if clean["energy_budget_j"] <= 0.0:
        raise ValueError(
            f"episode record {index} energy_budget_j must be positive, "
            f"got {clean['energy_budget_j']}"
        )
    return clean


def _pair(energy_j: float, nmse: float, link_bits: float) -> List[float]:
    """Return the requested (energy, NMSE, physical-link bits) triple."""
    return [float(energy_j), float(nmse), float(link_bits)]


def summarize_energy_episode_records(
    episode_records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Group raw episodes by energy and retain correctly paired extrema.

    The minimum- and maximum-NMSE transmission values are selected from the
    exact same episode as the respective NMSE.  The mean row is an aggregate
    triple and therefore contains both the mean NMSE and mean physical-link
    transmitted bits.
    """
    grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(episode_records):
        clean = _validated_record(record, index)
        grouped[clean["energy_budget_j"]].append(clean)
    if not grouped:
        raise ValueError("episode_records must not be empty")

    summaries: List[Dict[str, Any]] = []
    for energy_j in sorted(grouped):
        records = grouped[energy_j]
        nmse_values = np.asarray([record["nmse"] for record in records], dtype=float)
        link_values = np.asarray(
            [record["link_transmitted_bits"] for record in records],
            dtype=float,
        )
        min_index = int(np.argmin(nmse_values))
        max_index = int(np.argmax(nmse_values))
        min_record = records[min_index]
        max_record = records[max_index]
        mean_nmse = float(np.mean(nmse_values))
        mean_link_bits = float(np.mean(link_values))
        summaries.append(
            {
                "energy_budget_j": float(energy_j),
                "num_episodes": int(len(records)),
                "mean_nmse": mean_nmse,
                "mean_link_transmitted_bits": mean_link_bits,
                "min_nmse": float(min_record["nmse"]),
                "link_transmitted_bits_at_min_nmse": float(
                    min_record["link_transmitted_bits"]
                ),
                "max_nmse": float(max_record["nmse"]),
                "link_transmitted_bits_at_max_nmse": float(
                    max_record["link_transmitted_bits"]
                ),
                "mean_nmse_pair": _pair(energy_j, mean_nmse, mean_link_bits),
                "min_nmse_pair": _pair(
                    energy_j,
                    min_record["nmse"],
                    min_record["link_transmitted_bits"],
                ),
                "max_nmse_pair": _pair(
                    energy_j,
                    max_record["nmse"],
                    max_record["link_transmitted_bits"],
                ),
                "min_nmse_episode": dict(min_record),
                "max_nmse_episode": dict(max_record),
            }
        )
    return summaries


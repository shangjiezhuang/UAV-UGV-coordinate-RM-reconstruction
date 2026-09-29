"""Merge the clean-observation Q/NQ formal results and build one summary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil


PKG_ROOT = Path("/home/zsj/works/work1/code/du_iibtd_based_fading_delta")
ASTAR4_ROOT = PKG_ROOT / "formal_runs/astar4_quant_cleanobs_h160_uimp008_192k_s42"
REMAINING_ROOT = PKG_ROOT / "formal_runs/remaining16_qnq_cleanobs_h160_uimp008_192k_s42"
MERGED_ROOT = PKG_ROOT / "formal_runs/complete_qnq_cleanobs_h160_uimp008_192k_s42"
DEST = MERGED_ROOT / "frozen_test_seed200042_n3"

CURRENT_Q = {
    "ppo_astar_target_quant.json",
    "ppo_astar_support_quant.json",
    "ppo_astar_2path_support_quant.json",
    "ppo_astar_2path_support_recovery_lite_quant.json",
}
REMAINING_LEARNED = {
    "mappo_quant.json",
    "mappo_noquant.json",
    "ippo_quant.json",
    "ippo_noquant.json",
    "happo_quant.json",
    "happo_noquant.json",
    "mappo_cf_quant.json",
    "mappo_cf_noquant.json",
    "ppo_fixed_quant.json",
    "ppo_fixed_noquant.json",
    "ppo_heuristic_quant.json",
    "ppo_heuristic_noquant.json",
    "ppo_astar_target_noquant.json",
    "ppo_astar_support_noquant.json",
    "ppo_astar_2path_support_noquant.json",
    "ppo_astar_2path_support_recovery_lite_noquant.json",
}
BASELINES = {
    "total_random_quant.json",
    "total_random_noquant.json",
    "greedy_heuristic_quant.json",
    "greedy_heuristic_noquant.json",
    "greedy_astar_2path_support_quant.json",
    "greedy_astar_2path_support_noquant.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_group(source_root: Path, names: set[str], source_label: str):
    source_dir = source_root / "frozen_test_seed200042_n3"
    records = []
    for name in sorted(names):
        source = source_dir / name
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing formal result: {source}")
        destination = DEST / name
        shutil.copy2(source, destination)
        records.append(
            {
                "name": name,
                "source_batch": source_label,
                "source": str(source),
                "destination": str(destination),
                "sha256": sha256(destination),
            }
        )
    return records


def variant_from_name(name: str) -> str:
    return "noquant" if "noquant" in name else "quant"


def kind_from_name(name: str) -> str:
    return "baseline" if name in BASELINES else "learned"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    manifest.extend(copy_group(ASTAR4_ROOT, CURRENT_Q, "astar4_quant_cleanobs_h160"))
    manifest.extend(
        copy_group(
            REMAINING_ROOT,
            REMAINING_LEARNED | BASELINES,
            "remaining16_qnq_cleanobs_h160",
        )
    )
    if len(manifest) != 26:
        raise RuntimeError(f"Expected 26 merged results, got {len(manifest)}")

    rows = []
    for record in manifest:
        path = Path(record["destination"])
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        aggregate = result["aggregate"]
        rows.append(
            {
                "method": path.stem,
                "kind": kind_from_name(path.name),
                "variant": variant_from_name(path.name),
                "mean_nmse": aggregate["eval_mean_nmse"],
                "std_nmse": aggregate["eval_std_nmse"],
                "mean_outage_ratio": aggregate["eval_mean_outage_ratio"],
                "mean_service_completion_ratio": aggregate[
                    "eval_mean_service_completion_ratio"
                ],
                "min_scene_service_completion_ratio": aggregate[
                    "eval_min_scene_service_completion_ratio"
                ],
                "min_episode_service_completion_ratio": aggregate[
                    "eval_min_service_completion_ratio"
                ],
                "mean_return": aggregate["eval_mean_return"],
                "mean_final_queue_bits": aggregate["eval_mean_final_queue_bits"],
                "mean_dropped_bits": aggregate["eval_mean_dropped_bits"],
                "mean_link_transmitted_bits": aggregate[
                    "eval_mean_link_transmitted_bits"
                ],
                "energy_failure_rate": aggregate["eval_energy_failure_rate"],
                "num_total_episodes": aggregate["eval_num_total_episodes"],
            }
        )

    rows.sort(
        key=lambda row: (
            row["variant"],
            row["kind"],
            row["mean_nmse"],
            row["mean_outage_ratio"],
            -row["mean_service_completion_ratio"],
        )
    )
    with (MERGED_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": {
            "total_timesteps": 192000,
            "episode_max_steps": 160,
            "prefill_percent": 5,
            "prefill_budget_basis": 200,
            "seed": 42,
            "test_seed_base": 200042,
            "test_scenes": 8,
            "test_episodes_per_scene": 3,
            "uav_obs_dim": 14,
            "ugv_obs_dim": 14,
            "critic_state_dim": 19,
            "quant_uav_action_dim": 60,
            "noquant_uav_action_dim": 20,
            "ugv_action_dim": 5,
        },
        "result_count": len(manifest),
        "learned_result_count": sum(row["kind"] == "learned" for row in rows),
        "baseline_result_count": sum(row["kind"] == "baseline" for row in rows),
        "sources": manifest,
        "summary": rows,
    }
    with (MERGED_ROOT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    (MERGED_ROOT / "COMPLETE").touch()
    print(f"[MERGE COMPLETE] {MERGED_ROOT} ({len(manifest)} results)")


if __name__ == "__main__":
    main()

"""Compare selected full-length reward runs with frozen Reward-v1 baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


CASES = {
    "mappo_noquant_novel": ("noquant", "MAPPO", "novel_info"),
    "ppo_astar_noquant_repeat": ("noquant", "PPO-A*", "repeat_only"),
    "mappo_quant_novel": ("quant", "MAPPO", "novel_info"),
    "ppo_astar_quant_repeat": ("quant", "PPO-A*", "repeat_only"),
}


def _row(path: Path, variant: str, algorithm: str, reward: str) -> Dict[str, Any]:
    aggregate = json.loads(path.read_text(encoding="utf-8"))["aggregate"]

    def metric(key: str) -> float:
        return float(aggregate.get(key, float("nan")))

    return {
        "variant": variant,
        "algorithm": algorithm,
        "reward": reward,
        "result_path": str(path),
        "episodes": int(aggregate["eval_num_total_episodes"]),
        "mean_nmse": metric("eval_mean_nmse"),
        "best_nmse": metric("eval_best_nmse"),
        "worst_nmse": metric("eval_worst_nmse"),
        "novel_gbit": metric("eval_mean_novel_data_delivered_bits") * 1e-9,
        "link_gbit": metric("eval_mean_link_transmitted_bits") * 1e-9,
        "completion": metric("eval_mean_service_completion_ratio"),
        "outage": metric("eval_mean_outage_ratio"),
        "uav_stay": metric("eval_mean_uav_stay_ratio"),
        "full_repeat": metric("eval_mean_full_repeat_ratio"),
        "energy_failure": metric("eval_energy_failure_rate"),
        "full_horizon": metric("eval_full_horizon_rate"),
        "accounting_max_bits": metric("eval_max_data_accounting_violation_bits"),
    }


def load_rows(
    result_dir: Path,
    quant_baseline_dir: Path,
    noquant_baseline_dir: Path,
) -> List[Dict[str, Any]]:
    baseline_paths = {
        ("noquant", "MAPPO"): noquant_baseline_dir / "mappo_noquant_final.json",
        ("noquant", "PPO-A*"): noquant_baseline_dir / "ppo_astar_noquant_final.json",
        ("quant", "MAPPO"): quant_baseline_dir / "mappo_o3_final.json",
        ("quant", "PPO-A*"): quant_baseline_dir / "ppo_astar_o3_final.json",
    }
    rows: List[Dict[str, Any]] = []
    baselines: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key, path in baseline_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        baseline = _row(path, key[0], key[1], "Reward-v1")
        baseline["nmse_change_pct"] = 0.0
        baseline["novel_gbit_change"] = 0.0
        baselines[key] = baseline
        rows.append(baseline)

    for stem, (variant, algorithm, reward) in CASES.items():
        path = result_dir / f"{stem}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        selected = _row(path, variant, algorithm, reward)
        baseline = baselines[(variant, algorithm)]
        selected["nmse_change_pct"] = (
            100.0
            * (selected["mean_nmse"] - baseline["mean_nmse"])
            / baseline["mean_nmse"]
        )
        selected["novel_gbit_change"] = selected["novel_gbit"] - baseline["novel_gbit"]
        rows.append(selected)
    return rows


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Selected reward full retraining (192k transitions, seed 42)",
        "",
        "Frozen test: 8 scenes x 3 episodes, seed base 200042. NMSE/outage/stay are lower; completion and novel data are higher.",
        "",
        "| Variant | Algorithm | Reward | NMSE mean | NMSE min | NMSE max | dNMSE | Novel Gbit | dNovel | Completion | Outage | UAV stay | Full repeat |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reward_order = {"Reward-v1": 0, "novel_info": 1, "repeat_only": 1}
    for row in sorted(
        rows,
        key=lambda value: (
            value["variant"],
            value["algorithm"],
            reward_order[value["reward"]],
        ),
    ):
        repeat_text = (
            f"{row['full_repeat']:.3f}"
            if row["full_repeat"] == row["full_repeat"]
            else "n/a"
        )
        lines.append(
            f"| {row['variant']} | {row['algorithm']} | {row['reward']} | "
            f"{row['mean_nmse']:.6f} | {row['best_nmse']:.6f} | "
            f"{row['worst_nmse']:.6f} | {row['nmse_change_pct']:+.2f}% | "
            f"{row['novel_gbit']:.3f} | {row['novel_gbit_change']:+.3f} | "
            f"{row['completion']:.3f} | {row['outage']:.3f} | "
            f"{row['uav_stay']:.3f} | {repeat_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--quant-baseline-dir", type=Path, required=True)
    parser.add_argument("--noquant-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(
        result_dir=args.result_dir,
        quant_baseline_dir=args.quant_baseline_dir,
        noquant_baseline_dir=args.noquant_baseline_dir,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_prefix.with_suffix(".md").write_text(
        render_markdown(rows),
        encoding="utf-8",
    )
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()

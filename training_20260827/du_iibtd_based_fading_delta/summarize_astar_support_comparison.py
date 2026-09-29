"""Summarize the frozen three-way PPO+A* support comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


CASES: Tuple[Tuple[str, str, str], ...] = (
    (
        "ppo_astar_2path_support",
        "2PathSupport-A*",
        "astar_2path_support",
    ),
    ("ppo_astar_target", "TargetOnly-A*", "astar_target"),
    ("ppo_astar_support", "CurrentSupport-A*", "astar_support"),
)


def _load_case(result_dir: Path, stem: str, variant: str) -> Dict[str, object]:
    path = result_dir / f"{stem}_{variant}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["evaluation_config"]
    aggregate = payload["aggregate"]
    if int(config["ugv_comm_local_path_horizon"]) != 15:
        raise ValueError(f"{path} does not use local horizon 15")
    if int(config["ugv_comm_expanded_path_horizon"]) != 20:
        raise ValueError(f"{path} does not use expanded horizon 20")
    if abs(float(config["lambda_novel_info"])) > 1e-12:
        raise ValueError(f"{path} does not use lambda_novel_info=0")
    if abs(float(config["lambda_full_repeat"]) - 0.05) > 1e-12:
        raise ValueError(f"{path} does not use lambda_full_repeat=0.05")
    return {
        "path": str(path),
        "evaluation_config": config,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("quant", "noquant"), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    raw_cases = {}
    for stem, label, expected_mode in CASES:
        case = _load_case(args.result_dir, stem, args.variant)
        config = case["evaluation_config"]
        mode = str(config["ugv_control_mode"])
        if mode == "astar" and expected_mode == "astar_target":
            mode = "astar_target"
        if mode != expected_mode:
            raise ValueError(
                f"{case['path']} has ugv_control_mode={mode!r}, "
                f"expected {expected_mode!r}"
            )
        aggregate = case["aggregate"]
        row = {
            "method": label,
            "trainer": str(config["trainer"]),
            "ugv_control_mode": mode,
            "mean_nmse": float(aggregate["eval_mean_nmse"]),
            "std_nmse": float(aggregate["eval_std_nmse"]),
            "mean_return": float(aggregate["eval_mean_return"]),
            "mean_link_transmitted_bits": float(
                aggregate["eval_mean_link_transmitted_bits"]
            ),
            "mean_data_delivered_bits": float(
                aggregate["eval_mean_data_delivered_bits"]
            ),
            "mean_outage_ratio": float(aggregate["eval_mean_outage_ratio"]),
            "mean_service_completion_ratio": float(
                aggregate["eval_mean_service_completion_ratio"]
            ),
            "mean_full_repeat_ratio": float(
                aggregate["eval_mean_full_repeat_ratio"]
            ),
            "mean_ugv_move_dist": float(aggregate["eval_mean_ugv_move_dist"]),
        }
        rows.append(row)
        raw_cases[stem] = case

    output = {
        "variant": args.variant,
        "comparison_contract": {
            "local_horizon": 15,
            "expanded_horizon": 20,
            "lambda_novel_info": 0.0,
            "lambda_full_repeat": 0.05,
        },
        "rows": rows,
        "cases": raw_cases,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        f"# Three-way PPO+A* support comparison ({args.variant})",
        "",
        "All methods use UGV local/expanded horizons 15/20 and the same "
        "full-repeat penalty (`lambda_full_repeat=0.05`, "
        "`lambda_novel_info=0`).",
        "",
        "| Method | NMSE ↓ | Return ↑ | Link Gbit ↑ | Delivered Gbit ↑ | "
        "Outage ↓ | Service ↑ | Full repeat ↓ | UGV move |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {mean_nmse:.6f} ± {std_nmse:.6f} | "
            "{mean_return:.3f} | {link:.3f} | {delivered:.3f} | "
            "{mean_outage_ratio:.3f} | {mean_service_completion_ratio:.3f} | "
            "{mean_full_repeat_ratio:.3f} | {mean_ugv_move_dist:.3f} |".format(
                link=row["mean_link_transmitted_bits"] / 1e9,
                delivered=row["mean_data_delivered_bits"] / 1e9,
                **row,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()

"""Summarize matched frozen-test results for quant and noquant variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "eval_mean_nmse",
    "eval_mean_return",
    "eval_mean_link_transmitted_bits",
    "eval_mean_service_completion_ratio",
    "eval_mean_outage_ratio",
    "eval_mean_uav_move_dist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--noquant-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def method_name(payload: dict[str, Any]) -> str:
    config = payload["evaluation_config"]
    return str(config.get("trainer", config.get("controller")))


def load_results(directory: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "evaluation_config" not in payload or "aggregate" not in payload:
            continue
        aggregate = payload["aggregate"]
        results[method_name(payload)] = {
            metric: float(aggregate[metric]) for metric in METRICS
        }
    return results


def main() -> None:
    args = parse_args()
    quant = load_results(args.quant_dir)
    noquant = load_results(args.noquant_dir)
    methods = sorted(set(quant) & set(noquant))
    if not methods:
        raise ValueError("No matched quant/noquant methods were found")

    comparisons = []
    for method in methods:
        comparisons.append(
            {
                "method": method,
                "quant": quant[method],
                "noquant": noquant[method],
                "noquant_minus_quant": {
                    metric: noquant[method][metric] - quant[method][metric]
                    for metric in METRICS
                },
            }
        )

    payload = {
        "quant_dir": str(args.quant_dir),
        "noquant_dir": str(args.noquant_dir),
        "matched_methods": methods,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Quant vs Noquant Frozen-Test Summary",
        "",
        "All rows use the matched frozen protocol. Delta is noquant minus quant.",
        "",
        "| Method | Quant NMSE | Noquant NMSE | Δ NMSE | Quant link Gbit | Noquant link Gbit | Δ link Gbit | Quant completion | Noquant completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        q = item["quant"]
        n = item["noquant"]
        delta = item["noquant_minus_quant"]
        lines.append(
            f"| {item['method']} | {q['eval_mean_nmse']:.6f} | "
            f"{n['eval_mean_nmse']:.6f} | {delta['eval_mean_nmse']:+.6f} | "
            f"{q['eval_mean_link_transmitted_bits'] * 1e-9:.3f} | "
            f"{n['eval_mean_link_transmitted_bits'] * 1e-9:.3f} | "
            f"{delta['eval_mean_link_transmitted_bits'] * 1e-9:+.3f} | "
            f"{q['eval_mean_service_completion_ratio']:.3f} | "
            f"{n['eval_mean_service_completion_ratio']:.3f} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_markdown}")


if __name__ == "__main__":
    main()

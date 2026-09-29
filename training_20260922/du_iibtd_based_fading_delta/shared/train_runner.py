"""Algorithm-neutral quant/noquant launcher used by strategy entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _has_option(argv, option: str) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def run_training_entrypoint(
    trainer: str,
    source_dir: Path,
    argv=None,
    *,
    forced_variant: str | None = None,
) -> None:
    if forced_variant not in (None, "quant", "noquant"):
        raise ValueError(f"Unsupported forced variant: {forced_variant!r}")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--variant",
        choices=("quant", "noquant"),
        default=forced_variant or "quant",
    )
    known, remaining = pre_parser.parse_known_args(raw_args)
    if forced_variant is not None and known.variant != forced_variant:
        pre_parser.error(
            f"this entrypoint is fixed to --variant {forced_variant}, "
            f"not {known.variant}"
        )
    variant = forced_variant or known.variant
    source_dir = Path(source_dir).resolve()
    if not _has_option(remaining, "--model_dir"):
        remaining.extend(["--model_dir", str(source_dir / "checkpoints" / variant)])
    if not _has_option(remaining, "--log_dir"):
        remaining.extend(["--log_dir", str(source_dir / "logs" / variant)])

    if variant == "quant":
        from du_iibtd_based_fading_delta.shared.quant.runner import main as unified_main
    else:
        from du_iibtd_based_fading_delta.shared.noquant.runner import main as unified_main
    unified_main(remaining, forced_trainer=trainer)


run_algorithm_entrypoint = run_training_entrypoint


__all__ = ["run_algorithm_entrypoint", "run_training_entrypoint"]

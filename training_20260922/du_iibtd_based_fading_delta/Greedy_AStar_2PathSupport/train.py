"""Evaluate Greedy UAV + A* 2Path Support UGV in either environment variant."""

from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv=None) -> None:
    from du_iibtd_based_fading_delta.evaluate_fair_baseline import main as evaluate_main

    raw_args = list(sys.argv[1:] if argv is None else argv)
    evaluate_main(
        [
            *raw_args,
            "--controller",
            "greedy_astar_2path_support",
        ]
    )


if __name__ == "__main__":
    main()

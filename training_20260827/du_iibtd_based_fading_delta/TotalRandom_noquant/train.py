"""Evaluate noquant total-random baseline in the shared based executor."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.evaluate_fair_baseline import main as _evaluate_main


def main() -> None:
    _evaluate_main(
        [*sys.argv[1:], "--variant", "noquant", "--controller", "total_random"]
    )


if __name__ == "__main__":
    main()

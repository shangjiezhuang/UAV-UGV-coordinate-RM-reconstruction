"""Compatibility entrypoint for the algorithm-neutral checkpoint evaluator."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.shared import evaluate_checkpoint as _shared
from du_iibtd_based_fading_delta.shared.evaluate_checkpoint import *  # noqa: F401,F403

_optional_config_bool = _shared._optional_config_bool


def __getattr__(name):
    return getattr(_shared, name)


if __name__ == "__main__":
    _shared.main()


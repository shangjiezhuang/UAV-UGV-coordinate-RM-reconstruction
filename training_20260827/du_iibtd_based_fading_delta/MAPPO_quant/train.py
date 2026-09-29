"""MAPPO quant entrypoint with legacy shared-runner exports."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from du_iibtd_based_fading_delta.shared.quant.runner import *  # noqa: F401,F403
from du_iibtd_based_fading_delta.shared.quant.runner import main as _shared_main


def main(argv=None, forced_trainer=None) -> None:
    _shared_main(
        argv,
        forced_trainer="mappo" if forced_trainer is None else forced_trainer,
    )


if __name__ == "__main__":
    main()

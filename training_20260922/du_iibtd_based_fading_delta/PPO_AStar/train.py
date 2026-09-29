"""Train PPO+A* in either based environment variant."""

from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv=None) -> None:
    from du_iibtd_based_fading_delta.shared.train_runner import run_training_entrypoint

    run_training_entrypoint("ppo_astar", Path(__file__).parent, argv)


if __name__ == "__main__":
    main()

"""Compatibility wrapper for the shared MAPPO_noquant.buffer.RolloutBuffer implementation."""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED_DIR = Path(__file__).resolve().parents[1]
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from MAPPO_noquant.buffer import RolloutBuffer

__all__ = ["RolloutBuffer"]

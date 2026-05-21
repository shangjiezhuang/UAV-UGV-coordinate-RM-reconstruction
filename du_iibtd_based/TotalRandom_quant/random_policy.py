"""Compatibility wrapper for the shared TotalRandom_noquant.random_policy.RandomPolicy implementation."""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED_DIR = Path(__file__).resolve().parents[1]
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from TotalRandom_noquant.random_policy import RandomPolicy

__all__ = ["RandomPolicy"]

"""Small process-level path setup shared by executable entrypoints."""

from __future__ import annotations

import os
import tempfile


def configure_matplotlib_cache_dir() -> None:
    """Use a writable Matplotlib cache without changing an explicit setting."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    default_cache_dir = os.path.join(os.path.expanduser("~"), ".config", "matplotlib")
    if os.path.isdir(default_cache_dir) and os.access(default_cache_dir, os.W_OK):
        return
    uid = getattr(os, "getuid", lambda: "user")()
    fallback_dir = os.path.join(tempfile.gettempdir(), f"matplotlib-{uid}")
    os.makedirs(fallback_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = fallback_dir


__all__ = ["configure_matplotlib_cache_dir"]

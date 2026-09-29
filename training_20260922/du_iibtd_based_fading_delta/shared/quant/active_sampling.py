"""Quant compatibility view of the shared active-sampling stack."""

from __future__ import annotations

from du_iibtd_based_fading_delta.shared.noquant import active_sampling as _module

__all__ = [name for name in dir(_module) if not name.startswith("_")]
globals().update({name: getattr(_module, name) for name in __all__})

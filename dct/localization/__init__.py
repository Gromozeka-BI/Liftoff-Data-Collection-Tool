"""Stick-based online localization (vendored numpy-only implementation).

Public API:
    OnlineLocalizer, Reference, LocalizerResult — runtime particle filter
    Lap, load_dct_session, load_dct_sessions_dir — pandas-backed lap loaders
    reference_builder — high-level build/save facade keyed by track_id
"""
from dct.localization.online_localizer import (
    LocalizerResult,
    OnlineLocalizer,
    Reference,
)

__all__ = [
    "LocalizerResult",
    "OnlineLocalizer",
    "Reference",
]

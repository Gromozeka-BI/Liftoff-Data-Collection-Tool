"""Stick-based online localization (vendored numpy-only implementation).

Public API:
    OnlineLocalizer, Reference, LocalizerResult — runtime particle filter
    KFLayer2 — KF second layer (RC PF smoother with speed profile)
    Lap, load_dct_session, load_dct_sessions_dir — pandas-backed lap loaders
    reference_builder — high-level build/save facade keyed by track_id
"""
from dct.localization.online_localizer import (
    LocalizerResult,
    OnlineLocalizer,
    Reference,
)
from dct.localization.kf_layer2 import KFLayer2

__all__ = [
    "LocalizerResult",
    "OnlineLocalizer",
    "Reference",
    "KFLayer2",
]

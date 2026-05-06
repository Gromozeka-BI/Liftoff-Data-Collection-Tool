"""Snap-policy helper used by EventStrip drag and event add."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt6.QtCore import Qt

# Default policy by event type
_DEFAULT_TARGET = {
    "rh_lap":      "telemetry",
    "rh_gate":     "telemetry",
    "button_lap":  "playhead",
    "button_gate": "playhead",
    "session_start": "playhead",
    "session_stop":  "playhead",
}


def _nearest(arr, ts):
    if arr is None or len(arr) == 0:
        return ts
    arr = np.asarray(arr)
    idx = int(np.searchsorted(arr, ts))
    if idx >= len(arr):
        return float(arr[-1])
    if idx == 0:
        return float(arr[0])
    a = float(arr[idx - 1])
    b = float(arr[idx])
    return a if (ts - a) <= (b - ts) else b


def snap_ts(
    ts: float,
    *,
    event_type: str,
    modifiers: Qt.KeyboardModifier,
    tl_ts: Sequence[float] | None = None,
    rc_ts: Sequence[float] | None = None,
    playhead_ts: float | None = None,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> float:
    """Return ``ts`` snapped according to event type and active modifiers.

    Modifier overrides:
        Shift -> playhead
        Ctrl  -> telemetry sample
        Alt   -> RC sample
    """
    target = _DEFAULT_TARGET.get(event_type, "playhead")

    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        target = "playhead"
    elif modifiers & Qt.KeyboardModifier.ControlModifier:
        target = "telemetry"
    elif modifiers & Qt.KeyboardModifier.AltModifier:
        target = "rc"

    if target == "playhead" and playhead_ts is not None:
        snapped = float(playhead_ts)
    elif target == "telemetry":
        snapped = _nearest(tl_ts, ts)
    elif target == "rc":
        snapped = _nearest(rc_ts, ts)
    else:
        snapped = ts

    if clamp_min is not None:
        snapped = max(clamp_min, snapped)
    if clamp_max is not None:
        snapped = min(clamp_max, snapped)
    return snapped

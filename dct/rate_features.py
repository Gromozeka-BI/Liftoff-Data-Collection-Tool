"""Unified observation features for stick-based localization (Reference + online).

Variant A (``betaflight_classic_rpy_deg_s_v1``):
  * Throttle: already in [-1, 1], clipped.
  * Roll / pitch / yaw: Betaflight **classic** (non-Actual) angle-rate curve → deg/s,
    same formula as Betaflight ``applyBetaflightRates`` (rc.c).

Channel order everywhere: **[throttle, yaw, pitch, roll]** — same as
``lap_loader.STICK_COLS`` / Liftoff ``in_*`` fields.

Merging RC with UDP on one timeline (duplicate timestamps)
-------------------------------------------------------------
When two samples share the same ``ts_wall`` (or fall into the same bin after
resampling), you must pick a **tie-break rule** or you get undefined order.
Common choices:

1. **Last-wins** — keep overwriting; last sample at that time is used (simple,
   good if the stream is strictly chronological per source).
2. **Average** — merge duplicates (smooths noise, blurs sharp edges).
3. **Priority** — e.g. always prefer UDP over interpolated RC when timestamps
   collide.

Interpolation itself does not create duplicates unless the target grid already
has a point at that time; then the rule above applies when *writing* merged rows.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

FEATURE_BETAFLIGHT_CLASSIC_V1 = "betaflight_classic_rpy_deg_s_v1"


def _expo_fraction(expo: float) -> float:
    """Configurator may store expo as 0–1 or 0–100 (BF rcExpo)."""
    e = float(expo)
    return e / 100.0 if e > 1.0 else e


def _super_fraction(super_rate: float) -> float:
    """Super rate: BF stores integer percent; JSON may use 0–1 directly."""
    s = float(super_rate)
    return s / 100.0 if s > 1.0 else s


def _rc_rate_scaled(rc_rate: float) -> float:
    """Match Betaflight: ``rcRate = rcRates[axis] / 100`` then Actual-rate branch."""
    r = float(rc_rate)
    if r > 2.0:
        r += 14.54 * (r - 2.0)
    return r


def apply_betaflight_classic_axis(rc_commandf: float, rc_rate: float, super_rate: float, expo: float) -> float:
    """One axis: stick -1..1 → angle rate deg/s (Betaflight classic)."""
    rc = float(np.clip(rc_commandf, -1.0, 1.0))
    rc_abs = abs(rc)
    expof = _expo_fraction(expo)
    if expof > 0.0:
        rc = rc * (rc_abs * rc_abs) * expof + rc * (1.0 - expof)

    r = _rc_rate_scaled(rc_rate)
    angle_rate = 200.0 * r * rc
    s = _super_fraction(super_rate)
    if s > 0.0:
        denom = float(np.clip(1.0 - abs(rc) * s, 0.01, 1.0))
        angle_rate /= denom
    return float(angle_rate)


def apply_betaflight_classic_axis_vec(
    rc: np.ndarray,
    rc_rate: float,
    super_rate: float,
    expo: float,
) -> np.ndarray:
    """Vectorized classic curve; ``rc`` shape (N,)."""
    x = np.clip(rc.astype(np.float64), -1.0, 1.0)
    a = np.abs(x)
    expof = _expo_fraction(expo)
    if expof > 0.0:
        x = x * (a * a) * expof + x * (1.0 - expof)
    a = np.abs(x)
    r = _rc_rate_scaled(rc_rate)
    out = 200.0 * r * x
    s = _super_fraction(super_rate)
    if s > 0.0:
        denom = np.clip(1.0 - a * s, 0.01, 1.0)
        out = out / denom
    return out


def _axis_params(profile: dict[str, Any], axis_key: str) -> tuple[float, float, float]:
    ax = profile.get(axis_key)
    if not isinstance(ax, dict):
        return 1.0, 0.0, 0.0
    return (
        float(ax.get("rc_rate", 1.0)),
        float(ax.get("rate", 0.0)),
        float(ax.get("expo", 0.0)),
    )


def physical_observation_matrix(sticks: np.ndarray, rate_profile: dict[str, Any]) -> np.ndarray:
    """Map (N,4) sticks [thr,yaw,pitch,roll] in -1..1 → (N,4) physical observations.

    Column 0: throttle (clipped). Columns 1–3: yaw / pitch / roll rate in deg/s.
    """
    if rate_profile.get("model") != "betaflight":
        raise ValueError(f"Unsupported rate model: {rate_profile.get('model')!r}")

    sticks = np.atleast_2d(sticks).astype(np.float64)
    if sticks.shape[1] != 4:
        raise ValueError(f"sticks must be (N,4), got {sticks.shape}")

    n = sticks.shape[0]
    out = np.zeros((n, 4), dtype=np.float64)
    out[:, 0] = np.clip(sticks[:, 0], -1.0, 1.0)

    axes_keys = ("yaw", "pitch", "roll")
    for col, key in enumerate(axes_keys, start=1):
        rc_rate, super_rate, expo = _axis_params(rate_profile, key)
        out[:, col] = apply_betaflight_classic_axis_vec(sticks[:, col], rc_rate, super_rate, expo)

    return out.astype(np.float32)


def physical_observation_row(sticks_row: np.ndarray | list[float], rate_profile: dict[str, Any]) -> np.ndarray:
    """Single row (4,) same convention as :func:`physical_observation_matrix`."""
    m = physical_observation_matrix(np.asarray(sticks_row, dtype=np.float64).reshape(1, 4), rate_profile)
    return m[0]


def load_rate_profile(path: str | Any) -> dict[str, Any]:
    """Load ``rate_profile.json`` from a session directory or file path."""
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        p = p / "rate_profile.json"
    text = p.read_text(encoding="utf-8")
    return json.loads(text)

"""
Run OnlineLocalizer on a recorded session using two stick sources:
  1. rc_channels.parquet  — RC UART PWM channels
  2. telemetry.parquet    — Liftoff simulator in_* sticks

Both sources are processed through the Betaflight rate curve (rate_profile.json)
before being compared against the reference, so the comparison is in physical
units (deg/s) and is invariant to rate settings.

Output schema (localization_result.parquet):
  ts_wall                — wall-clock timestamp (s)
  pos_x / pos_y / pos_z  — ground truth from Liftoff telemetry (m)
  in_pos_x/y/z, in_q     — estimated position + uncertainty from in_* sticks (m)
  rc_pos_x/y/z, rc_q     — estimated position + uncertainty from RC channels (m)

Also saves trajectory_comparison_lap1.png — 2D top-down comparison for lap 1.

Usage:
    python tools/run_localization_result.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SESSION_DIR = Path(
    r"D:\DroneTrackerDB\Liftoff\Part_5_Эксперементальный"
    r"\2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001"
)
REF_NPZ = PROJECT_ROOT / "tracks" / "track-002" / "references" / "GromFF_1.npz"

from dct.localization import OnlineLocalizer  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
rc  = pd.read_parquet(SESSION_DIR / "rc_channels.parquet")
tel = pd.read_parquet(SESSION_DIR / "telemetry.parquet")
ev  = pd.read_parquet(SESSION_DIR / "events.parquet")

with open(SESSION_DIR / "invert.json") as f:
    invert = json.load(f)

with open(SESSION_DIR / "rate_profile.json") as f:
    rate_profile = json.load(f)

print(f"rc_channels : {len(rc):>6} rows")
print(f"telemetry   : {len(tel):>6} rows")
print(f"Rate profile: {rate_profile['name']}")
print(f"Reference   : {REF_NPZ.name}")

# ---------------------------------------------------------------------------
# Build stick arrays
# ---------------------------------------------------------------------------
# RC channels: PWM → -1..1
#   Order: [throttle=ch3, yaw=ch4, pitch=ch2, roll=ch1]
_RC_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_INVERT_KEYS = ["ch3", "ch4", "ch2", "ch1"]
rc_sticks = np.column_stack([
    (rc[ch].values.astype(float) - 1500.0) / 500.0
    for ch in _RC_ORDER
])
rc_invert = np.array([
    -1.0 if invert["rc"].get(k) else 1.0
    for k in _RC_INVERT_KEYS
])
rc_sticks *= rc_invert
rc_ts = rc["ts_wall"].values

# Liftoff in_* sticks (already -1..1)
#   Order: [in_throttle, in_yaw, in_pitch, in_roll]
_LF_COLS = ["in_throttle", "in_yaw", "in_pitch", "in_roll"]
_LF_INVERT_KEYS = ["in_throttle", "in_yaw", "in_pitch", "in_roll"]
lf_sticks = np.column_stack([tel[c].values for c in _LF_COLS])
lf_invert = np.array([
    -1.0 if invert["lf"].get(k) else 1.0
    for k in _LF_INVERT_KEYS
])
lf_sticks *= lf_invert
lf_ts = tel["ts_wall"].values

# ---------------------------------------------------------------------------
# Run localizer — RC channels
# ---------------------------------------------------------------------------
print("\nRunning localizer on RC channels...")
loc_rc = OnlineLocalizer.from_file(REF_NPZ)
rows_rc: list[dict] = []
prev_ts = None
for i in range(len(rc_sticks)):
    dt = float(rc_ts[i] - prev_ts) if prev_ts is not None else None
    prev_ts = rc_ts[i]
    r = loc_rc.update(rc_sticks[i], dt=dt, rate_profile=rate_profile)
    rows_rc.append({
        "ts_wall":      rc_ts[i],
        "rc_pos_x":     float(r.position_xyz[0]),
        "rc_pos_y":     float(r.position_xyz[1]),
        "rc_pos_z":     float(r.position_xyz[2]),
        "rc_q":         float(r.uncertainty_m),
    })
df_rc = pd.DataFrame(rows_rc)
print(f"  Done. {len(df_rc)} estimates.")

# ---------------------------------------------------------------------------
# Run localizer — Liftoff in_* sticks
# ---------------------------------------------------------------------------
print("Running localizer on Liftoff in_* sticks...")
loc_lf = OnlineLocalizer.from_file(REF_NPZ)
rows_lf: list[dict] = []
prev_ts = None
for i in range(len(lf_sticks)):
    dt = float(lf_ts[i] - prev_ts) if prev_ts is not None else None
    prev_ts = lf_ts[i]
    r = loc_lf.update(lf_sticks[i], dt=dt, rate_profile=rate_profile)
    rows_lf.append({
        "ts_wall":      lf_ts[i],
        "in_pos_x":     float(r.position_xyz[0]),
        "in_pos_y":     float(r.position_xyz[1]),
        "in_pos_z":     float(r.position_xyz[2]),
        "in_q":         float(r.uncertainty_m),
    })
df_lf = pd.DataFrame(rows_lf)
print(f"  Done. {len(df_lf)} estimates.")

# ---------------------------------------------------------------------------
# Merge and save parquet
# ---------------------------------------------------------------------------
# Ground truth timeline is from telemetry (pos_x/y/z + ts_wall)
df_truth = tel[["ts_wall", "pos_x", "pos_y", "pos_z"]].copy()

# Start from the denser source (telemetry), join RC estimates by nearest ts_wall
df_out = pd.merge_asof(
    df_truth.sort_values("ts_wall"),
    df_lf.sort_values("ts_wall"),
    on="ts_wall", direction="nearest",
)
df_out = pd.merge_asof(
    df_out.sort_values("ts_wall"),
    df_rc.sort_values("ts_wall"),
    on="ts_wall", direction="nearest",
)

# Final column order matches requested schema
df_out = df_out[[
    "ts_wall",
    "pos_x",    "pos_y",    "pos_z",
    "in_q",     "in_pos_x", "in_pos_y", "in_pos_z",
    "rc_q",     "rc_pos_x", "rc_pos_y", "rc_pos_z",
]]

out_path = SESSION_DIR / "localization_result.parquet"
df_out.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Rows : {len(df_out)}")
print(df_out.head(3).to_string())

# ---------------------------------------------------------------------------
# Lap 1 boundaries
# ---------------------------------------------------------------------------
lap_ev = (
    ev[ev["event_type"] == "rh_lap"]
    .sort_values("ts_wall")
    .reset_index(drop=True)
)
lap1_start = float(lap_ev.iloc[0]["ts_wall"])
lap1_end   = float(lap_ev.iloc[1]["ts_wall"])
print(f"\nLap 1: {lap1_start:.3f} -> {lap1_end:.3f}  "
      f"({lap1_end - lap1_start:.1f} s)")

mask_truth = (df_out["ts_wall"] >= lap1_start) & (df_out["ts_wall"] < lap1_end)
lap1 = df_out[mask_truth]

# ---------------------------------------------------------------------------
# Plot — 2D top-down, lap 1
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 9))

ax.plot(lap1["pos_x"],    lap1["pos_z"],
        color="black",       lw=2.0,               label="Ground truth (Liftoff pos)")
ax.plot(lap1["in_pos_x"], lap1["in_pos_z"],
        color="darkorange",  lw=1.3, alpha=0.85,   label="Estimated — telemetry in_* sticks")
ax.plot(lap1["rc_pos_x"], lap1["rc_pos_z"],
        color="steelblue",   lw=1.3, alpha=0.85,   label="Estimated — rc_channels")

# lap start marker
ax.scatter(
    [float(lap1["pos_x"].iloc[0])],
    [float(lap1["pos_z"].iloc[0])],
    color="green", s=100, zorder=5, label="Lap start",
)

ax.set_aspect("equal")
ax.set_xlabel("X, m")
ax.set_ylabel("Z, m")
ax.set_title("Lap 1 — trajectory comparison (top-down, XZ plane)\n"
             f"Reference: {REF_NPZ.name}  |  Rate: {rate_profile['name']}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = SESSION_DIR / "trajectory_comparison_lap1.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close()

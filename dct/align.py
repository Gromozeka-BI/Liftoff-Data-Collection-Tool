"""Telemetry / video / events alignment utilities.

All three parquet files share ts_wall (unix time, float64) as the common clock:

  telemetry.parquet        — one row per UDP packet (~100 Hz)
  events.parquet           — one row per discrete event (lap, gate, button)
  video_timestamps.parquet — one row per video frame (~30 Hz)

Alignment is nearest-neighbour on ts_wall.  Maximum error at 100 Hz telem = 5 ms.

Quick-start
-----------
    from dct.align import load_aligned, event_frames, lap_telemetry

    # Every video frame with its matching telemetry row
    frames = load_aligned("sessions/...")

    # Video frame indices closest to each lap event
    laps = event_frames("sessions/...", event_type="rh_lap")

    # Telemetry rows in a +-window around each lap
    for lap in lap_telemetry("sessions/...", window_s=2.0):
        print(lap[0]["pos_x"])   # position at moment of lap crossing
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _nearest(haystack_ts: np.ndarray, needle: float) -> int:
    return int(np.argmin(np.abs(haystack_ts - needle)))


# ---------------------------------------------------------------------------
# Core: video frame -> telemetry
# ---------------------------------------------------------------------------

def load_aligned(session_dir: str | Path) -> list[dict[str, Any]]:
    """One record per video frame with nearest telemetry row merged in.

    Extra fields on each record:
      frame_idx      — 0-based frame number in video.mp4
      ts_wall_frame  — exact wall clock at capture (from video_timestamps.parquet)
      dt_ms          — |ts_wall_frame - ts_wall_telem| — alignment error in ms
    """
    p = Path(session_dir)
    vt = pq.read_table(p / "video_timestamps.parquet")
    tel = pq.read_table(p / "telemetry.parquet")

    frame_ts  = np.array(vt.column("ts_wall").to_pylist())
    frame_idx = vt.column("frame_idx").to_pylist()
    telem_ts  = np.array(tel.column("ts_wall").to_pylist())
    telem_rows = tel.to_pylist()

    result: list[dict[str, Any]] = []
    for idx, fts in zip(frame_idx, frame_ts):
        ni = _nearest(telem_ts, fts)
        rec = dict(telem_rows[ni])
        rec["frame_idx"]     = idx
        rec["ts_wall_frame"] = float(fts)
        rec["dt_ms"]         = round(abs(fts - telem_ts[ni]) * 1000.0, 3)
        result.append(rec)
    return result


def alignment_stats(session_dir: str | Path) -> dict[str, float]:
    """Summary of alignment quality across all frames."""
    records = load_aligned(session_dir)
    dts = np.array([r["dt_ms"] for r in records])
    return {
        "frames":        len(dts),
        "dt_mean_ms":    round(float(np.mean(dts)), 3),
        "dt_median_ms":  round(float(np.median(dts)), 3),
        "dt_p95_ms":     round(float(np.percentile(dts, 95)), 3),
        "dt_max_ms":     round(float(np.max(dts)), 3),
    }


# ---------------------------------------------------------------------------
# Events -> video frames
# ---------------------------------------------------------------------------

def event_frames(
    session_dir: str | Path,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """For each event return the nearest video frame index and its telemetry.

    Returns list of dicts with keys:
      event_type, gate_id, lap_num, ts_wall_event,
      frame_idx, dt_frame_ms,
      pos_x, pos_y, pos_z, att_x, att_y, att_z, att_w  (telemetry at event)
    """
    p = Path(session_dir)
    ev_table = pq.read_table(p / "events.parquet")
    vt = pq.read_table(p / "video_timestamps.parquet")
    tel = pq.read_table(p / "telemetry.parquet")

    ev_rows = ev_table.to_pylist()
    frame_ts  = np.array(vt.column("ts_wall").to_pylist())
    frame_idx = vt.column("frame_idx").to_pylist()
    telem_ts  = np.array(tel.column("ts_wall").to_pylist())
    telem_rows = tel.to_pylist()

    result = []
    for ev in ev_rows:
        if event_type and ev.get("event_type") != event_type:
            continue
        ets = ev["ts_wall"]

        fi = _nearest(frame_ts, ets)
        ti = _nearest(telem_ts, ets)

        trow = telem_rows[ti]
        result.append({
            "event_type":   ev["event_type"],
            "gate_id":      ev.get("gate_id"),
            "lap_num":      ev.get("lap_num"),
            "ts_wall_event": ets,
            "frame_idx":    frame_idx[fi],
            "dt_frame_ms":  round(abs(ets - frame_ts[fi]) * 1000.0, 3),
            "pos_x": trow["pos_x"], "pos_y": trow["pos_y"], "pos_z": trow["pos_z"],
            "att_x": trow["att_x"], "att_y": trow["att_y"],
            "att_z": trow["att_z"], "att_w": trow["att_w"],
        })
    return result


# ---------------------------------------------------------------------------
# Events -> telemetry window
# ---------------------------------------------------------------------------

def lap_telemetry(
    session_dir: str | Path,
    window_s: float = 2.0,
    event_type: str = "rh_lap",
) -> list[list[dict[str, Any]]]:
    """Return telemetry rows in a +/- window_s around each lap event.

    Returns a list (one per lap) of lists of telemetry row dicts.
    """
    p = Path(session_dir)
    ev_table = pq.read_table(p / "events.parquet")
    tel = pq.read_table(p / "telemetry.parquet")

    telem_ts   = np.array(tel.column("ts_wall").to_pylist())
    telem_rows = tel.to_pylist()

    result = []
    for ev in ev_table.to_pylist():
        if ev.get("event_type") != event_type:
            continue
        ets = ev["ts_wall"]
        mask = np.abs(telem_ts - ets) <= window_s
        result.append([telem_rows[i] for i in np.where(mask)[0]])
    return result

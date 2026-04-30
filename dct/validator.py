"""Post-session validation against ТЗ FR-14/FR-15 criteria."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np


@dataclass
class ValidationResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def warn(self, msg: str) -> None:
        self.issues.append(f"WARN: {msg}")


def validate_session(session_dir: Path) -> ValidationResult:
    result = ValidationResult()

    telem_path = session_dir / "telemetry.parquet"
    events_path = session_dir / "events.parquet"

    if not telem_path.exists():
        result.fail("telemetry.parquet missing")
        return result

    table = pq.read_table(telem_path)
    n = len(table)
    result.stats["total_packets"] = n

    if n == 0:
        result.fail("telemetry.parquet is empty")
        return result

    ts_wall = table.column("ts_wall").to_pylist()
    ts_sim = table.column("ts_sim").to_pylist()

    # --- Sample rate check (≥ 95 Hz) ---
    duration = ts_wall[-1] - ts_wall[0]
    if duration > 0:
        actual_hz = (n - 1) / duration
        result.stats["sample_hz"] = round(actual_hz, 2)
        if actual_hz < 95.0:
            result.fail(f"sample rate {actual_hz:.1f} Hz < 95 Hz")
    else:
        result.warn("session duration is zero seconds")

    # --- Sequence gap check (missing seq ≤ 1%) ---
    seqs = table.column("seq").to_pylist()
    expected = seqs[-1] - seqs[0] + 1
    missing = expected - n
    miss_pct = missing / expected * 100 if expected > 0 else 0
    result.stats["missing_seq_pct"] = round(miss_pct, 3)
    if miss_pct > 1.0:
        result.fail(f"missing seq {miss_pct:.2f}% > 1%")

    # --- Timestamp drift check: максимальный gap между соседними ts_wall (≤ 100 ms) ---
    ts_arr = np.array(ts_wall)
    gaps = np.diff(ts_arr)
    max_gap_ms = float(np.max(gaps) * 1000) if len(gaps) > 0 else 0.0
    result.stats["max_gap_ms"] = round(max_gap_ms, 1)
    if max_gap_ms > 100.0:
        result.fail(f"max packet gap {max_gap_ms:.1f} ms > 100 ms")

    # --- Lap detection check (≥ 1 lap) ---
    laps = 0
    if events_path.exists():
        ev_table = pq.read_table(events_path)
        types = ev_table.column("event_type").to_pylist()
        laps = sum(1 for t in types if "lap" in t)
    result.stats["laps_detected"] = laps
    if laps < 1:
        result.fail("no laps detected — check gate detection or RH events")

    return result

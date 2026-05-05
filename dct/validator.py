"""Post-session validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass
class ValidationResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    stats:  dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def warn(self, msg: str) -> None:
        self.issues.append(f"WARN: {msg}")


_SRC_LIFTOFF = "liftoff"
_SRC_RC      = "rc"
_SRC_BOTH    = "both"


def validate_session(session_dir: Path, data_source: str = _SRC_LIFTOFF) -> ValidationResult:
    result = ValidationResult()
    use_liftoff = data_source in (_SRC_LIFTOFF, _SRC_BOTH)
    use_rc      = data_source in (_SRC_RC, _SRC_BOTH)

    if use_liftoff:
        _validate_liftoff(session_dir, result)
    if use_rc:
        _validate_rc(session_dir, result)

    return result


def _validate_liftoff(session_dir: Path, result: ValidationResult) -> None:
    telem_path  = session_dir / "telemetry.parquet"
    events_path = session_dir / "events.parquet"

    if not telem_path.exists():
        result.fail("telemetry.parquet missing")
        return

    table = pq.read_table(telem_path)
    n = len(table)
    result.stats["total_packets"] = n

    if n == 0:
        result.fail("telemetry.parquet is empty")
        return

    ts_wall = table.column("ts_wall").to_pylist()
    duration = ts_wall[-1] - ts_wall[0]
    if duration > 0:
        actual_hz = (n - 1) / duration
        result.stats["sample_hz"] = round(actual_hz, 2)
        if actual_hz < 95.0:
            result.fail(f"sample rate {actual_hz:.1f} Hz < 95 Hz")
    else:
        result.warn("session duration is zero seconds")

    seqs     = table.column("seq").to_pylist()
    expected = seqs[-1] - seqs[0] + 1
    missing  = expected - n
    miss_pct = missing / expected * 100 if expected > 0 else 0
    result.stats["missing_seq_pct"] = round(miss_pct, 3)
    if miss_pct > 1.0:
        result.fail(f"missing seq {miss_pct:.2f}% > 1%")

    ts_arr = np.array(ts_wall)
    gaps   = np.diff(ts_arr)
    max_gap_ms = float(np.max(gaps) * 1000) if len(gaps) > 0 else 0.0
    result.stats["max_gap_ms"] = round(max_gap_ms, 1)
    # Liftoff/Unity GC causes 200–400 ms pauses — warn only; fail above 500 ms
    if max_gap_ms > 500.0:
        result.fail(f"max packet gap {max_gap_ms:.1f} ms > 500 ms (severe dropout)")
    elif max_gap_ms > 200.0:
        result.warn(f"max packet gap {max_gap_ms:.1f} ms > 200 ms (Unity GC pause, data ok)")

    laps = gates_passed = 0
    if events_path.exists():
        ev_table = pq.read_table(events_path)
        types    = ev_table.column("event_type").to_pylist()
        laps     = sum(1 for t in types if "lap"  in t)
        gates_passed = sum(1 for t in types if "gate" in t)
    result.stats["laps_detected"] = laps
    result.stats["gates_passed"]  = gates_passed
    if gates_passed < 1:
        result.fail("no gates passed — session not saved (pilot must pass at least one gate)")


def _validate_rc(session_dir: Path, result: ValidationResult) -> None:
    """RC valid if any stick channel changes by >= 200 units within 3 seconds."""
    rc_path = session_dir / "rc_channels.parquet"
    result.stats["rc_valid"] = False

    if not rc_path.exists():
        result.fail("rc_channels.parquet missing")
        return

    table = pq.read_table(rc_path)
    n = len(table)
    result.stats["rc_packets"] = n

    if n == 0:
        result.fail("rc_channels.parquet is empty")
        return

    ts  = np.array(table.column("ts_wall").to_pylist())
    chs = [np.array(table.column(f"ch{i}").to_pylist(), dtype=np.int32) for i in range(1, 5)]

    valid = False
    j = 0
    for i in range(n):
        while j < n - 1 and ts[j + 1] - ts[i] <= 3.0:
            j += 1
        for arr in chs:
            if abs(int(arr[j]) - int(arr[i])) >= 200:
                valid = True
                break
        if valid:
            break

    result.stats["rc_valid"] = valid
    if not valid:
        result.fail("RC: no significant stick movement (>= 200 units within 3 s) detected")

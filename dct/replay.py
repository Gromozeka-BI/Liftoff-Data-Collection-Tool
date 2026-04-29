"""Replay engine — Этап 0.2 stub.

Reads a recorded session and re-emits telemetry packets over UDP and events
over the REST API at real-time (or accelerated) pace.
"""
from __future__ import annotations

import socket
import struct
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dct.session import load_track


def _pack_frame(row: dict[str, Any]) -> bytes:
    """Re-pack a telemetry row back into the LiftOff UDP wire format."""
    buf = struct.pack(
        "<f fff ffff fff fff ffff ff",
        row["ts_sim"],
        row["pos_x"], row["pos_y"], row["pos_z"],
        row["att_x"], row["att_y"], row["att_z"], row["att_w"],
        row["vel_x"], row["vel_y"], row["vel_z"],
        row["gyro_pitch"], row["gyro_roll"], row["gyro_yaw"],
        row["in_throttle"], row["in_yaw"], row["in_pitch"], row["in_roll"],
        row["bat_v"], row["bat_pct"],
    )
    motors = [row.get(f"motor_{i}", float("nan")) for i in range(4)]
    valid = [m for m in motors if m == m]  # filter NaN
    buf += struct.pack("<B", len(valid))
    for m in valid:
        buf += struct.pack("<f", m)
    return buf


class ReplayEngine:
    def __init__(
        self,
        session_dir: Path,
        target_host: str = "127.0.0.1",
        target_port: int = 9001,
        speed: float = 1.0,
    ):
        self._session_dir = session_dir
        self._host = target_host
        self._port = target_port
        self._speed = speed

    def run(self) -> None:
        table = pq.read_table(self._session_dir / "telemetry.parquet")
        rows = table.to_pylist()
        if not rows:
            raise ValueError("telemetry.parquet is empty")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        t0_wall = time.monotonic()
        t0_sim = rows[0]["ts_wall"]

        for row in rows:
            target_elapsed = (row["ts_wall"] - t0_sim) / self._speed
            actual_elapsed = time.monotonic() - t0_wall
            sleep = target_elapsed - actual_elapsed
            if sleep > 0:
                time.sleep(sleep)
            pkt = _pack_frame(row)
            sock.sendto(pkt, (self._host, self._port))

        sock.close()

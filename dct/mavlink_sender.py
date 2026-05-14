"""MAVLink UDP sender for Replay-derived telemetry."""
from __future__ import annotations

import time

from dct.mavlink_geo import GeoPoint


class MavlinkUdpSender:
    """Thin pymavlink wrapper that sends heartbeat and global position."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        source_system: int = 1,
        source_component: int = 1,
    ) -> None:
        from pymavlink import mavutil

        self._mavutil = mavutil
        self._conn = mavutil.mavlink_connection(
            f"udpout:{host}:{int(port)}",
            source_system=int(source_system),
            source_component=int(source_component),
        )
        self._last_heartbeat_mono = 0.0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def send_heartbeat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat_mono < 1.0:
            return
        mavlink = self._mavutil.mavlink
        self._conn.mav.heartbeat_send(
            mavlink.MAV_TYPE_GCS,
            mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavlink.MAV_STATE_ACTIVE,
        )
        self._last_heartbeat_mono = now

    def send_global_position(self, point: GeoPoint, ts_wall: float) -> None:
        self.send_heartbeat()
        self._conn.mav.global_position_int_send(
            int(max(ts_wall, 0.0) * 1000) & 0xFFFFFFFF,
            int(round(point.lat * 1e7)),
            int(round(point.lon * 1e7)),
            int(round(point.alt * 1000.0)),
            int(round(point.alt * 1000.0)),
            0,
            0,
            0,
            65535,
        )

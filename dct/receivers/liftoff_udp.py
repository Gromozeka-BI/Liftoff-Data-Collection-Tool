"""UDP receiver for LiftOff telemetry.

Packet layout (little-endian):
  Timestamp   1 float   4 B
  Position    3 floats 12 B
  Attitude    4 floats 16 B   quaternion X Y Z W
  Velocity    3 floats 12 B
  Gyro        3 floats 12 B   deg/s
  Input       4 floats 16 B   throttle yaw pitch roll
  Battery     2 floats  8 B   voltage pct  (optional — may be missing)
  MotorCount  1 byte    1 B
  MotorRPM    n floats  4n B
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from queue import Queue, Full
from typing import Any

from dct.log import get_logger

_log = get_logger("udp")


_HEADER_FMT = "<f fff ffff fff fff ffff"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 80 B without battery/motors
_BATTERY_SIZE = 8  # 2 floats


def _parse(raw: bytes, seq: int, ts_wall: float) -> dict[str, Any] | None:
    try:
        offset = 0
        ts_sim = struct.unpack_from("<f", raw, offset)[0]; offset += 4
        pos = struct.unpack_from("<fff", raw, offset); offset += 12
        att = struct.unpack_from("<ffff", raw, offset); offset += 16
        vel = struct.unpack_from("<fff", raw, offset); offset += 12
        gyro = struct.unpack_from("<fff", raw, offset); offset += 12
        inp = struct.unpack_from("<ffff", raw, offset); offset += 16

        bat_v, bat_pct = 0.0, 0.0
        if offset + _BATTERY_SIZE <= len(raw):
            bat_v, bat_pct = struct.unpack_from("<ff", raw, offset)
            offset += _BATTERY_SIZE

        motor_rpms = [float("nan")] * 4
        if offset < len(raw):
            n = struct.unpack_from("<B", raw, offset)[0]; offset += 1
            for i in range(min(n, 4)):
                motor_rpms[i] = struct.unpack_from("<f", raw, offset)[0]; offset += 4

        return {
            "seq": seq,
            "ts_wall": ts_wall,
            "ts_sim": ts_sim,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "att_x": att[0], "att_y": att[1], "att_z": att[2], "att_w": att[3],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "gyro_pitch": gyro[0], "gyro_roll": gyro[1], "gyro_yaw": gyro[2],
            "in_throttle": inp[0], "in_yaw": inp[1], "in_pitch": inp[2], "in_roll": inp[3],
            "bat_v": bat_v, "bat_pct": bat_pct,
            "motor_0": motor_rpms[0], "motor_1": motor_rpms[1],
            "motor_2": motor_rpms[2], "motor_3": motor_rpms[3],
        }
    except Exception:
        return None


class LiftoffUDPReceiver:
    def __init__(self, host: str, port: int, queue_size: int = 2000):
        self._host = host
        self._port = port
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=queue_size)
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._running = False
        self.dropped = 0
        self.received = 0
        self.failed = 0

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        _log.info("UDP receiver started on %s:%s", self._host, self._port)

    def stop(self) -> None:
        self._running = False
        self._sock.close()
        _log.info("UDP receiver stopped — received=%d dropped=%d failed=%d",
                  self.received, self.dropped, self.failed)

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            ts = time.time()
            self._seq += 1
            self.received += 1
            parsed = _parse(data, self._seq, ts)
            if parsed is None:
                self.failed += 1
                continue
            try:
                self._queue.put_nowait(parsed)
            except Full:
                self.dropped += 1
                if self.dropped % 100 == 1:
                    _log.warning("UDP queue full — total dropped=%d", self.dropped)

    @property
    def queue(self) -> Queue:
        return self._queue

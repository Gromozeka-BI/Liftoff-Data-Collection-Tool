"""Mock RotorHazard simulator.

Runs as a background thread. Receives telemetry frames, detects gate crossings
based on 3D proximity, and fires events to the DCT REST API.

Gate crossing logic:
  - drone enters radius → arm gate
  - drone exits radius → fire event (prevents double-fire while inside)
"""
from __future__ import annotations

import math
import threading
import time
from queue import Queue
from typing import Any

import urllib.request
import urllib.parse
import json


def _distance(px: float, py: float, pz: float, gate: dict[str, Any]) -> float:
    gx, gy, gz = gate["position"]
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2)


def _post(url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass


class RHSimulator:
    def __init__(self, api_base: str, gates: list[dict[str, Any]], start_finish_id: int, gate_radius: float):
        self._api = api_base.rstrip("/")
        self._gates = gates
        self._sf_id = start_finish_id
        self._radius = gate_radius
        self._inside: set[int] = set()  # gate ids currently inside radius
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=500)
        self._thread: threading.Thread | None = None
        self._running = False
        self.lap_count = 0

    def feed(self, frame: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(frame)
        except Exception:
            pass

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except Exception:
                continue
            self._process(frame)

    def _process(self, frame: dict[str, Any]) -> None:
        px, py, pz = frame["pos_x"], frame["pos_y"], frame["pos_z"]
        ts = frame["ts_wall"]

        for gate in self._gates:
            gid = gate["id"]
            dist = _distance(px, py, pz, gate)
            in_now = dist <= self._radius

            was_inside = gid in self._inside

            if in_now and not was_inside:
                self._inside.add(gid)

            elif not in_now and was_inside:
                self._inside.discard(gid)
                self._fire(gid, ts)

    def _fire(self, gate_id: int, ts: float) -> None:
        if gate_id == self._sf_id:
            self.lap_count += 1
            _post(f"{self._api}/api/v1/rh/lap", {"gate_id": gate_id, "ts_wall": ts})
        else:
            _post(f"{self._api}/api/v1/rh/gate", {"gate_id": gate_id, "ts_wall": ts})

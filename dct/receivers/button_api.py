"""FastAPI server for discrete event sources: RotorHazard adapter and buttons."""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dct.log import get_logger

_log = get_logger("button_api")


class LapEvent(BaseModel):
    pilot: str | None = None
    gate_id: int | None = None
    ts_wall: float | None = None


class GateEvent(BaseModel):
    gate_id: int
    pilot: str | None = None
    ts_wall: float | None = None


class ButtonAPI:
    """Runs a FastAPI server in a background thread, feeds events into a queue."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self.events: Queue[dict[str, Any]] = Queue()
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.lap_count = 0

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="DCT Event API", version="0.1")

        @app.get("/api/v1/status")
        def status():
            return {"status": "recording", "laps": self.lap_count}

        @app.post("/api/v1/rh/lap")
        def rh_lap(ev: LapEvent):
            self._push("rh_lap", ev.gate_id, ev.ts_wall)
            return {"ok": True}

        @app.post("/api/v1/rh/gate")
        def rh_gate(ev: GateEvent):
            self._push("rh_gate", ev.gate_id, ev.ts_wall)
            return {"ok": True}

        @app.post("/api/v1/button/lap")
        def btn_lap(ev: LapEvent):
            self._push("button_lap", ev.gate_id, ev.ts_wall)
            return {"ok": True}

        @app.post("/api/v1/button/gate")
        def btn_gate(ev: GateEvent):
            self._push("button_gate", ev.gate_id, ev.ts_wall)
            return {"ok": True}

        return app

    def _push(self, event_type: str, gate_id: int | None, ts: float | None) -> None:
        if event_type.endswith("lap"):
            self.lap_count += 1
        _log.debug("event: type=%s gate_id=%s lap_count=%d", event_type, gate_id, self.lap_count)
        self.events.put({
            "event_type": event_type,
            "gate_id": gate_id,
            "ts_wall": ts or time.time(),
            "lap_num": self.lap_count if "lap" in event_type else None,
        })

    def start(self) -> None:
        _log.info("ButtonAPI starting on %s:%s", self._host, self._port)
        cfg = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            loop="asyncio",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        _log.info("ButtonAPI stopping")
        if self._server:
            self._server.should_exit = True

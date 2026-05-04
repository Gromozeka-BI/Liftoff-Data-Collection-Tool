"""Data sources for the GUI.

LiveDataSource   — wraps UDP receiver + ButtonAPI + writers + RH sim + RC receiver.
ReplayDataSource — reads parquet and replays with timing control.
"""
from __future__ import annotations

import time
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from dct.config import settings
from dct.gui.video_reader import VideoReader
from dct.log import get_logger
from dct.rc_receiver import RCReceiver
from dct.receivers.liftoff_udp import LiftoffUDPReceiver
from dct.receivers.button_api import ButtonAPI
from dct.rh_simulator import RHSimulator
from dct.screen_recorder import (
    ScreenRecorder, CaptureDeviceRecorder, PyAvCaptureRecorder,
    list_all_video_device_names, is_virtual_device,
)
from dct.session import create_session, copy_track, load_track
from dct.storage.writer import TelemetryWriter, EventsWriter, RCChannelsWriter, TimelineWriter
from dct.system_logger import SystemLogger

_log = get_logger("data_source")

_SRC_LIFTOFF = "liftoff"
_SRC_RC      = "rc"
_SRC_BOTH    = "both"


# ── Stop thread ───────────────────────────────────────────────────────────────

class _StopThread(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, udp, rh_sim, recorder, api, tw, ew, rc_recv, rc_tw, tl_tw, syslog,
                 session_dir, stats, start_time, data_source_mode):
        super().__init__()
        self._udp        = udp
        self._rh_sim     = rh_sim
        self._recorder   = recorder
        self._api        = api
        self._tw         = tw
        self._ew         = ew
        self._rc_recv    = rc_recv
        self._rc_tw      = rc_tw
        self._tl_tw      = tl_tw
        self._syslog     = syslog
        self._session_dir = session_dir
        self._stats      = dict(stats)
        self._start_time = start_time
        self._ds_mode    = data_source_mode

    def run(self) -> None:
        import shutil
        from dct.session import finalize_meta
        from dct.validator import validate_session

        # Stop all components; each step is guarded so one failure doesn't
        # prevent subsequent steps (especially writer.close calls).
        try:
            if self._udp:     self._udp.stop()
        except Exception as e:
            _log.error("Error stopping UDP: %s", e)
        try:
            if self._rh_sim:  self._rh_sim.stop()
        except Exception as e:
            _log.error("Error stopping RH sim: %s", e)
        try:
            if self._recorder:
                self._recorder.stop()
                self._stats["frames"] = self._recorder.frames_written
        except Exception as e:
            _log.error("Error stopping recorder: %s", e)
        try:
            if self._api:     self._api.stop()
        except Exception as e:
            _log.error("Error stopping API: %s", e)
        try:
            if self._rc_recv: self._rc_recv.stop()
        except Exception as e:
            _log.error("Error stopping RC receiver: %s", e)
        try:
            if self._tw:      self._tw.close()
        except Exception as e:
            _log.error("Error closing telemetry writer: %s", e)
        try:
            if self._rc_tw:   self._rc_tw.close()
        except Exception as e:
            _log.error("Error closing RC writer: %s", e)
        try:
            if self._tl_tw:   self._tl_tw.close()
        except Exception as e:
            _log.error("Error closing timeline writer: %s", e)
        try:
            if self._ew:
                self._ew.write_event("session_stop", time.time(), source="dct")
                self._ew.close()
        except Exception as e:
            _log.error("Error closing events writer: %s", e)
        try:
            if self._syslog:  self._syslog.close()
        except Exception as e:
            _log.error("Error closing syslog: %s", e)

        if self._session_dir:
            try:
                finalize_meta(self._session_dir, self._stats["packets"],
                              self._stats["laps"], self._start_time)
            except Exception as e:
                _log.error("Error finalizing meta: %s", e)

            result = None
            try:
                result = validate_session(self._session_dir, self._ds_mode)
            except Exception as e:
                _log.error("Validation crashed for %s: %s", self._session_dir, e)

            # Only delete sessions that have no data at all (empty directory).
            # If validation crashed (corrupt file, etc.) we keep the session so
            # the user doesn't lose data.  If validation passed but found no
            # useful content, still delete (zero packets and no RC activity).
            should_delete = False
            if result is not None:
                no_gates   = result.stats.get("gates_passed", 0) < 1
                no_rc      = not result.stats.get("rc_valid")
                no_packets = result.stats.get("rc_packets", 0) == 0 and self._stats.get("packets", 0) == 0
                should_delete = no_gates and no_rc and no_packets

            if should_delete:
                _log.warning("No valid data — deleting session %s", self._session_dir)
                deleted = False
                for _attempt in range(5):
                    try:
                        shutil.rmtree(self._session_dir)
                        deleted = True
                        break
                    except Exception:
                        time.sleep(0.3)
                if not deleted:
                    _log.error("Failed to delete session after retries: %s", self._session_dir)

            self.result_ready.emit({
                "session_dir": str(self._session_dir),
                "stats":       self._stats,
                "validation":  result,
            })


# ── Live ──────────────────────────────────────────────────────────────────────

class LiveDataSource(QObject):
    telemetry_updated = pyqtSignal(dict)
    telemetry_batch   = pyqtSignal(list)
    event_fired       = pyqtSignal(dict)
    stats_updated     = pyqtSignal(dict)
    video_frame       = pyqtSignal(object)
    rc_batch          = pyqtSignal(list)        # list of RC frame dicts
    rc_status_changed = pyqtSignal(bool)        # True=online
    session_started   = pyqtSignal(str)
    session_stopped   = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._udp:        LiftoffUDPReceiver | None = None
        self._api:        ButtonAPI | None          = None
        self._recorder:   ScreenRecorder | CaptureDeviceRecorder | None = None
        self._rh_sim:     RHSimulator | None        = None
        self._tw:         TelemetryWriter | None    = None
        self._ew:         EventsWriter | None       = None
        self._rc_recv:    RCReceiver | None         = None
        self._rc_tw:      RCChannelsWriter | None   = None
        self._tl_tw:      TimelineWriter | None     = None
        self._syslog:     SystemLogger | None       = None
        self._session_dir: Path | None              = None
        self._start_time  = 0.0
        self._stats:      dict[str, Any]            = {}
        self._stop_thread: _StopThread | None       = None
        self._ds_mode     = _SRC_LIFTOFF
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ── public ─────────────────────────────────────────────────────────────

    def start_session(self, cfg: dict) -> None:
        self._ds_mode = cfg.get("data_source", _SRC_LIFTOFF)
        use_liftoff   = self._ds_mode in (_SRC_LIFTOFF, _SRC_BOTH)
        use_rc        = self._ds_mode in (_SRC_RC, _SRC_BOTH)

        self._session_dir = create_session(
            cfg["pilot"], cfg["drone"], cfg["track"], cfg.get("purpose", "training"),
            base_dir=cfg.get("sessions_dir"),
        )
        if cfg.get("track_path"):
            copy_track(self._session_dir, Path(cfg["track_path"]))

        track_data = load_track(self._session_dir)

        # System logger — always
        self._syslog = SystemLogger(self._session_dir)
        self._syslog.log("sources_configured", data_source=self._ds_mode,
                         video_source=cfg.get("video_source", {}).get("type"),
                         rc_port=cfg.get("rc_port"))

        # Writers
        self._tw   = TelemetryWriter(self._session_dir,
                                     flush_rows=settings.parquet_flush_rows,
                                     flush_interval=settings.parquet_flush_interval)
        self._ew   = EventsWriter(self._session_dir)
        self._rc_tw = RCChannelsWriter(self._session_dir) if use_rc else None
        self._tl_tw = TimelineWriter(self._session_dir)
        self._ew.write_event("session_start", time.time(), source="dct")

        # Liftoff UDP
        if use_liftoff:
            self._udp = LiftoffUDPReceiver(settings.udp_host, settings.udp_port)
            try:
                self._udp.start()
            except OSError as e:
                raise RuntimeError(f"Cannot bind UDP {settings.udp_port}: {e}") from e

        # Button API + RotorHazard sim — always (needed for gate/lap even in RC-only mode)
        self._api = ButtonAPI(settings.api_host, settings.api_port)
        self._api.start()

        if track_data:
            gates  = track_data.get("gates", [])
            sf_id  = next((g["id"] for g in gates if g.get("is_start_finish")), 0)
            self._rh_sim = RHSimulator(
                f"http://127.0.0.1:{settings.api_port}",
                gates, sf_id, settings.rh_gate_radius,
            )
            self._rh_sim.start()

        # Video recorder
        if not cfg.get("no_video"):
            vsrc = cfg.get("video_source") or {"type": "screen"}
            if vsrc.get("type") == "device":
                dev_idx = vsrc["index"]
                # Detect virtual cameras: libav/dshow cannot open them, use OpenCV instead.
                all_names = list_all_video_device_names()
                dev_name = all_names[dev_idx] if dev_idx < len(all_names) else ""
                if dev_name and is_virtual_device(dev_name):
                    _log.info(
                        "Device '%s' is a virtual camera — using OpenCV CaptureDeviceRecorder",
                        dev_name,
                    )
                    self._recorder = CaptureDeviceRecorder(
                        self._session_dir / "video.mp4",
                        device_index=dev_idx,
                        fps=settings.screen_fps,
                        target_w=settings.screen_width,
                        target_h=settings.screen_height,
                    )
                else:
                    self._recorder = PyAvCaptureRecorder(
                        self._session_dir / "video.mp4",
                        device_index=dev_idx,
                        fps=settings.screen_fps,
                        target_w=settings.screen_width,
                        target_h=settings.screen_height,
                    )
            else:
                self._recorder = ScreenRecorder(
                    self._session_dir / "video.mp4",
                    settings.screen_window_title,
                    fps=settings.screen_fps,
                    target_w=settings.screen_width,
                    target_h=settings.screen_height,
                )
            self._recorder.start()

        # RC receiver
        if use_rc and cfg.get("rc_port"):
            self._rc_recv = RCReceiver(
                cfg["rc_port"],
                on_status_change=self._on_rc_status,
            )
            self._rc_recv.start()

        self._stats      = {"packets": 0, "laps": 0, "dropped": 0,
                            "duration": 0.0, "hz": 0.0, "frames": 0, "rc_packets": 0}
        self._start_time = time.time()
        self._timer.start(33)
        _log.info("Session started: %s (mode=%s)", self._session_dir, self._ds_mode)
        self.session_started.emit(str(self._session_dir))

    def stop_session(self) -> None:
        if self._stop_thread is not None and self._stop_thread.isRunning():
            _log.warning("stop_session() called while stop already in progress — ignoring")
            return
        self._timer.stop()
        self._stop_thread = _StopThread(
            self._udp, self._rh_sim, self._recorder, self._api,
            self._tw, self._ew, self._rc_recv, self._rc_tw, self._tl_tw, self._syslog,
            self._session_dir, self._stats, self._start_time, self._ds_mode,
        )
        self._stop_thread.result_ready.connect(self._on_stop_finished)
        self._stop_thread.start()
        self._udp = self._rh_sim = self._recorder = self._api = None
        self._rc_recv = self._tw = self._ew = self._rc_tw = self._tl_tw = self._syslog = None

    def mark_lap(self) -> None:
        if self._api:
            import urllib.request, json
            data = json.dumps({"ts_wall": time.time()}).encode()
            req  = urllib.request.Request(
                f"http://127.0.0.1:{settings.api_port}/api/v1/button/lap",
                data=data, headers={"Content-Type": "application/json"},
            )
            try: urllib.request.urlopen(req, timeout=1)
            except Exception: pass

    def mark_gate(self, gate_id: int) -> None:
        if self._api:
            import urllib.request, json
            data = json.dumps({"gate_id": gate_id, "ts_wall": time.time()}).encode()
            req  = urllib.request.Request(
                f"http://127.0.0.1:{settings.api_port}/api/v1/button/gate",
                data=data, headers={"Content-Type": "application/json"},
            )
            try: urllib.request.urlopen(req, timeout=1)
            except Exception: pass

    # ── internal ───────────────────────────────────────────────────────────

    def _on_rc_status(self, online: bool) -> None:
        if self._syslog:
            self._syslog.log("rc_status", online=online)
        self.rc_status_changed.emit(online)

    def _on_stop_finished(self, result: dict) -> None:
        val = result.get("validation")
        if val:
            status = "PASSED" if val.passed else "FAILED"
            _log.info("Session stopped: %s | packets=%d laps=%d validation=%s",
                      result.get("session_dir"), result["stats"].get("packets", 0),
                      result["stats"].get("laps", 0), status)
            for issue in val.issues:
                _log.warning("Validation issue: %s", issue)
        self._session_dir = None
        self.session_stopped.emit(result)

    def _tick(self) -> None:
        # ── Liftoff telemetry ────────────────────────────────────────────
        lf_frames: list[dict] = []
        if self._udp:
            while len(lf_frames) < 300:
                try:
                    frame = self._udp.queue.get_nowait()
                except Empty:
                    break
                self._tw.write(frame)
                if self._rh_sim:
                    self._rh_sim.feed(frame)
                self._stats["packets"] += 1
                lf_frames.append(frame)

        if lf_frames:
            self.telemetry_batch.emit(lf_frames)
            self.telemetry_updated.emit(lf_frames[-1])

        # ── RC frames ────────────────────────────────────────────────────
        rc_frames: list[dict] = []
        if self._rc_recv:
            while len(rc_frames) < 300:
                try:
                    frame = self._rc_recv.queue.get_nowait()
                except Empty:
                    break
                if self._rc_tw:
                    self._rc_tw.write(frame)
                self._stats["rc_packets"] += 1
                rc_frames.append(frame)

        if rc_frames:
            self.rc_batch.emit(rc_frames)

        # ── Button events ────────────────────────────────────────────────
        while self._api:
            try:
                ev = self._api.events.get_nowait()
            except Empty:
                break
            self._ew.write_event(
                ev["event_type"], ev["ts_wall"],
                gate_id=ev.get("gate_id"),
                lap_num=ev.get("lap_num"),
                source="api",
            )
            if "lap" in ev["event_type"]:
                self._stats["laps"] += 1
            self.event_fired.emit(ev)

        # ── Timeline tick ────────────────────────────────────────────────
        ts_now = time.time()
        if self._tl_tw:
            self._tl_tw.tick(ts_now)

        # ── Stats + video frame ──────────────────────────────────────────
        dur = ts_now - self._start_time
        self._stats["duration"] = dur
        self._stats["dropped"]  = self._udp.dropped if self._udp else 0
        self._stats["hz"]       = round(self._stats["packets"] / dur, 1) if dur > 0 else 0.0
        if self._recorder:
            self._stats["frames"] = self._recorder.frames_written
            if self._recorder.latest_frame_bgr is not None:
                self.video_frame.emit(self._recorder.latest_frame_bgr)
        self.stats_updated.emit(dict(self._stats))


# ── Replay ────────────────────────────────────────────────────────────────────

class ReplayDataSource(QObject):
    telemetry_updated = pyqtSignal(dict)
    telemetry_batch   = pyqtSignal(list)
    rc_batch          = pyqtSignal(list)
    event_fired       = pyqtSignal(dict)
    video_frame       = pyqtSignal(object)
    stats_updated     = pyqtSignal(dict)
    progress_updated  = pyqtSignal(float, float)
    finished          = pyqtSignal()

    def __init__(self, session_dir: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        import pyarrow.parquet as pq

        p = Path(session_dir)

        # ── Telemetry ─────────────────────────────────────────────────────
        telem_path = p / "telemetry.parquet"
        self._rows: list[dict] = (
            pq.read_table(telem_path).to_pylist() if telem_path.exists() else []
        )
        ev_path = p / "events.parquet"
        self._events: list[dict] = (
            pq.read_table(ev_path).to_pylist() if ev_path.exists() else []
        )

        # ── RC channels ───────────────────────────────────────────────────
        rc_path = p / "rc_channels.parquet"
        self._rc_rows: list[dict] = (
            pq.read_table(rc_path).to_pylist() if rc_path.exists() else []
        )
        self._rc_ts_arr = (
            np.array([r["ts_wall"] for r in self._rc_rows]) if self._rc_rows else np.array([])
        )
        self._rc_idx = 0

        # ── Timeline (primary clock for scrubbing) ────────────────────────
        tl_path = p / "timeline.parquet"
        if tl_path.exists():
            tl_rows = pq.read_table(tl_path).to_pylist()
            self._tl_ts = np.array([r["ts_wall"] for r in tl_rows])
        elif self._rows:
            self._tl_ts = np.array([r["ts_wall"] for r in self._rows])
        elif self._rc_rows:
            self._tl_ts = self._rc_ts_arr.copy()
        else:
            self._tl_ts = np.array([])

        # Telemetry index arrays
        self._ts_arr = (
            np.array([r["ts_wall"] for r in self._rows]) if self._rows else np.array([])
        )
        self._ev_ts_arr = (
            np.array([e["ts_wall"] for e in self._events]) if self._events else np.array([])
        )
        self._lap_event_mask = np.array(
            ["lap" in e.get("event_type", "") for e in self._events], dtype=bool
        ) if self._events else np.array([], dtype=bool)

        self._idx    = 0
        self._ev_idx = 0
        self._speed  = 1.0
        self._paused = True
        self._wall_anchor = 0.0
        self._sim_anchor  = 0.0

        # ── Video ─────────────────────────────────────────────────────────
        self._video_reader: VideoReader | None = None
        vid_path    = p / "video.mp4"
        vid_ts_path = p / "video_timestamps.parquet"
        if vid_path.exists() and vid_ts_path.exists():
            vid_ts = np.array(pq.read_table(vid_ts_path)["ts_wall"].to_pylist())
            self._video_reader = VideoReader(vid_path, vid_ts)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ── public ─────────────────────────────────────────────────────────────

    @property
    def total_duration(self) -> float:
        if len(self._tl_ts) < 2:
            return 0.0
        return float(self._tl_ts[-1] - self._tl_ts[0])

    @property
    def first_ts(self) -> float:
        return float(self._tl_ts[0]) if len(self._tl_ts) else 0.0

    def play(self) -> None:
        if len(self._tl_ts) == 0:
            return
        if self._idx >= len(self._rows) and self._rc_idx >= len(self._rc_rows):
            self.seek(0.0)
        self._wall_anchor = time.monotonic()
        self._sim_anchor  = (
            self._rows[self._idx]["ts_wall"] if self._idx < len(self._rows)
            else (self._rc_rows[self._rc_idx]["ts_wall"] if self._rc_idx < len(self._rc_rows)
                  else self._tl_ts[0])
        )
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def toggle_play(self) -> None:
        self.pause() if not self._paused else self.play()

    def set_speed(self, speed: float) -> None:
        if not self._paused and self._idx > 0:
            self._sim_anchor  = self._rows[self._idx - 1]["ts_wall"] if self._rows else self._tl_ts[0]
            self._wall_anchor = time.monotonic()
        self._speed = speed

    def seek(self, fraction: float) -> None:
        if not len(self._tl_ts):
            return
        t0, t1 = self._tl_ts[0], self._tl_ts[-1]
        target = t0 + max(0.0, min(1.0, fraction)) * (t1 - t0)

        if len(self._ts_arr):
            self._idx    = int(np.searchsorted(self._ts_arr, target))
            self._idx    = max(0, min(self._idx, len(self._rows) - 1))
        if len(self._rc_ts_arr):
            self._rc_idx = int(np.searchsorted(self._rc_ts_arr, target))
            self._rc_idx = max(0, min(self._rc_idx, len(self._rc_rows) - 1))
        if len(self._ev_ts_arr):
            self._ev_idx = int(np.searchsorted(self._ev_ts_arr, target))

        self._sim_anchor  = target
        self._wall_anchor = time.monotonic()

        if self._rows and self._idx < len(self._rows):
            self.telemetry_updated.emit(self._rows[self._idx])
        self._emit_video_frame(target)

    def _tick(self) -> None:
        if self._paused or not len(self._tl_ts):
            return

        target = self._sim_anchor + (time.monotonic() - self._wall_anchor) * self._speed

        # Emit telemetry batch
        batch: list[dict] = []
        while self._idx < len(self._rows) and self._rows[self._idx]["ts_wall"] <= target and len(batch) < 50:
            batch.append(self._rows[self._idx])
            self._idx += 1
        if self._rows and self._idx < len(self._rows) and self._rows[self._idx]["ts_wall"] < target - 0.5:
            self._idx = int(np.searchsorted(self._ts_arr, target - 0.1))
        if batch:
            self.telemetry_batch.emit(batch)
            self.telemetry_updated.emit(batch[-1])

        # Emit RC batch
        rc_batch: list[dict] = []
        while self._rc_idx < len(self._rc_rows) and self._rc_rows[self._rc_idx]["ts_wall"] <= target and len(rc_batch) < 50:
            rc_batch.append(self._rc_rows[self._rc_idx])
            self._rc_idx += 1
        if rc_batch:
            self.rc_batch.emit(rc_batch)

        # Events
        while self._ev_idx < len(self._events) and self._events[self._ev_idx]["ts_wall"] <= target:
            self.event_fired.emit(self._events[self._ev_idx])
            self._ev_idx += 1

        # Always update video position every tick — video must advance even in
        # RC-only mode where telemetry batch is empty.
        self._emit_video_frame(target)

        # Check finished (use timeline as primary)
        tl_done = target >= self._tl_ts[-1]
        if tl_done:
            self._paused = True
            self.progress_updated.emit(self.total_duration, self.total_duration)
            self.finished.emit()
            return

        current = target - self._tl_ts[0]
        self.progress_updated.emit(current, self.total_duration)

        laps_now = int(np.sum(self._lap_event_mask[:self._ev_idx])) if len(self._lap_event_mask) else 0
        hz = 0.0
        if len(self._ts_arr) > 10:
            median_dt = float(np.median(np.diff(self._ts_arr)))
            hz = round(1.0 / median_dt, 1) if median_dt > 0 else 0.0
        self.stats_updated.emit({
            "packets":  self._idx,
            "hz":       hz,
            "laps":     laps_now,
            "dropped":  0,
            "duration": current,
        })

    def _emit_video_frame(self, ts_wall: float) -> None:
        if self._video_reader is None:
            return
        self._video_reader.set_target_ts(ts_wall)
        frame = self._video_reader.get_latest()
        if frame is not None:
            self.video_frame.emit(frame)

    def deleteLater(self) -> None:
        if self._video_reader is not None:
            self._video_reader.stop()
            self._video_reader = None
        super().deleteLater()

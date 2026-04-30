"""Data sources for the GUI.

LiveDataSource   — wraps UDP receiver + ButtonAPI + writers + RH sim.
ReplayDataSource — reads parquet and replays with timing control.

Both emit the same signals so the UI is source-agnostic.
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
from dct.receivers.liftoff_udp import LiftoffUDPReceiver
from dct.receivers.button_api import ButtonAPI
from dct.rh_simulator import RHSimulator
from dct.screen_recorder import ScreenRecorder
from dct.session import create_session, copy_track, load_track
from dct.storage.writer import TelemetryWriter, EventsWriter

_log = get_logger("data_source")


# ── Фоновый поток для остановки сессии ───────────────────────────────────────

class _StopThread(QThread):
    """Выполняет тяжёлые операции остановки (recorder.join, flush, validate) вне UI-потока."""
    result_ready = pyqtSignal(dict)

    def __init__(self, udp, rh_sim, recorder, api, tw, ew,
                 session_dir, stats, start_time):
        super().__init__()
        self._udp = udp
        self._rh_sim = rh_sim
        self._recorder = recorder
        self._api = api
        self._tw = tw
        self._ew = ew
        self._session_dir = session_dir
        self._stats = dict(stats)
        self._start_time = start_time

    def run(self) -> None:
        import shutil
        from dct.session import finalize_meta
        from dct.validator import validate_session

        if self._udp:    self._udp.stop()
        if self._rh_sim: self._rh_sim.stop()
        if self._recorder:
            self._recorder.stop()
            self._stats["frames"] = self._recorder.frames_written
        if self._api:    self._api.stop()
        if self._tw:     self._tw.close()
        if self._ew:
            self._ew.write_event("session_stop", time.time(), source="dct")
            self._ew.close()
        if self._session_dir:
            finalize_meta(self._session_dir, self._stats["packets"],
                          self._stats["laps"], self._start_time)
            result = validate_session(self._session_dir)

            # Если пилот не прошёл ни одного гейта — удаляем сессию
            if result.stats.get("gates_passed", 0) < 1:
                _log.warning("No gates passed — deleting session %s", self._session_dir)
                try:
                    shutil.rmtree(self._session_dir)
                except Exception as e:
                    _log.error("Failed to delete session dir: %s", e)

            self.result_ready.emit({
                "session_dir": str(self._session_dir),
                "stats": self._stats,
                "validation": result,
            })


# ── Live ──────────────────────────────────────────────────────────────────────

class LiveDataSource(QObject):
    telemetry_updated = pyqtSignal(dict)   # latest frame (~30 fps)
    telemetry_batch   = pyqtSignal(list)   # all frames since last tick
    event_fired       = pyqtSignal(dict)
    stats_updated     = pyqtSignal(dict)
    video_frame       = pyqtSignal(object) # np.ndarray BGR
    session_started   = pyqtSignal(str)    # session dir path
    session_stopped   = pyqtSignal(dict)   # {session_dir, stats, validation}

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._udp: LiftoffUDPReceiver | None = None
        self._api: ButtonAPI | None = None
        self._recorder: ScreenRecorder | None = None
        self._rh_sim: RHSimulator | None = None
        self._tw: TelemetryWriter | None = None
        self._ew: EventsWriter | None = None
        self._session_dir: Path | None = None
        self._start_time = 0.0
        self._stats: dict[str, Any] = {}
        self._stop_thread: _StopThread | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ── public API ─────────────────────────────────────────────────────────

    def start_session(self, cfg: dict) -> None:
        self._session_dir = create_session(
            cfg["pilot"], cfg["drone"], cfg["track"], cfg.get("purpose", "training")
        )
        if cfg.get("track_path"):
            copy_track(self._session_dir, Path(cfg["track_path"]))

        track_data = load_track(self._session_dir)

        self._tw = TelemetryWriter(
            self._session_dir,
            flush_rows=settings.parquet_flush_rows,
            flush_interval=settings.parquet_flush_interval,
        )
        self._ew = EventsWriter(self._session_dir)
        self._ew.write_event("session_start", time.time(), source="dct")

        self._udp = LiftoffUDPReceiver(settings.udp_host, settings.udp_port)
        try:
            self._udp.start()
        except OSError as e:
            raise RuntimeError(f"Cannot bind UDP {settings.udp_port}: {e}") from e

        self._api = ButtonAPI(settings.api_host, settings.api_port)
        self._api.start()

        if not cfg.get("no_video"):
            self._recorder = ScreenRecorder(
                self._session_dir / "video.mp4",
                settings.screen_window_title,
                fps=settings.screen_fps,
                target_w=settings.screen_width,
                target_h=settings.screen_height,
            )
            self._recorder.start()

        if not cfg.get("no_rh_sim") and track_data:
            gates = track_data.get("gates", [])
            sf_id = next((g["id"] for g in gates if g.get("is_start_finish")), 0)
            self._rh_sim = RHSimulator(
                f"http://127.0.0.1:{settings.api_port}",
                gates, sf_id, settings.rh_gate_radius,
            )
            self._rh_sim.start()

        self._stats = {"packets": 0, "laps": 0, "dropped": 0, "duration": 0.0,
                       "hz": 0.0, "frames": 0}
        self._start_time = time.time()
        self._timer.start(33)
        _log.info("Session started: %s", self._session_dir)
        self.session_started.emit(str(self._session_dir))

    def stop_session(self) -> None:
        """Немедленно останавливает тикер и запускает тяжёлые операции в фоне."""
        self._timer.stop()
        self._stop_thread = _StopThread(
            self._udp, self._rh_sim, self._recorder, self._api,
            self._tw, self._ew, self._session_dir, self._stats, self._start_time,
        )
        self._stop_thread.result_ready.connect(self._on_stop_finished)
        self._stop_thread.start()
        # Сбрасываем ссылки — поток владеет объектами до завершения
        self._udp = self._rh_sim = self._recorder = self._api = None
        self._tw = self._ew = None

    def _on_stop_finished(self, result: dict) -> None:
        val = result.get("validation")
        if val:
            _log.info("Session stopped: %s | packets=%d laps=%d validation=%s",
                      result.get("session_dir"), result["stats"].get("packets", 0),
                      result["stats"].get("laps", 0),
                      "PASSED" if val.passed else "FAILED")
            if not val.passed:
                for issue in val.issues:
                    _log.warning("Validation issue: %s", issue)
        self._session_dir = None
        self.session_stopped.emit(result)

    def mark_lap(self) -> None:
        if self._api:
            import urllib.request, json
            data = json.dumps({"ts_wall": time.time()}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{settings.api_port}/api/v1/button/lap",
                data=data, headers={"Content-Type": "application/json"},
            )
            try: urllib.request.urlopen(req, timeout=1)
            except Exception: pass

    def mark_gate(self, gate_id: int) -> None:
        if self._api:
            import urllib.request, json
            data = json.dumps({"gate_id": gate_id, "ts_wall": time.time()}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{settings.api_port}/api/v1/button/gate",
                data=data, headers={"Content-Type": "application/json"},
            )
            try: urllib.request.urlopen(req, timeout=1)
            except Exception: pass

    # ── internal ───────────────────────────────────────────────────────────

    def _tick(self) -> None:
        frames: list[dict] = []
        while len(frames) < 300:
            try:
                frame = self._udp.queue.get_nowait()
            except Empty:
                break
            self._tw.write(frame)
            if self._rh_sim:
                self._rh_sim.feed(frame)
            self._stats["packets"] += 1
            frames.append(frame)

        if frames:
            self.telemetry_batch.emit(frames)
            self.telemetry_updated.emit(frames[-1])

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
                _log.info("Lap %d recorded", self._stats["laps"])
            else:
                _log.debug("Event: %s gate=%s", ev["event_type"], ev.get("gate_id"))
            self.event_fired.emit(ev)

        dur = time.time() - self._start_time
        self._stats["duration"] = dur
        self._stats["dropped"]  = self._udp.dropped if self._udp else 0
        self._stats["hz"] = round(self._stats["packets"] / dur, 1) if dur > 0 else 0.0
        if self._recorder:
            self._stats["frames"] = self._recorder.frames_written
            # 30fps: ScreenRecorder захватывает в своём потоке с 30fps,
            # передаём последний готовый кадр на каждый тик GUI (тоже ~30fps)
            if self._recorder.latest_frame_bgr is not None:
                self.video_frame.emit(self._recorder.latest_frame_bgr)
        self.stats_updated.emit(dict(self._stats))


# ── Replay ────────────────────────────────────────────────────────────────────

class ReplayDataSource(QObject):
    telemetry_updated = pyqtSignal(dict)
    telemetry_batch   = pyqtSignal(list)
    event_fired       = pyqtSignal(dict)
    video_frame       = pyqtSignal(object)         # np.ndarray BGR
    stats_updated     = pyqtSignal(dict)           # зеркало LiveDataSource.stats_updated
    progress_updated  = pyqtSignal(float, float)   # current_s, total_s
    finished          = pyqtSignal()

    def __init__(self, session_dir: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        import pyarrow.parquet as pq
        p = Path(session_dir)
        self._rows: list[dict] = pq.read_table(p / "telemetry.parquet").to_pylist()
        ev_path = p / "events.parquet"
        self._events: list[dict] = pq.read_table(ev_path).to_pylist() if ev_path.exists() else []
        # Кэшируем массив ts_wall — seek() вызывается часто при скраббинге
        self._ts_arr = np.array([r["ts_wall"] for r in self._rows]) if self._rows else np.array([])
        self._ev_ts_arr = np.array([e["ts_wall"] for e in self._events]) if self._events else np.array([])

        # Предвычисляем количество lap-событий для быстрого поиска в _tick()
        self._lap_event_mask = np.array(
            ["lap" in e.get("event_type", "") for e in self._events], dtype=bool
        ) if self._events else np.array([], dtype=bool)
        self._idx = 0
        self._ev_idx = 0
        self._speed = 1.0
        self._paused = True
        self._wall_anchor = 0.0
        self._sim_anchor = 0.0

        # Видео: запускаем фоновый VideoReader (cv2 НЕ в Qt-потоке)
        self._video_reader: VideoReader | None = None
        vid_path = p / "video.mp4"
        vid_ts_path = p / "video_timestamps.parquet"
        if vid_path.exists() and vid_ts_path.exists():
            vid_ts = np.array(pq.read_table(vid_ts_path)["ts_wall"].to_pylist())
            self._video_reader = VideoReader(vid_path, vid_ts)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # 33ms = ~30fps; 16ms было избыточно и перегружало main thread

    @property
    def total_duration(self) -> float:
        if len(self._rows) < 2:
            return 0.0
        return self._rows[-1]["ts_wall"] - self._rows[0]["ts_wall"]

    @property
    def first_ts(self) -> float:
        return self._rows[0]["ts_wall"] if self._rows else 0.0

    def play(self) -> None:
        if self._idx >= len(self._rows):
            self.seek(0.0)
        self._wall_anchor = time.monotonic()
        self._sim_anchor  = self._rows[self._idx]["ts_wall"]
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def toggle_play(self) -> None:
        self.pause() if not self._paused else self.play()

    def set_speed(self, speed: float) -> None:
        if not self._paused and self._idx > 0:
            self._sim_anchor  = self._rows[self._idx - 1]["ts_wall"]
            self._wall_anchor = time.monotonic()
        self._speed = speed

    def seek(self, fraction: float) -> None:
        if not self._rows:
            return
        t0, t1 = self._ts_arr[0], self._ts_arr[-1]
        target = t0 + max(0.0, min(1.0, fraction)) * (t1 - t0)
        # Используем кэшированные массивы — не пересоздаём на каждое движение скраббера
        self._idx = int(np.searchsorted(self._ts_arr, target))
        self._idx = max(0, min(self._idx, len(self._rows) - 1))
        self._ev_idx = int(np.searchsorted(self._ev_ts_arr, target)) if len(self._ev_ts_arr) else 0
        self._sim_anchor  = target
        self._wall_anchor = time.monotonic()
        if self._idx < len(self._rows):
            self.telemetry_updated.emit(self._rows[self._idx])

    def _tick(self) -> None:
        if self._paused or not self._rows:
            return
        target = self._sim_anchor + (time.monotonic() - self._wall_anchor) * self._speed
        batch: list[dict] = []
        # Не берём больше ~50 кадров за тик (0.5s при 100Hz).
        # Если отстали сильнее — просто пропустим данные, но UI не зависнет.
        while self._idx < len(self._rows) and self._rows[self._idx]["ts_wall"] <= target and len(batch) < 50:
            batch.append(self._rows[self._idx])
            self._idx += 1
        # Если сильно отстали — перепрыгиваем вперёд, чтобы не накапливать долг
        if self._idx < len(self._rows) and self._rows[self._idx]["ts_wall"] < target - 0.5:
            self._idx = int(np.searchsorted(self._ts_arr, target - 0.1))
        if batch:
            self.telemetry_batch.emit(batch)
            self.telemetry_updated.emit(batch[-1])
            self._emit_video_frame(batch[-1]["ts_wall"])

        while self._ev_idx < len(self._events) and self._events[self._ev_idx]["ts_wall"] <= target:
            self.event_fired.emit(self._events[self._ev_idx])
            self._ev_idx += 1

        if self._idx >= len(self._rows):
            self._paused = True
            self.progress_updated.emit(self.total_duration, self.total_duration)
            self.finished.emit()
            return

        current = self._rows[self._idx]["ts_wall"] - self._rows[0]["ts_wall"]
        self.progress_updated.emit(current, self.total_duration)

        # Статистика для StatusPanel — воспроизводим то, что было во время записи
        laps_now = int(np.sum(self._lap_event_mask[:self._ev_idx])) if len(self._lap_event_mask) else 0
        hz = round((self._ts_arr[1] - self._ts_arr[0]) ** -1, 1) if len(self._ts_arr) > 1 else 0.0
        # hz берём как медианный интервал по всей сессии (стабильнее мгновенного)
        if len(self._ts_arr) > 10:
            median_dt = float(np.median(np.diff(self._ts_arr)))
            hz = round(1.0 / median_dt, 1) if median_dt > 0 else 0.0
        self.stats_updated.emit({
            "packets":  self._idx,           # пакетов воспроизведено до текущей позиции
            "hz":       hz,                  # реальная частота из записанных данных
            "laps":     laps_now,            # кругов до текущей позиции
            "dropped":  0,                   # при replay дропов нет
            "duration": current,             # текущая позиция в секундах
        })

    def _emit_video_frame(self, ts_wall: float) -> None:
        """Запрашивает кадр у фонового VideoReader (не блокирует)."""
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

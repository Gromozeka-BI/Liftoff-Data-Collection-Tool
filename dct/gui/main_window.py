"""DCT main window — Record / Replay modes in a single QMainWindow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from dct.gui import theme
from dct.gui.data_source import LiveDataSource, ReplayDataSource
from dct.log import get_logger

_log = get_logger("main_window")
from dct.gui.widgets.track_map import TrackMapWidget
from dct.gui.widgets.stick_graphs import StickGraphsWidget
from dct.gui.widgets.video_preview import VideoPreviewWidget
from dct.gui.widgets.record_bar import RecordBar
from dct.gui.widgets.replay_bar import ReplayBar

_MODE_RECORD = 0
_MODE_REPLAY = 1


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DCT — Data Collection Toolkit")
        self.resize(1400, 860)

        self._mode = _MODE_RECORD
        self._live: LiveDataSource | None = None
        self._replay: ReplayDataSource | None = None
        self._lap_count = 0
        self._total_laps = 0

        self._build_ui()
        self._connect_record_bar()
        self._connect_replay_bar()

        # Global shortcuts active in both modes
        QShortcut(QKeySequence("F11"), self, self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._replay_space)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        # Top mode bar
        vbox.addWidget(self._build_mode_bar())

        # Main content area (resizable splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left: track map
        self._map = TrackMapWidget()
        splitter.addWidget(self._map)

        # Right: video + graphs (vertical splitter)
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.setHandleWidth(3)
        self._video = VideoPreviewWidget()
        self._graphs = StickGraphsWidget()
        right_split.addWidget(self._video)
        right_split.addWidget(self._graphs)
        right_split.setSizes([200, 560])
        splitter.addWidget(right_split)

        splitter.setSizes([840, 560])
        vbox.addWidget(splitter, stretch=1)

        # Bottom stacked bar: record / replay
        self._stack = QStackedWidget()
        self._rec_bar = RecordBar()
        self._rep_bar = ReplayBar()
        self._stack.addWidget(self._rec_bar)   # index 0
        self._stack.addWidget(self._rep_bar)   # index 1
        self._stack.setFixedHeight(130)
        vbox.addWidget(self._stack)

    def _build_mode_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(f"background-color: {theme.PANEL}; border-bottom: 1px solid {theme.BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(4)

        self._btn_mode_rec = QPushButton("● RECORD")
        self._btn_mode_rep = QPushButton("▶ REPLAY")
        for btn in (self._btn_mode_rec, self._btn_mode_rep):
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setMinimumWidth(110)

        self._btn_mode_rec.setChecked(True)
        self._btn_mode_rec.clicked.connect(lambda: self._switch_mode(_MODE_RECORD))
        self._btn_mode_rep.clicked.connect(lambda: self._switch_mode(_MODE_REPLAY))

        lay.addWidget(self._btn_mode_rec)
        lay.addWidget(self._btn_mode_rep)
        lay.addStretch()

        self._lbl_session = QLabel("")
        self._lbl_session.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        lay.addWidget(self._lbl_session)
        return bar

    # ── mode switching ─────────────────────────────────────────────────────

    def _switch_mode(self, mode: int) -> None:
        # Запрет переключения во время записи
        if mode == _MODE_REPLAY and self._live and getattr(self._live, '_recording', False):
            return
        self._mode = mode
        self._btn_mode_rec.setChecked(mode == _MODE_RECORD)
        self._btn_mode_rep.setChecked(mode == _MODE_REPLAY)
        self._stack.setCurrentIndex(mode)
        # Очищаем карту и графики при каждом переключении
        self._map.clear_trail()
        self._graphs.clear()
        self._video.clear_frame()
        self._lap_count = 0
        if mode == _MODE_REPLAY:
            self._rep_bar.reload_sessions()

    # ── record bar wiring ──────────────────────────────────────────────────

    def _connect_record_bar(self) -> None:
        rb = self._rec_bar
        rb.start_requested.connect(self._on_start_session)
        rb.stop_requested.connect(self._on_stop_session)
        rb.lap_requested.connect(self._on_mark_lap)
        rb.gate_requested.connect(self._on_mark_nearest_gate)
        rb.sf_requested.connect(self._on_mark_sf_gate)

    @pyqtSlot(dict)
    def _on_start_session(self, cfg: dict) -> None:
        # Load track for the map
        track_path = cfg.get("track_path")
        if track_path:
            try:
                with open(track_path, encoding="utf-8") as f:
                    track_data = json.load(f)
                self._map.setup_track(track_data)
                self._current_track = track_data
            except Exception:
                self._current_track = None
        else:
            self._current_track = None

        self._live = LiveDataSource(self)
        self._live.telemetry_updated.connect(self._on_telemetry)
        self._live.telemetry_batch.connect(self._graphs.update_batch)
        self._live.event_fired.connect(self._on_event)
        self._live.stats_updated.connect(self._rec_bar.status.update_stats)
        self._live.video_frame.connect(self._on_live_video_frame)
        self._live.session_started.connect(self._on_session_started)
        self._live.session_stopped.connect(self._on_session_stopped)

        try:
            self._live.start_session(cfg)
        except RuntimeError as e:
            _log.error("Failed to start session: %s", e)
            QMessageBox.critical(self, "Start error", str(e))
            self._live = None
            return

        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._rec_bar.set_recording(True)

    @pyqtSlot()
    def _on_stop_session(self) -> None:
        if self._live:
            self._live.stop_session()

    @pyqtSlot()
    def _on_mark_lap(self) -> None:
        if self._live:
            self._live.mark_lap()

    @pyqtSlot()
    def _on_mark_nearest_gate(self) -> None:
        if not self._live or not hasattr(self, "_current_track") or not self._current_track:
            return
        latest = getattr(self, "_latest_frame", None)
        if not latest:
            return
        gates = self._current_track.get("gates", [])
        sf_id = next((g["id"] for g in gates if g.get("is_start_finish")), None)
        import math
        best_id, best_d = 0, float("inf")
        for g in gates:
            if g["id"] == sf_id:
                continue
            gx, _gy, gz = g["position"]
            d = math.sqrt((latest["pos_x"]-gx)**2 + (latest["pos_z"]-gz)**2)
            if d < best_d:
                best_d, best_id = d, g["id"]
        self._live.mark_gate(best_id)

    @pyqtSlot()
    def _on_mark_sf_gate(self) -> None:
        if not self._live or not hasattr(self, "_current_track") or not self._current_track:
            return
        gates = self._current_track.get("gates", [])
        sf = next((g for g in gates if g.get("is_start_finish")), None)
        if sf:
            self._live.mark_gate(sf["id"])

    # ── replay bar wiring ──────────────────────────────────────────────────

    def _connect_replay_bar(self) -> None:
        rb = self._rep_bar
        rb.session_selected.connect(self._on_replay_session_selected)
        rb.play_pause.connect(self._on_replay_play_pause)
        rb.seek_fraction.connect(self._on_replay_seek)
        rb.speed_changed.connect(self._on_replay_speed)

    @pyqtSlot(str)
    def _on_replay_session_selected(self, path: str) -> None:
        _log.info("Replay session selected: %s", path)
        p = Path(path)
        track_file = p / "track.json"
        if track_file.exists():
            with open(track_file, encoding="utf-8") as f:
                self._map.setup_track(json.load(f))

        if self._replay:
            self._replay.deleteLater()

        telem = p / "telemetry.parquet"
        if not telem.exists():
            return
        self._replay = ReplayDataSource(p, self)
        self._replay.telemetry_updated.connect(self._on_telemetry)
        self._replay.telemetry_batch.connect(self._graphs.update_batch)
        self._replay.event_fired.connect(self._on_event)
        self._replay.video_frame.connect(self._on_replay_video_frame)
        self._replay.progress_updated.connect(self._rep_bar.update_progress)
        self._replay.finished.connect(lambda: self._rep_bar.set_playing(False))
        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._lbl_session.setText(p.name)

    @pyqtSlot()
    def _on_replay_play_pause(self) -> None:
        if not self._replay:
            return
        self._replay.toggle_play()
        self._rep_bar.set_playing(not self._replay._paused)

    @pyqtSlot(float)
    def _on_replay_seek(self, fraction: float) -> None:
        if self._replay:
            self._replay.seek(fraction)

    @pyqtSlot(float)
    def _on_replay_speed(self, speed: float) -> None:
        if self._replay:
            self._replay.set_speed(speed)

    def _replay_space(self) -> None:
        if self._mode == _MODE_REPLAY:
            self._on_replay_play_pause()

    # ── common slots ───────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_telemetry(self, frame: dict) -> None:
        self._latest_frame = frame
        self._map.update_drone(frame)
        status = self._rec_bar.status if self._mode == _MODE_RECORD else self._rep_bar.status
        status.update_telemetry(frame)

    @pyqtSlot(dict)
    def _on_event(self, ev: dict) -> None:
        if "lap" in ev.get("event_type", ""):
            self._lap_count += 1
            status = self._rec_bar.status if self._mode == _MODE_RECORD else self._rep_bar.status
            self._rep_bar.set_lap(self._lap_count, self._total_laps)

    @pyqtSlot(str)
    def _on_session_started(self, path: str) -> None:
        self._lbl_session.setText(Path(path).name)

    @pyqtSlot(dict)
    def _on_session_stopped(self, result: dict) -> None:
        self._rec_bar.set_recording(False)
        val = result.get("validation")
        if val:
            status_str = "PASSED" if val.passed else "FAILED"
            issues = "\n".join(val.issues) if val.issues else "All checks passed."
            QMessageBox.information(
                self, f"Session validation {status_str}",
                f"Session: {result.get('session_dir', '')}\n\n"
                f"Stats: {val.stats}\n\n{issues}",
            )
        self._live = None

    # ── video frames ───────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_live_video_frame(self, frame) -> None:
        # ScreenRecorder даёт BGR
        self._video.update_frame(frame, is_rgb=False)

    @pyqtSlot(object)
    def _on_replay_video_frame(self, frame) -> None:
        # VideoReader уже конвертировал в RGB в фоновом потоке
        self._video.update_frame(frame, is_rgb=True)

    # ── misc ───────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:
        self._rec_bar.cleanup()  # снимаем global keyboard hooks
        if self._live:
            self._live.stop_session()
        event.accept()

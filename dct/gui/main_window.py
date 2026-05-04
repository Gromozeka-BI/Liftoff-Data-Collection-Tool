"""DCT main window — Record / Replay modes in a single QMainWindow."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

from dct.gui import theme
from dct.gui.data_source import LiveDataSource, ReplayDataSource
from dct.gui.widgets.track_map import TrackMapWidget
from dct.gui.widgets.stick_graphs import StickGraphsWidget
from dct.gui.widgets.video_preview import VideoPreviewWidget
from dct.gui.widgets.record_bar import RecordBar
from dct.gui.widgets.replay_bar import ReplayBar
from dct.video_preview_source import VideoPreviewSource
from dct.log import get_logger

_log = get_logger("main_window")

_MODE_RECORD = 0
_MODE_REPLAY = 1


class MainWindow(QMainWindow):
    _preview_frame_ready = pyqtSignal(object)  # thread-safe bridge for preview frames
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DCT — Data Collection Toolkit")
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(_LOGO_PATH)))
        self.resize(1400, 860)

        self._mode = _MODE_RECORD
        self._live:    LiveDataSource | None     = None
        self._replay:  ReplayDataSource | None   = None
        self._preview: VideoPreviewSource | None = None
        self._lap_count    = 0
        self._total_laps   = 0
        self._current_track: dict | None = None
        self._latest_frame:  dict | None = None

        self._build_ui()
        self._connect_record_bar()
        self._connect_replay_bar()

        QShortcut(QKeySequence("F11"), self, self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._replay_space)

        # Bridge: preview thread → Qt main thread → VideoPreviewWidget
        self._preview_frame_ready.connect(self._on_preview_frame_main)

        # Delay preview start to avoid mouse jitter from mss/DirectX init
        QTimer.singleShot(2000, lambda: self._start_preview(self._rec_bar.current_video_source()))

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        vbox.addWidget(self._build_mode_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        self._map = TrackMapWidget()
        splitter.addWidget(self._map)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.setHandleWidth(3)
        right_split.setMinimumWidth(400)   # prevent graph column from collapsing
        self._video  = VideoPreviewWidget()
        self._graphs = StickGraphsWidget()
        right_split.addWidget(self._video)
        right_split.addWidget(self._graphs)
        right_split.setSizes([200, 560])
        splitter.addWidget(right_split)
        splitter.setSizes([840, 560])
        splitter.setMinimumWidth(780)
        vbox.addWidget(splitter, stretch=1)

        self._stack   = QStackedWidget()
        self._rec_bar = RecordBar()
        self._rep_bar = ReplayBar()
        self._stack.addWidget(self._rec_bar)
        self._stack.addWidget(self._rep_bar)
        self._stack.setFixedHeight(145)
        vbox.addWidget(self._stack)

    def _build_mode_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(f"background-color: {theme.PANEL}; border-bottom: 1px solid {theme.BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(4)

        if _LOGO_PATH.exists():
            logo_lbl = QLabel()
            pix = QPixmap(str(_LOGO_PATH)).scaledToHeight(30, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            lay.addWidget(logo_lbl)

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

    # ── mode switching ──────────────────────────────────────────────────────

    def _switch_mode(self, mode: int) -> None:
        if mode == _MODE_REPLAY and self._live and getattr(self._live, '_recording', False):
            return
        self._mode = mode
        self._btn_mode_rec.setChecked(mode == _MODE_RECORD)
        self._btn_mode_rep.setChecked(mode == _MODE_REPLAY)
        self._stack.setCurrentIndex(mode)
        self._map.clear_trail()
        self._graphs.clear()
        self._video.clear_frame()
        self._lap_count = 0

        if mode == _MODE_RECORD:
            self._start_preview(self._rec_bar.current_video_source())
        else:
            self._stop_preview()
            self._rep_bar.reload_sessions()

    # ── always-on preview ───────────────────────────────────────────────────

    def _start_preview(self, source_cfg: dict) -> None:
        self._stop_preview()
        self._preview = VideoPreviewSource(source_cfg, self._on_preview_frame)
        self._preview.start()

    def _stop_preview(self) -> None:
        if self._preview:
            self._preview.stop()
            self._preview = None

    def _on_preview_frame(self, frame) -> None:
        # Called from preview thread — route through queued signal for thread safety
        self._preview_frame_ready.emit(frame)

    @pyqtSlot(object)
    def _on_preview_frame_main(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=False)

    # ── record bar ──────────────────────────────────────────────────────────

    def _connect_record_bar(self) -> None:
        rb = self._rec_bar
        rb.start_requested.connect(self._on_start_session)
        rb.stop_requested.connect(self._on_stop_session)
        rb.lap_requested.connect(self._on_mark_lap)
        rb.gate_requested.connect(self._on_mark_nearest_gate)
        rb.sf_requested.connect(self._on_mark_sf_gate)
        rb.video_source_changed.connect(self._on_video_source_changed)

    @pyqtSlot(dict)
    def _on_video_source_changed(self, source_cfg: dict) -> None:
        # Restart preview only if not currently recording
        if self._mode == _MODE_RECORD and self._preview is not None:
            self._start_preview(source_cfg)

    @pyqtSlot(dict)
    def _on_start_session(self, cfg: dict) -> None:
        self._stop_preview()   # recorder takes over video

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

        import time as _time
        self._graphs.set_time_zero(_time.time())

        ds_mode = cfg.get("data_source", "liftoff")

        self._live = LiveDataSource(self)
        self._live.telemetry_updated.connect(self._on_telemetry)
        self._live.telemetry_batch.connect(self._graphs.update_batch)
        self._live.rc_batch.connect(self._graphs.update_rc_batch)
        self._live.rc_status_changed.connect(self._rec_bar.set_rc_status)
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
            self._start_preview(self._rec_bar.current_video_source())
            return

        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._rec_bar.set_recording(True)

    @pyqtSlot()
    def _on_stop_session(self) -> None:
        if self._live:
            live, self._live = self._live, None   # clear immediately — prevents double stop from closeEvent
            live.stop_session()

    @pyqtSlot()
    def _on_mark_lap(self) -> None:
        if self._live:
            self._live.mark_lap()

    @pyqtSlot()
    def _on_mark_nearest_gate(self) -> None:
        if not self._live or not self._current_track:
            return
        latest = self._latest_frame
        if not latest:
            return
        import math
        gates = self._current_track.get("gates", [])
        sf_id = next((g["id"] for g in gates if g.get("is_start_finish")), None)
        best_id, best_d = 0, float("inf")
        for g in gates:
            if g["id"] == sf_id:
                continue
            gx, _gy, gz = g["position"]
            d = math.sqrt((latest["pos_x"] - gx) ** 2 + (latest["pos_z"] - gz) ** 2)
            if d < best_d:
                best_d, best_id = d, g["id"]
        self._live.mark_gate(best_id)

    @pyqtSlot()
    def _on_mark_sf_gate(self) -> None:
        if not self._live or not self._current_track:
            return
        gates = self._current_track.get("gates", [])
        sf    = next((g for g in gates if g.get("is_start_finish")), None)
        if sf:
            self._live.mark_gate(sf["id"])

    # ── replay bar ──────────────────────────────────────────────────────────

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

        # Restore invert state saved during recording
        invert_path = p / "invert.json"
        if invert_path.exists():
            import json as _json
            try:
                self._graphs.set_invert_state(
                    _json.loads(invert_path.read_text(encoding="utf-8"))
                )
            except Exception:
                pass

        rc_exists = (p / "rc_channels.parquet").exists()
        tl_exists = (p / "timeline.parquet").exists()
        if not (p / "telemetry.parquet").exists() and not tl_exists and not rc_exists:
            return

        self._replay = ReplayDataSource(p, self)
        if self._replay.first_ts > 0:
            self._graphs.set_time_zero(self._replay.first_ts)

        self._replay.telemetry_updated.connect(self._on_telemetry)
        self._replay.telemetry_batch.connect(self._graphs.update_batch)
        self._replay.rc_batch.connect(self._graphs.update_rc_batch)
        self._replay.event_fired.connect(self._on_event)
        self._replay.stats_updated.connect(self._rep_bar.status.update_stats)
        self._replay.video_frame.connect(self._on_replay_video_frame)
        self._replay.progress_updated.connect(self._rep_bar.update_progress)
        self._replay.finished.connect(lambda: self._rep_bar.set_playing(False))
        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._lbl_session.setText(p.name)

        # Show first frame without pressing Play
        self._replay.seek(0.0)

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
            self._map.clear_trail()
            self._graphs.clear()

    @pyqtSlot(float)
    def _on_replay_speed(self, speed: float) -> None:
        if self._replay:
            self._replay.set_speed(speed)

    def _replay_space(self) -> None:
        if self._mode == _MODE_REPLAY:
            self._on_replay_play_pause()

    # ── common slots ────────────────────────────────────────────────────────

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
            self._rep_bar.set_lap(self._lap_count, self._total_laps)

    @pyqtSlot(str)
    def _on_session_started(self, path: str) -> None:
        self._lbl_session.setText(Path(path).name)
        # Save current invert state into session dir so Replay can restore it
        import json as _json
        try:
            invert = self._graphs.get_invert_state()
            (Path(path) / "invert.json").write_text(
                _json.dumps(invert, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    @pyqtSlot(dict)
    def _on_session_stopped(self, result: dict) -> None:
        self._rec_bar.set_recording(False)
        val = result.get("validation")
        if val:
            status_str = "PASSED" if val.passed else "FAILED"
            issues     = "\n".join(val.issues) if val.issues else "All checks passed."
            QMessageBox.information(
                self, f"Session validation {status_str}",
                f"Session: {result.get('session_dir', '')}\n\n"
                f"Stats: {val.stats}\n\n{issues}",
            )
        self._live = None
        # Resume live preview after recording ends
        self._start_preview(self._rec_bar.current_video_source())

    @pyqtSlot(object)
    def _on_live_video_frame(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=False)

    @pyqtSlot(object)
    def _on_replay_video_frame(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=True)

    # ── misc ────────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:
        self._stop_preview()
        self._rec_bar.cleanup()
        if self._live:
            self._live.stop_session()
        event.accept()

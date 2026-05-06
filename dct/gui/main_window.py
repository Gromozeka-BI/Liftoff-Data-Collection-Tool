"""DCT main window — Record / Replay / Race in a single QMainWindow.

Uses the GUI 2.0 layout: TopBar + content splitter (Map | Video+Sticks | Sidebar)
and a BottomStrip whose contents switch between Record buttons and the Replay
event editor.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QByteArray, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QGuiApplication, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget,
)

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

from dct.gui import theme, ui_settings
from dct.gui.data_source import LiveDataSource, ReplayDataSource
from dct.gui.widgets.bottom_strip import BottomStrip
from dct.gui.widgets.sidebar import PAGE_REPLAY, PAGE_SETUP, Sidebar
from dct.gui.widgets.replay_page import ReplayPage
from dct.gui.widgets.setup_page import SetupPage
from dct.gui.widgets.stick_graphs import StickGraphsWidget
from dct.gui.widgets.top_bar import MODE_RACE, MODE_RECORD, MODE_REPLAY, TopBar
from dct.gui.widgets.track_map import TrackMapWidget
from dct.gui.widgets.video_pip import VideoPiP
from dct.gui.widgets.video_preview import VideoPreviewWidget
from dct.localization import OnlineLocalizer
from dct.video_preview_source import VideoPreviewSource
from dct.log import get_logger

_log = get_logger("main_window")


def _b64(qb: QByteArray) -> str:
    return bytes(qb.toBase64()).decode("ascii")


def _from_b64(s: str) -> QByteArray:
    return QByteArray.fromBase64(s.encode("ascii"))


class MainWindow(QMainWindow):
    _preview_frame_ready = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DCT — Data Collection Toolkit")
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(_LOGO_PATH)))
        self.setMinimumSize(1080, 720)

        self._mode = MODE_RECORD
        self._prev_mode = MODE_RECORD
        self._race_active = False

        self._live: LiveDataSource | None = None
        self._replay: ReplayDataSource | None = None
        self._preview: VideoPreviewSource | None = None
        self._starting_session: bool = False
        self._lap_count = 0
        self._total_laps = 0
        self._current_track: dict | None = None
        self._latest_frame: dict | None = None
        self._localizer: OnlineLocalizer | None = None
        self._prev_ts_wall: float | None = None
        self._last_loc: tuple[float, float] | None = None
        self._last_replay_path: str | None = None
        self._race_pip: VideoPiP | None = None

        self._build_ui()
        self._connect_signals()
        self._restore_window_state()

        QShortcut(QKeySequence("F11"), self, self._toggle_race_mode)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._exit_race_mode)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._replay_space)

        self._preview_frame_ready.connect(self._on_preview_frame_main)
        QTimer.singleShot(2000, lambda: self._start_preview(self._setup_page.current_video_source()))

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._top_bar = TopBar()
        vbox.addWidget(self._top_bar)

        # Main splitter: content | sidebar
        self._main_split = QSplitter(Qt.Orientation.Horizontal)
        self._main_split.setObjectName("split_main")
        self._main_split.setChildrenCollapsible(False)
        self._main_split.setHandleWidth(3)

        # Content splitter: map | right column
        self._content_split = QSplitter(Qt.Orientation.Horizontal)
        self._content_split.setObjectName("split_content")
        self._content_split.setChildrenCollapsible(False)
        self._content_split.setHandleWidth(3)

        self._map = TrackMapWidget()
        self._map.setMinimumWidth(360)
        self._content_split.addWidget(self._map)

        self._right_col = QSplitter(Qt.Orientation.Vertical)
        self._right_col.setObjectName("split_right")
        self._right_col.setChildrenCollapsible(False)
        self._right_col.setHandleWidth(3)
        self._right_col.setMinimumWidth(360)
        self._video = VideoPreviewWidget()
        self._graphs = StickGraphsWidget()
        self._right_col.addWidget(self._video)
        self._right_col.addWidget(self._graphs)
        self._right_col.setSizes([220, 580])
        self._content_split.addWidget(self._right_col)
        self._content_split.setStretchFactor(0, 3)
        self._content_split.setStretchFactor(1, 2)
        self._content_split.setSizes([900, 600])

        self._main_split.addWidget(self._content_split)

        # Sidebar
        self._sidebar = Sidebar()
        self._setup_page = SetupPage()
        self._replay_page = ReplayPage()
        self._sidebar.add_page(self._setup_page)
        self._sidebar.add_page(self._replay_page)
        self._main_split.addWidget(self._sidebar)
        self._main_split.setStretchFactor(0, 1)
        self._main_split.setStretchFactor(1, 0)
        self._main_split.setSizes([1360, 160])

        vbox.addWidget(self._main_split, stretch=1)

        # Bottom strip
        self._bottom = BottomStrip()
        vbox.addWidget(self._bottom)

        self._is_portrait = False

    # ── connections ────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._top_bar.mode_changed.connect(self._switch_mode)
        self._top_bar.summary_clicked.connect(self._on_summary_clicked)
        self._top_bar.toggle_sidebar.connect(self._sidebar.toggle)

        self._sidebar.collapsed_changed.connect(self._top_bar.set_sidebar_collapsed)
        self._sidebar.page_changed.connect(self._on_sidebar_page_changed)

        self._setup_page.video_source_changed.connect(self._on_video_source_changed)
        self._setup_page.cfg_changed.connect(self._refresh_summary)
        self._setup_page.localizer_settings_changed.connect(self._on_loc_show_changed)
        self._setup_page.reset_filter_button().clicked.connect(self._reset_localizer)

        self._replay_page.session_selected.connect(self._on_replay_session_selected)

        # Bottom strip — record
        self._bottom.start_clicked.connect(self._on_start_clicked)
        self._bottom.stop_clicked.connect(self._on_stop_session)
        self._bottom.lap_clicked.connect(self._on_mark_lap)
        self._bottom.gate_clicked.connect(self._on_mark_nearest_gate)
        self._bottom.sf_clicked.connect(self._on_mark_sf_gate)

        # Bottom strip — replay
        self._bottom.play_pause.connect(self._on_replay_play_pause)
        self._bottom.seek_fraction.connect(self._on_replay_seek)
        self._bottom.speed_changed.connect(self._on_replay_speed)
        self._bottom.event_add_requested.connect(self._on_replay_event_add)
        self._bottom.event_delete_requested.connect(self._on_replay_event_delete)
        self._bottom.event_seek_requested.connect(self._on_replay_event_seek)
        self._bottom.event_drag_started.connect(self._on_replay_drag_started)
        self._bottom.event_drag_ended.connect(self._on_replay_drag_ended)
        self._bottom.event_inline_apply.connect(self._on_replay_inline_apply)
        self._bottom.nudge_frame_requested.connect(self._on_nudge_frame)
        self._bottom.nudge_ms_requested.connect(self._on_nudge_ms)

        self._refresh_summary()

    # ── mode switching ─────────────────────────────────────────────────────

    def _switch_mode(self, mode: int) -> None:
        if mode == MODE_RACE:
            self._toggle_race_mode()
            return
        if self._race_active:
            self._exit_race_mode()
        if mode == MODE_REPLAY and self._live and getattr(self._live, "_recording", False):
            self._top_bar.set_mode(MODE_RECORD)
            return
        self._mode = mode
        self._prev_mode = mode
        self._top_bar.set_mode(mode)

        if mode == MODE_RECORD:
            self._sidebar.set_page(PAGE_SETUP)
            self._bottom.set_record_mode()
            self._start_preview(self._setup_page.current_video_source())
        else:
            self._sidebar.set_page(PAGE_REPLAY)
            self._bottom.set_replay_mode()
            self._stop_preview()
            self._replay_page.reload_sessions()
            self._deactivate_localizer_full()

        self._map.clear_trail()
        self._graphs.clear()
        self._video.clear_frame()
        self._lap_count = 0

    def _on_sidebar_page_changed(self, idx: int) -> None:
        # If the user clicks a sidebar tab, stay in sync with main mode.
        if idx == PAGE_SETUP and self._mode != MODE_RECORD:
            self._switch_mode(MODE_RECORD)
        elif idx == PAGE_REPLAY and self._mode != MODE_REPLAY:
            self._switch_mode(MODE_REPLAY)

    def _on_summary_clicked(self) -> None:
        if self._sidebar.is_collapsed():
            self._sidebar.set_collapsed(False)
        self._sidebar.set_page(PAGE_SETUP)

    # ── preview ────────────────────────────────────────────────────────────

    def _start_preview(self, source_cfg: dict) -> None:
        self._stop_preview()
        try:
            self._preview = VideoPreviewSource(source_cfg, self._on_preview_frame)
            self._preview.start()
        except Exception as exc:
            _log.warning("Preview start failed: %s", exc)
            self._preview = None

    def _stop_preview(self) -> None:
        if self._preview:
            self._preview.stop()
            self._preview = None

    def _on_preview_frame(self, frame) -> None:
        self._preview_frame_ready.emit(frame)

    @pyqtSlot(object)
    def _on_preview_frame_main(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=False)
        if self._race_pip is not None:
            self._race_pip.update_frame(frame, is_rgb=False)

    @pyqtSlot(dict)
    def _on_video_source_changed(self, source_cfg: dict) -> None:
        if self._mode == MODE_RECORD and self._preview is not None:
            self._start_preview(source_cfg)

    # ── localizer ──────────────────────────────────────────────────────────

    def _teardown_localizer(self) -> None:
        self._localizer = None
        self._prev_ts_wall = None
        self._last_loc = None
        self._map.clear_localizer_overlay()
        self._map.update_hud(self._latest_frame, None)

    def _deactivate_localizer_full(self) -> None:
        self._teardown_localizer()
        self._map.clear_reference_path()

    def _init_localizer_from_cfg(self, cfg: dict) -> None:
        self._teardown_localizer()
        if not cfg.get("localizer_enabled"):
            self._map.clear_reference_path()
            return
        path = cfg.get("localizer_reference_path")
        if not path:
            self._map.clear_reference_path()
            return
        p = Path(str(path))
        if not p.is_file():
            QMessageBox.warning(self, "Localizer", f"Файл не найден:\n{p}")
            self._map.clear_reference_path()
            return
        try:
            data = np.load(p, allow_pickle=False)
            pos = data["pos"]
            self._map.set_reference_path(pos[:, 0], pos[:, 2])
            self._localizer = OnlineLocalizer.from_file(p)
            self._localizer.reset()
            self._prev_ts_wall = None
        except Exception as exc:
            _log.error("Localizer init failed: %s", exc)
            QMessageBox.warning(self, "Localizer", f"Не удалось загрузить эталон:\n{exc}")
            self._map.clear_reference_path()
            self._localizer = None

    def _reset_localizer(self) -> None:
        if self._localizer is None:
            return
        try:
            self._localizer.reset()
        except Exception:
            pass
        self._prev_ts_wall = None
        self._map.clear_localizer_overlay()
        self._last_loc = None
        self._map.update_hud(self._latest_frame, None)

    def _on_loc_show_changed(self) -> None:
        s = self._setup_page.localizer_show_state()
        self._map.set_reference_path_visible(s["path"])
        self._map.set_localizer_arrow_visible(s["arrow"])
        self._map.set_localizer_trail_visible(s["trail"])

    # ── record session ─────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_start_clicked(self) -> None:
        if self._mode != MODE_RECORD:
            return
        if self._live is not None or self._starting_session:
            _log.debug(
                "Start request ignored — session already running or starting",
            )
            return
        cfg = self._setup_page.build_cfg()
        if cfg is None:
            return
        self._on_start_session(cfg)

    @pyqtSlot(dict)
    def _on_start_session(self, cfg: dict) -> None:
        if self._mode != MODE_RECORD:
            return
        if self._live is not None or self._starting_session:
            _log.warning(
                "Start request ignored — session already running or starting",
            )
            return
        self._starting_session = True
        try:
            self._do_start_session(cfg)
        finally:
            self._starting_session = False

    def _do_start_session(self, cfg: dict) -> None:
        self._stop_preview()

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

        self._init_localizer_from_cfg(cfg)
        self._on_loc_show_changed()

        import time as _time
        self._graphs.set_time_zero(_time.time())

        ds_mode = cfg.get("data_source", "liftoff")
        self._live = LiveDataSource(self)
        self._live.telemetry_updated.connect(self._on_telemetry)
        self._live.telemetry_batch.connect(self._graphs.update_batch)
        self._live.rc_batch.connect(self._graphs.update_rc_batch)
        self._live.rc_status_changed.connect(self._setup_page.set_rc_status)
        self._live.event_fired.connect(self._on_event)
        self._live.stats_updated.connect(self._on_stats)
        self._live.video_frame.connect(self._on_live_video_frame)
        self._live.session_started.connect(self._on_session_started)
        self._live.session_stopped.connect(self._on_session_stopped)

        try:
            self._live.start_session(cfg)
        except RuntimeError as exc:
            _log.error("Failed to start session: %s", exc)
            QMessageBox.critical(self, "Start error", str(exc))
            partial, self._live = self._live, None
            try:
                if partial is not None:
                    partial.stop_session()
            except Exception:  # noqa: BLE001
                _log.exception("Failed to clean up partially-started session")
            self._deactivate_localizer_full()
            self._start_preview(self._setup_page.current_video_source())
            return

        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._setup_page.set_recording(True)
        self._top_bar.set_recording(True)
        self._bottom.set_recording(True)

    @pyqtSlot()
    def _on_stop_session(self) -> None:
        if self._live:
            live, self._live = self._live, None
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
        sf = next((g for g in gates if g.get("is_start_finish")), None)
        if sf:
            self._live.mark_sf_lap(sf["id"])

    # ── replay session ─────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_replay_session_selected(self, path: str) -> None:
        if path == self._last_replay_path:
            return
        self._last_replay_path = path
        _log.info("Replay session selected: %s", path)
        p = Path(path)

        track_file = p / "track.json"
        if track_file.exists():
            with open(track_file, encoding="utf-8") as f:
                track_data = json.load(f)
            self._map.setup_track(track_data)
            self._current_track = track_data
        else:
            self._current_track = None

        if self._replay is not None:
            try:
                self._replay.deleteLater()
            except RuntimeError:
                pass
            self._replay = None

        invert_path = p / "invert.json"
        if invert_path.exists():
            try:
                self._graphs.set_invert_state(
                    json.loads(invert_path.read_text(encoding="utf-8")),
                )
            except Exception:
                pass

        if not (p / "telemetry.parquet").exists() \
           and not (p / "timeline.parquet").exists() \
           and not (p / "rc_channels.parquet").exists():
            return

        self._replay = ReplayDataSource(p, self)
        if self._replay.first_ts > 0:
            self._graphs.set_time_zero(self._replay.first_ts)

        self._replay.events_changed.connect(self._on_replay_events_changed)
        t0 = self._replay.first_ts
        t1 = t0 + self._replay.total_duration
        replay_ctrl = self._bottom.replay_controls()
        replay_ctrl.set_session_time_range(t0, t1)
        replay_ctrl.set_events(self._replay._events)
        replay_ctrl.set_snap_sources(
            tl_ts=self._replay._tl_ts,
            rc_ts=self._replay._rc_ts_arr,
        )
        if self._current_track:
            replay_ctrl.set_gates(self._current_track.get("gates", []))

        self._replay.telemetry_updated.connect(self._on_telemetry)
        self._replay.telemetry_batch.connect(self._graphs.update_batch)
        self._replay.rc_batch.connect(self._graphs.update_rc_batch)
        self._replay.event_fired.connect(self._on_event)
        self._replay.stats_updated.connect(self._on_stats)
        self._replay.video_frame.connect(self._on_replay_video_frame)
        self._replay.progress_updated.connect(replay_ctrl.update_progress)
        self._replay.finished.connect(lambda: replay_ctrl.set_playing(False))

        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._top_bar.set_summary(p.name)

        self._replay.seek(0.0)

    @pyqtSlot()
    def _on_replay_play_pause(self) -> None:
        if not self._replay:
            return
        self._replay.toggle_play()
        self._bottom.replay_controls().set_playing(not self._replay._paused)

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

    @pyqtSlot(str, int)
    def _on_replay_event_add(self, event_type: str, gate_id: int) -> None:
        if not self._replay:
            return
        self._replay.pause()
        self._bottom.replay_controls().set_playing(False)
        ts = self._replay.current_ts
        self._replay.add_event(event_type, ts, gate_id=gate_id)

    @pyqtSlot(dict)
    def _on_replay_event_delete(self, ev: dict) -> None:
        if not self._replay:
            return
        self._replay.delete_event(ev.get("seq", -1))

    @pyqtSlot(float)
    def _on_replay_event_seek(self, ts_wall: float) -> None:
        if not self._replay:
            return
        self._replay.seek_to_ts(ts_wall)
        self._map.clear_trail()
        self._graphs.clear()

    @pyqtSlot(dict)
    def _on_replay_drag_started(self, _ev: dict) -> None:
        if not self._replay:
            return
        self._replay.pause()
        self._bottom.replay_controls().set_playing(False)

    @pyqtSlot(int, float)
    def _on_replay_drag_ended(self, seq: int, ts_wall: float) -> None:
        if not self._replay:
            return
        self._replay.update_event(seq, ts_wall=ts_wall)
        self._replay.seek_to_ts(ts_wall)

    @pyqtSlot(int, str, float, int)
    def _on_replay_inline_apply(self, seq: int, etype: str, ts_wall: float, gate_id: int) -> None:
        if not self._replay:
            return
        self._replay.update_event(
            seq,
            ts_wall=ts_wall,
            event_type=etype,
            gate_id=gate_id,
        )

    @pyqtSlot(int)
    def _on_nudge_frame(self, delta: int) -> None:
        if not self._replay:
            return
        sel = self._bottom.replay_controls().selected_event()
        if sel is None:
            return
        seq = int(sel.get("seq", -1))
        ts = float(sel.get("ts_wall", 0.0))
        ts_arr = self._replay._tl_ts
        if len(ts_arr) == 0:
            return
        idx = int(np.searchsorted(ts_arr, ts))
        idx = max(0, min(len(ts_arr) - 1, idx + delta))
        new_ts = float(ts_arr[idx])
        self._replay.update_event(seq, ts_wall=new_ts)
        self._replay.seek_to_ts(new_ts)

    @pyqtSlot(int)
    def _on_nudge_ms(self, delta_ms: int) -> None:
        if not self._replay:
            return
        sel = self._bottom.replay_controls().selected_event()
        if sel is None:
            return
        seq = int(sel.get("seq", -1))
        ts = float(sel.get("ts_wall", 0.0)) + delta_ms / 1000.0
        self._replay.update_event(seq, ts_wall=ts)
        self._replay.seek_to_ts(ts)

    def _on_replay_events_changed(self, events: list) -> None:
        self._bottom.replay_controls().set_events(events)

    def _replay_space(self) -> None:
        if self._mode == MODE_REPLAY:
            self._on_replay_play_pause()

    # ── common ─────────────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_telemetry(self, frame: dict) -> None:
        self._latest_frame = frame
        self._map.update_drone(frame)
        if self._mode == MODE_RECORD and self._localizer is not None:
            ts = float(frame["ts_wall"])
            prev = self._prev_ts_wall
            dt = (ts - prev) if prev is not None else None
            if dt is not None and (dt < 0 or dt > 2.0):
                dt = None
            self._prev_ts_wall = ts
            try:
                res = self._localizer.update(
                    [
                        frame["in_throttle"],
                        frame["in_yaw"],
                        frame["in_pitch"],
                        frame["in_roll"],
                    ],
                    dt,
                )
                self._map.update_localizer_estimate(
                    float(res.position_xyz[0]),
                    float(res.position_xyz[2]),
                )
                self._last_loc = (float(res.progress), float(res.uncertainty_m))
            except Exception as exc:
                _log.warning("Localizer update failed: %s", exc)
        self._map.update_hud(frame, self._last_loc)

    @pyqtSlot(dict)
    def _on_event(self, ev: dict) -> None:
        if "lap" in ev.get("event_type", ""):
            self._lap_count += 1
            self._bottom.replay_controls().set_lap(self._lap_count, self._total_laps)

    @pyqtSlot(dict)
    def _on_stats(self, stats: dict) -> None:
        self._top_bar.set_stats(
            duration=float(stats.get("duration", 0.0)),
            hz=float(stats.get("hz", 0.0)),
        )

    @pyqtSlot(str)
    def _on_session_started(self, path: str) -> None:
        self._top_bar.set_summary(Path(path).name)
        try:
            invert = self._graphs.get_invert_state()
            (Path(path) / "invert.json").write_text(
                json.dumps(invert, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    @pyqtSlot(dict)
    def _on_session_stopped(self, result: dict) -> None:
        self._setup_page.set_recording(False)
        self._top_bar.set_recording(False)
        self._bottom.set_recording(False)

        val = result.get("validation")
        if val:
            errors = [i for i in val.issues if not i.startswith("WARN:")]
            warns = [i for i in val.issues if i.startswith("WARN:")]
            if not val.passed:
                body = "\n".join(errors)
                if warns:
                    body += "\n\n" + "\n".join(warns)
                QMessageBox.warning(
                    self, "Session validation FAILED",
                    f"Session: {result.get('session_dir', '')}\n\n"
                    f"Stats: {val.stats}\n\n{body}",
                )
            elif warns:
                QMessageBox.warning(
                    self, "Session validation PASSED (with warnings)",
                    f"Session: {result.get('session_dir', '')}\n\n"
                    f"Stats: {val.stats}\n\n" + "\n".join(warns),
                )
            else:
                QMessageBox.information(
                    self, "Session validation PASSED",
                    f"Session: {result.get('session_dir', '')}\n\n"
                    f"Stats: {val.stats}\n\nAll checks passed.",
                )
        self._live = None
        self._deactivate_localizer_full()
        self._start_preview(self._setup_page.current_video_source())

    @pyqtSlot(object)
    def _on_live_video_frame(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=False)
        if self._race_pip is not None:
            self._race_pip.update_frame(frame, is_rgb=False)

    @pyqtSlot(object)
    def _on_replay_video_frame(self, frame) -> None:
        self._video.update_frame(frame, is_rgb=True)
        if self._race_pip is not None:
            self._race_pip.update_frame(frame, is_rgb=True)

    def _refresh_summary(self) -> None:
        self._top_bar.set_summary(self._setup_page.summary_text())

    # ── race mode ──────────────────────────────────────────────────────────

    def _toggle_race_mode(self) -> None:
        self._set_race_mode(not self._race_active)

    def _exit_race_mode(self) -> None:
        if self._race_active:
            self._set_race_mode(False)

    def _set_race_mode(self, on: bool) -> None:
        if on == self._race_active:
            return
        if on:
            self._prev_mode = self._mode
            self._sidebar.setVisible(False)
            self._bottom.setVisible(False)
            self._top_bar.setVisible(False)
            self._right_col.setVisible(False)
            self._map.set_hud_race_mode(True)
            if self._race_pip is None:
                self._race_pip = VideoPiP(self.centralWidget())
                self._race_pip.show()
                self._race_pip.raise_()
            self._top_bar.set_mode(MODE_RACE)
            self.showFullScreen()
        else:
            self.showNormal()
            self._sidebar.setVisible(not self._sidebar.is_collapsed())
            self._bottom.setVisible(True)
            self._top_bar.setVisible(True)
            self._right_col.setVisible(True)
            self._map.set_hud_race_mode(False)
            if self._race_pip is not None:
                self._race_pip.hide()
                self._race_pip.deleteLater()
                self._race_pip = None
            self._top_bar.set_mode(self._prev_mode)
        self._race_active = on

    # ── persistence ────────────────────────────────────────────────────────

    def _restore_window_state(self) -> None:
        s = ui_settings.load()
        win = s.get("window") or {}
        geom = win.get("geometry")
        if geom:
            try:
                self.restoreGeometry(_from_b64(geom))
            except Exception:
                pass
        # Verify on-screen
        screens = QGuiApplication.screens()
        if not any(self.geometry().intersects(scr.geometry()) for scr in screens):
            self.resize(1400, 860)
            self.move(100, 100)

        for split, key in (
            (self._main_split, "split_main"),
            (self._content_split, "split_content"),
            (self._right_col, "split_right"),
        ):
            state = win.get(key)
            if state:
                try:
                    split.restoreState(_from_b64(state))
                except Exception:
                    pass

        sb = s.get("sidebar") or {}
        if sb.get("collapsed"):
            self._sidebar.set_collapsed(True)
            self._top_bar.set_sidebar_collapsed(True)

    def _save_window_state(self) -> None:
        s = ui_settings.load()
        win = s.setdefault("window", {})
        try:
            win["geometry"] = _b64(self.saveGeometry())
        except Exception:
            pass
        for split, key in (
            (self._main_split, "split_main"),
            (self._content_split, "split_content"),
            (self._right_col, "split_right"),
        ):
            try:
                win[key] = _b64(split.saveState())
            except Exception:
                pass
        screen = self.screen()
        if screen is not None:
            win["screen_name"] = screen.name()
        sb = s.setdefault("sidebar", {})
        sb["collapsed"] = bool(self._sidebar.is_collapsed())
        sb["width"] = int(self._sidebar.width())
        ui_settings.save(s)

    # ── adaptive layout ────────────────────────────────────────────────────

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        portrait = self.height() > self.width()
        if portrait != self._is_portrait:
            self._is_portrait = portrait
            self._apply_orientation(portrait)

    def _apply_orientation(self, portrait: bool) -> None:
        if portrait:
            self._content_split.setOrientation(Qt.Orientation.Vertical)
            total_h = max(400, self._content_split.height() or 800)
            self._content_split.setSizes([int(total_h * 0.6), int(total_h * 0.4)])
            self._right_col.setOrientation(Qt.Orientation.Horizontal)
        else:
            self._content_split.setOrientation(Qt.Orientation.Horizontal)
            total_w = max(800, self._content_split.width() or 1200)
            self._content_split.setSizes([int(total_w * 0.6), int(total_w * 0.4)])
            self._right_col.setOrientation(Qt.Orientation.Vertical)

    # ── close ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        try:
            self._save_window_state()
        except Exception:
            pass
        self._stop_preview()
        self._bottom.cleanup()
        if self._live:
            self._live.stop_session()
        event.accept()

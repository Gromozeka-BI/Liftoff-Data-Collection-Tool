"""DCT main window — Record / Replay / Race in a single QMainWindow.

Uses the GUI 2.0 layout: TopBar + content splitter (Map | Video+Sticks | Sidebar)
and a BottomStrip whose contents switch between Record buttons and the Replay
event editor.
"""
from __future__ import annotations

import json
import logging
import time
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
from dct.gui.widgets.stick_graphs import (
    StickGraphsWidget,
    lf_sticks_with_invert,
    rc_frame_to_sticks_norm,
)
from dct.gui.widgets.top_bar import MODE_RACE, MODE_RECORD, MODE_REPLAY, TopBar
from dct.gui.widgets.track_map import TrackMapWidget
from dct.gui.widgets.video_pip import VideoPiP
from dct.gui.widgets.video_preview import VideoPreviewWidget
from dct.localization import OnlineLocalizer
from dct.localization import reference_builder as refbuild
from dct.localization.kf_layer2 import KFLayer2
from dct.rate_features import FEATURE_BETAFLIGHT_CLASSIC_V1
from dct.session import load_meta
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
        self._localizer_rc: OnlineLocalizer | None = None
        self._localizer_legacy: OnlineLocalizer | None = None
        self._kf_layer2: KFLayer2 | None = None
        self._prev_ts_wall: float | None = None
        self._prev_rc_ts_wall: float | None = None
        self._prev_kf_ts_wall: float | None = None
        self._record_ds_mode: str = "liftoff"
        self._current_rate_profile: dict | None = None  # rate profile of current session
        self._last_loc: tuple[float, float] | None = None
        self._dual_loc_log_mono: float = 0.0
        self._last_lf_sticks: list[float] | None = None
        self._last_rc_sticks: list[float] | None = None
        self._last_lf_loc: tuple[float, float, float, float] | None = None  # x,z,prog,sig
        self._last_rc_loc: tuple[float, float, float, float] | None = None
        self._last_legacy_loc: tuple[float, float, float, float] | None = None
        self._last_kf_loc: tuple[float, float, float, float] | None = None
        self._last_replay_path: str | None = None
        self._race_pip: VideoPiP | None = None

        self._build_ui()
        self._connect_signals()
        self._restore_window_state()
        self._propagate_invert_to_pages()

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
        self._replay_page.localizer_settings_changed.connect(self._on_replay_loc_settings_changed)
        self._replay_page.reset_filter_button().clicked.connect(self._reset_localizer)

        # Keep pages in sync with invert checkboxes so dialogs use the right convention
        self._graphs.invert_changed.connect(self._on_invert_changed)

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
            self._teardown_localizer()
            # Re-init localizer immediately if a session was already selected.
            if self._last_replay_path:
                self._try_init_localizer_for_replay(Path(self._last_replay_path))

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
        # Discard preview frames that arrive (via the Qt event queue) after we
        # have already switched away from Record mode — they would compete with
        # replay video and cause flickering.
        if self._mode != MODE_RECORD:
            return
        self._video.update_frame(frame, is_rgb=False)
        if self._race_pip is not None:
            self._race_pip.update_frame(frame, is_rgb=False)

    @pyqtSlot(dict)
    def _on_video_source_changed(self, source_cfg: dict) -> None:
        if self._mode == MODE_RECORD and self._preview is not None:
            self._start_preview(source_cfg)

    @pyqtSlot(dict)
    def _on_invert_changed(self, state: dict) -> None:
        """Propagate LF invert state to pages so reference dialogs use the same convention."""
        inv_lf = state.get("lf", {})
        self._setup_page.set_invert_lf(inv_lf)
        self._replay_page.set_invert_lf(inv_lf)

    def _propagate_invert_to_pages(self) -> None:
        """Push current invert state to pages (called after set_invert_state bypasses signal)."""
        inv_lf = self._graphs.get_invert_state().get("lf", {})
        self._setup_page.set_invert_lf(inv_lf)
        self._replay_page.set_invert_lf(inv_lf)

    # ── localizer ──────────────────────────────────────────────────────────

    def _teardown_localizer(self) -> None:
        self._localizer = None
        self._localizer_rc = None
        self._localizer_legacy = None
        self._kf_layer2 = None
        self._prev_ts_wall = None
        self._prev_rc_ts_wall = None
        self._prev_kf_ts_wall = None
        self._current_rate_profile = None
        self._last_loc = None
        self._dual_loc_log_mono = 0.0
        self._last_lf_sticks = None
        self._last_rc_sticks = None
        self._last_lf_loc = None
        self._last_rc_loc = None
        self._last_legacy_loc = None
        self._last_kf_loc = None
        self._map.clear_localizer_overlay()
        self._map.update_hud(self._latest_frame, None, has_gt=(self._latest_frame is not None))

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
            p_bf_npz, p_leg_npz = refbuild.resolve_bf_and_legacy_npz(p)

            def _load_pos(npz: Path) -> np.ndarray:
                with np.load(npz, allow_pickle=False) as d:
                    return d["pos"]

            pos_src: Path | None = None
            if p_bf_npz is not None and refbuild.npz_feature_kind(p_bf_npz) == FEATURE_BETAFLIGHT_CLASSIC_V1:
                pos_src = p_bf_npz
            elif p_leg_npz is not None:
                pos_src = p_leg_npz
            elif p_bf_npz is not None:
                pos_src = p_bf_npz
            else:
                raise RuntimeError("Не удалось определить эталон по выбранному файлу")
            pos = _load_pos(pos_src)
            self._map.set_reference_path(pos[:, 0], pos[:, 2])

            self._localizer = None
            self._localizer_rc = None
            self._localizer_legacy = None
            self._prev_ts_wall = None
            self._prev_rc_ts_wall = None

            ds = cfg.get("data_source", "liftoff")
            obs_sigma: float = float(cfg.get("obs_sigma", 1.5))
            channel_weights: list[float] | None = cfg.get("channel_weights", None)
            _pf_extra: dict = {"obs_sigma": obs_sigma}
            if channel_weights is not None:
                _pf_extra["channel_weights"] = np.asarray(channel_weights, dtype=float)

            bf_path: Path | None = None
            if p_bf_npz is not None and refbuild.npz_feature_kind(p_bf_npz) == FEATURE_BETAFLIGHT_CLASSIC_V1:
                bf_path = p_bf_npz
                self._localizer = OnlineLocalizer.from_file(bf_path, **_pf_extra)
                self._localizer.reset()

            # Legacy (raw-sticks) localizer only makes sense when driven by LF telemetry.
            # Skip it for RC-only sessions — nothing would ever call update() on it.
            if ds != "rc":
                if p_leg_npz is not None:
                    if bf_path is None or p_leg_npz.resolve() != bf_path.resolve():
                        self._localizer_legacy = OnlineLocalizer.from_file(p_leg_npz, **_pf_extra)
                        self._localizer_legacy.reset()
                elif p_bf_npz is not None and bf_path is None:
                    self._localizer_legacy = OnlineLocalizer.from_file(p_bf_npz, **_pf_extra)
                    self._localizer_legacy.reset()

            inv_lf0 = self._graphs.get_invert_state().get("lf", {})
            if ds != "rc" and any(inv_lf0.values()):
                _log.info(
                    "Localizer: LF invert active %s — sticks are sign-flipped before "
                    "matching.  Build reference with the same invert settings via the "
                    "'Build…' dialog so the feature vectors share the same convention.",
                    {k: v for k, v in inv_lf0.items() if v},
                )
            if ds == "both" and self._localizer is not None:
                self._localizer_rc = OnlineLocalizer.from_file(bf_path, **_pf_extra)
                self._localizer_rc.reset()
                _log.info(
                    "RC localizer (Betaflight): same .npz as Liftoff BF — blue dotted trail; "
                    "throttled INFO compares RC vs LF when legacy is off, else see loc_triple.",
                )
            elif ds == "rc":
                # RC-only: _localizer is already the BF localizer, driven by RC sticks.
                _log.info(
                    "RC-only localizer (Betaflight): driven by RC sticks → gold trail/arrow.",
                )
            _log.info(
                "Localizer obs_sigma=%.1f  channel_weights=%s",
                obs_sigma,
                channel_weights if channel_weights is not None else "[1,1,1,1]",
            )
            bundle_bits: list[str] = []
            if self._localizer is not None and bf_path is not None:
                label = "RC Betaflight" if ds == "rc" else "LF Betaflight"
                bundle_bits.append(f"{label} ← {bf_path.name}")
            if self._localizer_legacy is not None:
                leg_src = p_leg_npz if p_leg_npz is not None else p_bf_npz
                bundle_bits.append(f"LF legacy raw ← {leg_src.name if leg_src else '?'}")
            if self._localizer_rc is not None:
                bundle_bits.append("RC Betaflight ← same BF .npz")
            if bundle_bits:
                legend = (
                    "gold=RC BF"
                    if ds == "rc"
                    else "gold=LF BF, blue=RC BF, green dash-dot=LF raw"
                )
                _log.info("Localizer bundle (%s): %s", legend, " | ".join(bundle_bits))
            if bf_path is not None and p_leg_npz is None:
                _log.info(
                    "Legacy sidecar missing next to %s — raw-stick PF disabled. "
                    "Re-run \"Build && Save\" to emit *_legacy_sticks.npz.",
                    bf_path.name,
                )
            if bf_path is None and self._localizer_legacy is not None:
                _log.info(
                    "Only legacy-stick .npz in bundle — RC Betaflight localizer needs the BF file.",
                )

            # KF Layer 2: работает поверх RC-локализатора (режимы "rc" и "both")
            rc_loc_for_kf = self._localizer_rc if ds == "both" else (
                self._localizer if ds == "rc" else None
            )
            if rc_loc_for_kf is not None:
                self._kf_layer2 = KFLayer2(rc_loc_for_kf.ref)
                self._kf_layer2.reset()
                _log.info("KF Layer 2 initialized (ds=%s)", ds)
            else:
                self._kf_layer2 = None

        except Exception as exc:
            _log.error("Localizer init failed: %s", exc)
            QMessageBox.warning(self, "Localizer", f"Не удалось загрузить эталон:\n{exc}")
            self._map.clear_reference_path()
            self._localizer = None
            self._localizer_rc = None
            self._localizer_legacy = None

    def _reset_localizer(self) -> None:
        if (
            self._localizer is None
            and self._localizer_rc is None
            and self._localizer_legacy is None
        ):
            return
        try:
            if self._localizer is not None:
                self._localizer.reset()
            if self._localizer_rc is not None:
                self._localizer_rc.reset()
            if self._localizer_legacy is not None:
                self._localizer_legacy.reset()
        except Exception:
            pass
        self._prev_ts_wall = None
        self._prev_rc_ts_wall = None
        self._prev_kf_ts_wall = None
        self._dual_loc_log_mono = 0.0
        self._last_lf_sticks = None
        self._last_rc_sticks = None
        self._last_lf_loc = None
        self._last_rc_loc = None
        self._last_legacy_loc = None
        self._last_kf_loc = None
        if self._kf_layer2 is not None:
            self._kf_layer2.reset()
        self._map.clear_localizer_overlay()
        self._last_loc = None
        self._map.update_hud(self._latest_frame, None, has_gt=(self._latest_frame is not None))

    def _reset_localizer_on_seek(self) -> None:
        """Reset particle filter state and clear map overlays after a replay seek."""
        self._reset_localizer()

    def _build_locs_dict(self) -> dict[str, tuple[float, float]] | None:
        """Собрать словарь активных оценок локализаторов для HUD.

        Returns
        -------
        dict с ключами "LF", "RC", "Legacy", "KF" → (progress, sigma_m) или None.
        """
        locs: dict[str, tuple[float, float]] = {}
        if self._last_lf_loc is not None:
            locs["LF"] = (self._last_lf_loc[2], self._last_lf_loc[3])
        if self._last_rc_loc is not None:
            locs["RC"] = (self._last_rc_loc[2], self._last_rc_loc[3])
        # rc-only mode: _localizer is the RC one, stored in _last_loc
        if not locs.get("RC") and self._record_ds_mode == "rc" and self._last_loc is not None:
            locs["RC"] = self._last_loc
        if self._last_legacy_loc is not None:
            locs["Legacy"] = (self._last_legacy_loc[2], self._last_legacy_loc[3])
        if self._last_kf_loc is not None:
            locs["KF"] = (self._last_kf_loc[2], self._last_kf_loc[3])
        return locs if locs else None

    @staticmethod
    def _session_track_id(session_path: Path) -> str:
        """Return the track_id for *session_path*.

        Priority:
        1. ``track.json`` top-level ``id`` or ``track_id`` field
        2. ``meta.json`` ``track`` or ``track_id`` field
        3. Parse the session folder name (``…_track-<id>_session-…``)
        """
        track_file = session_path / "track.json"
        if track_file.exists():
            try:
                with open(track_file, encoding="utf-8") as f:
                    td = json.load(f)
                tid = str(td.get("id") or td.get("track_id") or "").strip()
                if tid:
                    return tid
            except Exception:
                pass

        meta_file = session_path / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    m = json.load(f)
                tid = str(m.get("track") or m.get("track_id") or "").strip()
                if tid:
                    return tid
            except Exception:
                pass

        # Last resort: parse folder name  e.g. "…_track-track-001_session-002"
        import re
        m2 = re.search(r"_track-(.+?)_session-", session_path.name)
        if m2:
            return m2.group(1)

        return ""

    def _try_init_localizer_for_replay(self, session_path: Path) -> None:
        """Initialize (or deactivate) the localizer based on the selected replay session.

        Reads ``track.json`` → ``track_id`` → finds the default reference for
        that track via ``reference_builder.default_for_track``.  Also reads
        ``meta.json`` to determine the data-source mode so the RC localizer is
        set up when needed.
        """
        # Respect the enable checkbox on the Replay page
        if not self._replay_page.localizer_enabled():
            self._deactivate_localizer_full()
            return

        track_id = self._session_track_id(session_path)
        if not track_id:
            _log.info("Replay localizer: no track_id — localizer disabled for this session.")
            self._deactivate_localizer_full()
            return

        # Prefer the profile explicitly selected in the Replay page; fall back to default
        ref_path = self._replay_page.current_localizer_path()
        if ref_path is None:
            ref_path = refbuild.default_for_track(track_id)
        if ref_path is None:
            _log.info(
                "Replay localizer: no reference found for track '%s' — localizer disabled.",
                track_id,
            )
            self._deactivate_localizer_full()
            return

        # Determine data-source mode from meta.json
        ds_mode = "liftoff"
        meta_file = session_path / "meta.json"
        if meta_file.exists():
            try:
                meta = load_meta(session_path)
                ds_mode = meta.get("data_source", "liftoff")
            except Exception as exc:
                _log.warning("Replay localizer: cannot read meta.json: %s", exc)

        # Auto-detect RC-only sessions: no telemetry.parquet but rc_channels.parquet present
        if ds_mode == "liftoff" and not (session_path / "telemetry.parquet").exists():
            if (session_path / "rc_channels.parquet").exists():
                ds_mode = "rc"
                _log.info(
                    "Replay localizer: no telemetry.parquet found — auto-detected RC-only session, "
                    "switching to ds_mode='rc'.",
                )

        # Build a cfg that mirrors what _init_localizer_from_cfg expects
        cfg = {
            "localizer_enabled": True,
            "localizer_reference_path": str(ref_path),
            "data_source": ds_mode,
            "obs_sigma": self._replay_page.current_obs_sigma(),
            "channel_weights": self._replay_page.channel_weights(),
        }
        self._init_localizer_from_cfg(cfg)
        self._record_ds_mode = ds_mode

        # Load the session's own rate profile so update() converts sticks with
        # the correct rates, making the reference reusable across rate changes.
        rp_file = session_path / "rate_profile.json"
        if rp_file.exists():
            try:
                import json as _json
                self._current_rate_profile = _json.loads(rp_file.read_text(encoding="utf-8"))
                _log.info(
                    "Replay localizer: rate_profile loaded from session (%s)",
                    self._current_rate_profile.get("name", "?"),
                )
            except Exception as exc:
                _log.warning("Replay localizer: cannot read rate_profile.json: %s", exc)
                self._current_rate_profile = None
        else:
            self._current_rate_profile = None

        self._on_loc_show_changed()
        _log.info(
            "Replay localizer: track='%s' ds='%s' ref='%s'",
            track_id, ds_mode, ref_path.name,
        )

    def _log_localizer_compare(self, ts_wall: float) -> None:
        """Throttled INFO: dual (LF_BF vs RC_BF) and optional triple (+ legacy raw sticks)."""
        now = time.monotonic()
        if now - self._dual_loc_log_mono < 1.0:
            return
        self._dual_loc_log_mono = now
        lf_s = self._last_lf_sticks
        rc_s = self._last_rc_sticks
        lf_p = self._last_lf_loc
        rc_p = self._last_rc_loc
        leg_p = self._last_legacy_loc

        if self._record_ds_mode == "both" and lf_s and rc_s and lf_p and rc_p:
            d = [round(rc_s[i] - lf_s[i], 4) for i in range(4)]
            dx = rc_p[0] - lf_p[0]
            dz = rc_p[1] - lf_p[1]
            dp = rc_p[2] - lf_p[2]
            dsig = rc_p[3] - lf_p[3]
            if leg_p is not None:
                _log.info(
                    "loc_triple ts=%.3f Δsticks(rc-lf) T,Y,P,R=%s | "
                    "LF_BF xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                    "RC_BF xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                    "LF_raw xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                    "Δxz RC−LF=(%.3f,%.3f) Δprog RC−LF=%.4f LEG−LF=%.4f RC−LEG=%.4f Δσ=%.3f",
                    ts_wall,
                    d,
                    lf_p[0], lf_p[1], lf_p[2], lf_p[3],
                    rc_p[0], rc_p[1], rc_p[2], rc_p[3],
                    leg_p[0], leg_p[1], leg_p[2], leg_p[3],
                    dx, dz, dp, leg_p[2] - lf_p[2], rc_p[2] - leg_p[2], dsig,
                )
            else:
                _log.info(
                    "loc_dual ts_rc=%.3f Δsticks(rc-lf) T,Y,P,R=%s | "
                    "LF out: xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                    "RC out: xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                    "Δxz=(%.3f,%.3f) Δprog=%.4f Δσ=%.3f",
                    ts_wall,
                    d,
                    lf_p[0], lf_p[1], lf_p[2], lf_p[3],
                    rc_p[0], rc_p[1], rc_p[2], rc_p[3],
                    dx, dz, dp, dsig,
                )
            return

        if self._record_ds_mode == "liftoff" and leg_p is not None and lf_p is not None:
            _log.info(
                "loc_lf_bf_vs_legacy ts=%.3f LF_BF xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                "LF_raw xz=(%.2f,%.2f) prog=%.3f σ=%.2f | "
                "Δxz=(%.3f,%.3f) Δprog=%.4f Δσ=%.3f",
                ts_wall,
                lf_p[0], lf_p[1], lf_p[2], lf_p[3],
                leg_p[0], leg_p[1], leg_p[2], leg_p[3],
                leg_p[0] - lf_p[0], leg_p[1] - lf_p[1], leg_p[2] - lf_p[2], leg_p[3] - lf_p[3],
            )
            return

        if self._record_ds_mode == "both":
            _log.debug(
                "loc_compare(wait) ts=%.3f lf_st=%s rc_st=%s lf_p=%s rc_p=%s leg=%s",
                ts_wall,
                lf_s is not None,
                rc_s is not None,
                lf_p is not None,
                rc_p is not None,
                leg_p is not None,
            )

    def _on_loc_show_changed(self) -> None:
        if self._mode == MODE_REPLAY:
            s = self._replay_page.localizer_show_state()
        else:
            s = self._setup_page.localizer_show_state()
        self._map.set_reference_path_visible(s["path"])
        self._map.set_localizer_arrow_visible(s["arrow"])
        self._map.set_localizer_trail_visible(s["trail"])

    @pyqtSlot()
    def _on_replay_loc_settings_changed(self) -> None:
        """Re-initialize the localizer when the user changes settings in the Replay page."""
        if self._mode != MODE_REPLAY:
            return
        self._on_loc_show_changed()
        if self._last_replay_path:
            self._try_init_localizer_for_replay(Path(self._last_replay_path))
            self._reset_localizer_on_seek()

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

        # Capture the rate profile NOW from cfg so the localizer can convert sticks
        # to deg/s immediately — rate_profile.json is only written on session STOP,
        # so reading it in _on_session_started always yields None during recording.
        rate = cfg.get("rate")
        self._current_rate_profile = rate if isinstance(rate, dict) and rate else None
        if self._current_rate_profile:
            _log.info(
                "Session rate_profile set for localizer (from cfg): %s",
                self._current_rate_profile.get("name", "?"),
            )

        self._init_localizer_from_cfg(cfg)
        self._on_loc_show_changed()

        import time as _time
        self._graphs.set_time_zero(_time.time())

        ds_mode = cfg.get("data_source", "liftoff")
        self._record_ds_mode = ds_mode
        self._live = LiveDataSource(self)
        self._live.telemetry_updated.connect(self._on_telemetry)
        self._live.telemetry_batch.connect(self._graphs.update_batch)
        self._live.rc_batch.connect(self._graphs.update_rc_batch)
        self._live.rc_batch.connect(self._on_rc_batch_localizer)
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
        # Use the unified helper so meta.json / folder name are also checked
        replay_track_id = self._session_track_id(p)
        self._replay_page.set_localizer_track(replay_track_id)

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
                # set_invert_state blocks checkbox signals, propagate manually
                self._propagate_invert_to_pages()
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
        self._replay.rc_batch.connect(self._on_rc_batch_localizer)
        self._replay.event_fired.connect(self._on_event)
        self._replay.stats_updated.connect(self._on_stats)
        self._replay.video_frame.connect(self._on_replay_video_frame)
        self._replay.progress_updated.connect(replay_ctrl.update_progress)
        self._replay.finished.connect(lambda: replay_ctrl.set_playing(False))

        self._map.clear_trail()
        self._graphs.clear()
        self._lap_count = 0
        self._top_bar.set_summary(p.name)

        self._try_init_localizer_for_replay(p)

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
            self._reset_localizer_on_seek()

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
        self._reset_localizer_on_seek()

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
        self._reset_localizer_on_seek()

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
        self._reset_localizer_on_seek()

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
        self._reset_localizer_on_seek()

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
        if (
            self._mode in (MODE_RECORD, MODE_REPLAY)
            and self._record_ds_mode in ("liftoff", "both")
            and (self._localizer is not None or self._localizer_legacy is not None)
        ):
            ts = float(frame["ts_wall"])
            prev = self._prev_ts_wall
            dt = (ts - prev) if prev is not None else None
            if dt is not None and (dt < 0 or dt > 2.0):
                dt = None
            self._prev_ts_wall = ts
            try:
                inv_lf = self._graphs.get_invert_state().get("lf", {})
                sticks_lf = lf_sticks_with_invert(frame, inv_lf)
                if self._record_ds_mode == "both" or self._localizer_legacy is not None:
                    self._last_lf_sticks = list(sticks_lf)

                if self._localizer is not None:
                    res = self._localizer.update(sticks_lf, dt, rate_profile=self._current_rate_profile)
                    self._map.update_localizer_estimate(
                        float(res.position_xyz[0]),
                        float(res.position_xyz[2]),
                    )
                    self._last_loc = (float(res.progress), float(res.uncertainty_m))
                    if self._record_ds_mode == "both" or self._localizer_legacy is not None:
                        self._last_lf_loc = (
                            float(res.position_xyz[0]),
                            float(res.position_xyz[2]),
                            float(res.progress),
                            float(res.uncertainty_m),
                        )

                if self._localizer_legacy is not None:
                    sticks_raw = [
                        float(frame.get("in_throttle", 0.0)),
                        float(frame.get("in_yaw", 0.0)),
                        float(frame.get("in_pitch", 0.0)),
                        float(frame.get("in_roll", 0.0)),
                    ]
                    res_leg = self._localizer_legacy.update(sticks_raw, dt)
                    self._map.update_localizer_legacy_estimate(
                        float(res_leg.position_xyz[0]),
                        float(res_leg.position_xyz[2]),
                    )
                    self._last_legacy_loc = (
                        float(res_leg.position_xyz[0]),
                        float(res_leg.position_xyz[2]),
                        float(res_leg.progress),
                        float(res_leg.uncertainty_m),
                    )
                    if self._localizer is None:
                        self._last_loc = (
                            float(res_leg.progress),
                            float(res_leg.uncertainty_m),
                        )
                    if self._record_ds_mode == "liftoff":
                        self._log_localizer_compare(ts)
            except Exception as exc:
                _log.warning("Localizer update failed: %s", exc)
        self._map.update_hud(frame, self._build_locs_dict(), has_gt=(frame is not None))

    @pyqtSlot(list)
    def _on_rc_batch_localizer(self, frames: list) -> None:
        """Drive PF from RC (RC-only) or second PF from RC in Liftoff+RC (dual compare)."""
        if self._mode not in (MODE_RECORD, MODE_REPLAY):
            return
        ds = self._record_ds_mode
        if ds not in ("rc", "both"):
            return
        loc = self._localizer if ds == "rc" else self._localizer_rc
        if loc is None:
            return
        inv = self._graphs.get_invert_state().get("rc", {})
        for frame in frames:
            ts = float(frame["ts_wall"])
            prev = self._prev_rc_ts_wall
            dt = (ts - prev) if prev is not None else None
            if dt is not None and (dt < 0 or dt > 2.0):
                dt = None
            self._prev_rc_ts_wall = ts
            try:
                sticks = rc_frame_to_sticks_norm(frame, inv)
                res = loc.update(sticks, dt, rate_profile=self._current_rate_profile)
                if ds == "rc":
                    x = float(res.position_xyz[0])
                    z = float(res.position_xyz[2])
                    prog = float(res.progress)
                    sigma = float(res.uncertainty_m)
                    self._map.update_localizer_estimate(x, z)
                    self._last_loc = (prog, sigma)
                    self._last_rc_sticks = list(sticks)
                    # Throttled log (≤1 Hz)
                    now = time.monotonic()
                    if now - self._dual_loc_log_mono >= 1.0:
                        self._dual_loc_log_mono = now
                        _log.info(
                            "loc_rc ts=%.3f sticks T,Y,P,R=%s | "
                            "RC_BF xz=(%.2f,%.2f) prog=%.3f σ=%.2f",
                            ts,
                            [round(s, 4) for s in sticks],
                            x, z, prog, sigma,
                        )
                else:
                    self._last_rc_sticks = list(sticks)
                    self._map.update_localizer_rc_estimate(
                        float(res.position_xyz[0]),
                        float(res.position_xyz[2]),
                    )
                    self._last_rc_loc = (
                        float(res.position_xyz[0]),
                        float(res.position_xyz[2]),
                        float(res.progress),
                        float(res.uncertainty_m),
                    )
                    self._log_localizer_compare(ts)

                # KF Layer 2 — применяем к результату RC-локализатора
                if self._kf_layer2 is not None:
                    dt_kf = (ts - self._prev_kf_ts_wall) if self._prev_kf_ts_wall is not None else None
                    if dt_kf is not None and (dt_kf < 0 or dt_kf > 2.0):
                        dt_kf = None
                    self._prev_kf_ts_wall = ts
                    try:
                        res_kf = self._kf_layer2.update(res, dt_kf)
                        self._map.update_localizer_kf_estimate(
                            float(res_kf.position_xyz[0]),
                            float(res_kf.position_xyz[2]),
                        )
                        self._last_kf_loc = (
                            float(res_kf.position_xyz[0]),
                            float(res_kf.position_xyz[2]),
                            float(res_kf.progress),
                            float(res_kf.uncertainty_m),
                        )
                    except Exception as exc:
                        _log.warning("KF Layer 2 update failed: %s", exc)

            except Exception as exc:
                _log.warning("Localizer RC update failed: %s", exc)
        self._map.update_hud(
            self._latest_frame,
            self._build_locs_dict(),
            has_gt=(self._latest_frame is not None),
        )

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
        # rate_profile is already set in _do_start_session from cfg["rate"]
        # before the session started.  Nothing to do here.

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
        # Only restart the preview when we are still in Record mode.  If the user
        # switched to Replay before this async callback fired, starting a screen
        # capture preview here would compete with replay video frames and cause
        # flickering.
        if self._mode == MODE_RECORD:
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

        # Restore last active mode (default: Record)
        saved_mode = s.get("last_mode", "record")
        if saved_mode == "replay":
            # Use a short delay so the window is fully shown before switching
            QTimer.singleShot(0, lambda: self._switch_mode(MODE_REPLAY))

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
        s["last_mode"] = "replay" if self._mode == MODE_REPLAY else "record"
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

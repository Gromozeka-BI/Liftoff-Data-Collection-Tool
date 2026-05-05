"""Bottom control bar for RECORD mode."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import threading

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from dct.screen_recorder import scan_video_devices
from dct.rc_receiver import scan_serial_ports
from dct.gui import theme, ui_settings
from dct.gui.global_hotkeys import GlobalHotkeyManager
from dct.gui.widgets.status_panel import StatusPanel

_PROFILES_DIR = Path("profiles")

_SRC_LIFTOFF = "liftoff"
_SRC_RC      = "rc"
_SRC_BOTH    = "both"


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _scan_dir(subdir: str) -> list[dict]:
    d = _PROFILES_DIR / subdir
    if not d.exists():
        return []
    return [_load_json(p) for p in sorted(d.glob("*.json"))]


def _scan_tracks() -> list[dict]:
    d = Path("tracks")
    if not d.exists():
        return []
    return [_load_json(p) for p in sorted(d.glob("*.json"))]


def _pilots() -> list[dict]:
    p = _PROFILES_DIR / "pilots.json"
    return _load_json(p) if p.exists() else []


class RecordBar(QWidget):
    start_requested      = pyqtSignal(dict)
    stop_requested       = pyqtSignal()
    lap_requested        = pyqtSignal()
    gate_requested       = pyqtSignal()
    sf_requested         = pyqtSignal()
    video_source_changed = pyqtSignal(dict)
    # internal signals for background scan results
    _video_devices_ready = pyqtSignal(list)
    _com_ports_ready     = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recording         = False
        self._loading           = False   # True while programmatically restoring settings
        self._saved_com_port    = ""
        self._saved_video_index = None
        self._localizer_ref_path = ""
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(6, 4, 6, 4)

        # ── Session config ─────────────────────────────────────────────────
        cfg_box = QGroupBox("Session config")
        cfg_lay = QHBoxLayout(cfg_box)
        cfg_lay.setSpacing(6)

        self._combo_pilot  = self._make_combo("Pilot")
        self._combo_drone  = self._make_combo("Drone")
        self._combo_rate   = self._make_combo("Rate")
        self._combo_camera = self._make_combo("Camera")
        self._combo_track  = self._make_combo("Track")

        for combo in (self._combo_pilot, self._combo_drone,
                      self._combo_rate, self._combo_camera, self._combo_track):
            cfg_lay.addWidget(combo)
        root.addWidget(cfg_box)

        # ── Data sources ───────────────────────────────────────────────────
        ds_box = QGroupBox("Data sources")
        ds_lay = QVBoxLayout(ds_box)
        ds_lay.setSpacing(3)

        # Mode combo
        self._combo_ds_mode = QComboBox()
        self._combo_ds_mode.addItem("Liftoff only", userData=_SRC_LIFTOFF)
        self._combo_ds_mode.addItem("RC only",      userData=_SRC_RC)
        self._combo_ds_mode.addItem("Liftoff + RC", userData=_SRC_BOTH)
        self._combo_ds_mode.setToolTip("Select active telemetry sources")
        self._combo_ds_mode.currentIndexChanged.connect(self._on_ds_mode_changed)
        ds_lay.addWidget(self._combo_ds_mode)

        # COM port row
        rc_row = QHBoxLayout()
        rc_row.setSpacing(3)
        self._combo_com = QComboBox()
        self._combo_com.setMinimumWidth(90)
        self._combo_com.setToolTip("RC receiver COM port")
        self._btn_refresh_com = QPushButton("↻")
        self._btn_refresh_com.setFixedWidth(22)
        self._btn_refresh_com.setToolTip("Refresh COM ports")
        self._btn_refresh_com.clicked.connect(self._refresh_com_ports)
        self._lbl_rc_status = QLabel("●")
        self._lbl_rc_status.setStyleSheet(f"color: {theme.WARN}; font-size: 14px;")
        self._lbl_rc_status.setToolTip("RC receiver: offline")
        rc_row.addWidget(self._combo_com)
        rc_row.addWidget(self._btn_refresh_com)
        rc_row.addWidget(self._lbl_rc_status)
        ds_lay.addLayout(rc_row)
        root.addWidget(ds_box)

        # ── Video source ───────────────────────────────────────────────────
        vid_box = QGroupBox("Video source")
        vid_vlay = QVBoxLayout(vid_box)
        vid_vlay.setSpacing(3)

        vid_row = QHBoxLayout()
        vid_row.setSpacing(4)
        self._combo_source = QComboBox()
        self._combo_source.setMinimumWidth(150)
        self._btn_refresh_src = QPushButton("↻")
        self._btn_refresh_src.setFixedWidth(22)
        self._btn_refresh_src.setToolTip("Refresh capture devices")
        self._btn_refresh_src.clicked.connect(self._refresh_video_sources)
        self._combo_source.currentIndexChanged.connect(self._on_video_source_changed)
        vid_row.addWidget(self._combo_source)
        vid_row.addWidget(self._btn_refresh_src)
        vid_vlay.addLayout(vid_row)

        # Save folder row
        save_row = QHBoxLayout()
        save_row.setSpacing(4)
        self._lbl_sessions_dir = QLabel()
        self._lbl_sessions_dir.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
        self._lbl_sessions_dir.setToolTip("Папка для сохранения сессий")
        saved_dir = ui_settings.load().get("sessions_dir", "sessions")
        self._lbl_sessions_dir.setText(str(saved_dir))
        self._btn_browse_dir = QPushButton("📁")
        self._btn_browse_dir.setFixedWidth(26)
        self._btn_browse_dir.setToolTip("Выбрать папку сохранения сессий")
        self._btn_browse_dir.clicked.connect(self._browse_sessions_dir)
        save_row.addWidget(self._lbl_sessions_dir, stretch=1)
        save_row.addWidget(self._btn_browse_dir)
        vid_vlay.addLayout(save_row)
        root.addWidget(vid_box)

        # ── Stick localizer (reference.npz) ─────────────────────────────
        loc_box = QGroupBox("Localizer")
        loc_lay = QVBoxLayout(loc_box)
        loc_lay.setSpacing(3)
        loc_row = QHBoxLayout()
        loc_row.setSpacing(4)
        self._chk_localizer = QCheckBox("Enable")
        self._chk_localizer.setToolTip(
            "Онлайн-оценка позиции по стикам Liftoff; нужен .npz эталонного круга",
        )
        self._btn_ref_npz = QPushButton("…")
        self._btn_ref_npz.setFixedWidth(28)
        self._btn_ref_npz.setToolTip("Выбрать reference.npz")
        self._btn_ref_npz.clicked.connect(self._browse_localizer_ref)
        self._lbl_ref_npz = QLabel("(no file)")
        self._lbl_ref_npz.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
        self._lbl_ref_npz.setToolTip("")
        self._lbl_ref_npz.setMinimumWidth(120)
        self._lbl_ref_npz.setMaximumWidth(220)
        self._lbl_ref_npz.setWordWrap(False)
        loc_row.addWidget(self._chk_localizer)
        loc_row.addWidget(self._btn_ref_npz)
        loc_row.addWidget(self._lbl_ref_npz, stretch=1)
        loc_lay.addLayout(loc_row)
        root.addWidget(loc_box)

        # ── Status ────────────────────────────────────────────────────────
        self.status = StatusPanel()
        root.addWidget(self.status)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_box = QGroupBox("Controls")
        btn_lay = QVBoxLayout(btn_box)
        btn_lay.setSpacing(4)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        btn_lay.addLayout(row1)
        btn_lay.addLayout(row2)

        self._btn_start = self._make_btn("6  START", "btn_start", row1)
        self._btn_stop  = self._make_btn("7  STOP",  "btn_stop",  row1)
        self._btn_lap   = self._make_btn("8  LAP",   "",          row2)
        self._btn_gate  = self._make_btn("9  GATE",  "",          row2)
        self._btn_sf    = self._make_btn("0  S/F",   "",          row2)
        root.addWidget(btn_box)

        # Connections
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self.stop_requested)
        self._btn_lap.clicked.connect(self.lap_requested)
        self._btn_gate.clicked.connect(self.gate_requested)
        self._btn_sf.clicked.connect(self.sf_requested)

        self._shortcuts = []
        for key, slot in [("6", self._on_start), ("7", self.stop_requested),
                          ("8", self.lap_requested), ("9", self.gate_requested),
                          ("0", self.sf_requested)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

        self._hotkeys = GlobalHotkeyManager()
        for key, slot in [("6", self._on_start), ("7", self._stop_safe),
                          ("8", self._lap_safe), ("9", self._gate_safe),
                          ("0", self._sf_safe)]:
            self._hotkeys.register(key, slot)

        # Wire background scan signals before deferring scans
        self._video_devices_ready.connect(self._apply_video_devices)
        self._com_ports_ready.connect(self._apply_com_ports)

        self._set_buttons_recording(False)
        self.reload_profiles()          # fills combos AND restores saved selections
        self._on_ds_mode_changed()

        # Connect save-on-change AFTER initial restore so we don't overwrite
        # good saved state with empty values during startup population.
        for combo in (self._combo_pilot, self._combo_drone, self._combo_rate,
                      self._combo_camera, self._combo_track):
            combo.currentIndexChanged.connect(self._save_settings)
        self._combo_ds_mode.currentIndexChanged.connect(self._save_settings)
        self._combo_com.currentIndexChanged.connect(self._save_settings)
        self._chk_localizer.stateChanged.connect(self._save_localizer_settings)

        # Only scan COM ports automatically (lightweight). Video device scan is
        # triggered by the user via ↻ to avoid DirectShow interference with the
        # mouse cursor on Windows. The screen-capture option is always available.
        self._combo_source.addItem("Liftoff (screen capture)", userData={"type": "screen"})
        QTimer.singleShot(200, self._refresh_com_ports)

    # ── public API ─────────────────────────────────────────────────────────

    def reload_profiles(self) -> None:
        self._loading = True
        try:
            self._fill_combo(self._combo_pilot,  _pilots(),            "nickname")
            self._fill_combo(self._combo_drone,  _scan_dir("drones"),  "name")
            self._fill_combo(self._combo_rate,   _scan_dir("rates"),   "name")
            self._fill_combo(self._combo_camera, _scan_dir("cameras"), "name")
            self._fill_combo(self._combo_track,  _scan_tracks(),       "name")
        finally:
            self._loading = False
        self._restore_settings()

    def _browse_sessions_dir(self) -> None:
        current = ui_settings.load().get("sessions_dir", "sessions")
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения сессий", current,
        )
        if folder:
            self._lbl_sessions_dir.setText(folder)
            ui_settings.update("sessions_dir", folder)

    def _apply_localizer_label(self) -> None:
        p = self._localizer_ref_path
        self._lbl_ref_npz.setToolTip(p or "")
        if not p:
            self._lbl_ref_npz.setText("(no file)")
            return
        name = Path(p).name
        if len(name) <= 26:
            self._lbl_ref_npz.setText(name)
        else:
            self._lbl_ref_npz.setText(f"{name[:11]}…{name[-12:]}")

    def _browse_localizer_ref(self) -> None:
        start_dir = str(Path(self._localizer_ref_path).parent) if self._localizer_ref_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Эталонный круг (reference.npz)",
            start_dir or ".",
            "NumPy archives (*.npz);;All files (*.*)",
        )
        if path:
            self._localizer_ref_path = path
            self._apply_localizer_label()
            self._save_localizer_settings()

    def _save_localizer_settings(self) -> None:
        if self._loading:
            return
        d = ui_settings.load()
        d["localizer_reference_npz"] = self._localizer_ref_path
        d["localizer_enabled"] = self._chk_localizer.isChecked()
        ui_settings.save(d)

    def get_sessions_dir(self) -> str:
        return ui_settings.load().get("sessions_dir", "sessions")

    def _restore_settings(self) -> None:
        """Apply previously saved combo selections (called after profiles reload)."""
        s = ui_settings.load()
        self._loading = True
        try:
            for combo, key in [
                (self._combo_pilot,  "pilot"),
                (self._combo_drone,  "drone"),
                (self._combo_rate,   "rate"),
                (self._combo_camera, "camera"),
                (self._combo_track,  "track"),
            ]:
                text = s.get(key, "")
                if text:
                    idx = combo.findText(text)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

            # Data source mode
            mode = s.get("ds_mode")
            if mode:
                for i in range(self._combo_ds_mode.count()):
                    if self._combo_ds_mode.itemData(i) == mode:
                        self._combo_ds_mode.setCurrentIndex(i)
                        break

            self._localizer_ref_path = str(s.get("localizer_reference_npz", "") or "")
            self._chk_localizer.blockSignals(True)
            self._chk_localizer.setChecked(bool(s.get("localizer_enabled", False)))
            self._chk_localizer.blockSignals(False)
            self._apply_localizer_label()
        finally:
            self._loading = False

        # COM port and video source are restored after async scans complete;
        # stash the saved values so the scan callbacks can pick them up.
        self._saved_com_port    = s.get("com_port", "")
        self._saved_video_index = s.get("video_source_index")   # device index or None

    def _save_settings(self) -> None:
        """Persist current combo selections to ui_settings.json."""
        if self._loading:
            return
        d = ui_settings.load()
        d.update({
            "pilot":    self._combo_pilot.currentText(),
            "drone":    self._combo_drone.currentText(),
            "rate":     self._combo_rate.currentText(),
            "camera":   self._combo_camera.currentText(),
            "track":    self._combo_track.currentText(),
            "ds_mode":  self._combo_ds_mode.currentData(),
            "com_port": self._combo_com.currentText(),
        })
        ui_settings.save(d)

    def set_recording(self, active: bool) -> None:
        self._recording = active
        self._set_buttons_recording(active)
        self.status.set_recording(active)
        for w in (self._combo_pilot, self._combo_drone, self._combo_rate,
                  self._combo_camera, self._combo_track,
                  self._combo_ds_mode, self._combo_com, self._btn_refresh_com,
                  self._combo_source, self._btn_refresh_src,
                  self._chk_localizer, self._btn_ref_npz):
            w.setEnabled(not active)

    def set_rc_status(self, online: bool) -> None:
        """Update the RC online/offline indicator (called from main thread)."""
        if online:
            self._lbl_rc_status.setStyleSheet(f"color: {theme.OK}; font-size: 14px;")
            self._lbl_rc_status.setToolTip("RC receiver: online")
        else:
            self._lbl_rc_status.setStyleSheet(f"color: {theme.ERR}; font-size: 14px;")
            self._lbl_rc_status.setToolTip("RC receiver: offline")

    def current_video_source(self) -> dict:
        return self._combo_source.currentData() or {"type": "screen"}

    def cleanup(self) -> None:
        self._hotkeys.unregister_all()

    # ── private slots ──────────────────────────────────────────────────────

    def _stop_safe(self) -> None:
        if self._recording: self.stop_requested.emit()

    def _lap_safe(self) -> None:
        if self._recording: self.lap_requested.emit()

    def _gate_safe(self) -> None:
        if self._recording: self.gate_requested.emit()

    def _sf_safe(self) -> None:
        if self._recording: self.sf_requested.emit()

    def _on_ds_mode_changed(self) -> None:
        mode = self._combo_ds_mode.currentData()
        rc_needed = mode in (_SRC_RC, _SRC_BOTH)
        self._combo_com.setEnabled(rc_needed)
        self._btn_refresh_com.setEnabled(rc_needed)
        self._lbl_rc_status.setVisible(rc_needed)

    def _on_video_source_changed(self) -> None:
        data = self._combo_source.currentData()
        if data:
            self.video_source_changed.emit(data)

    def _refresh_com_ports(self) -> None:
        """Scan COM ports in a background thread to avoid blocking the UI."""
        threading.Thread(target=self._scan_com_bg, daemon=True).start()

    def _scan_com_bg(self) -> None:
        self._com_ports_ready.emit(scan_serial_ports())

    @pyqtSlot(list)
    def _apply_com_ports(self, ports: list) -> None:
        current = self._combo_com.currentText()
        self._combo_com.blockSignals(True)
        self._combo_com.clear()
        for port in ports:
            self._combo_com.addItem(port)
        # Prefer: currently selected → saved setting
        target = current or getattr(self, "_saved_com_port", "")
        idx = self._combo_com.findText(target)
        if idx >= 0:
            self._combo_com.setCurrentIndex(idx)
        self._combo_com.blockSignals(False)

    def _refresh_video_sources(self) -> None:
        """Scan capture devices in a background thread (cv2 can be slow on Windows)."""
        threading.Thread(target=self._scan_video_bg, daemon=True).start()

    def _scan_video_bg(self) -> None:
        self._video_devices_ready.emit(scan_video_devices())

    @pyqtSlot(list)
    def _apply_video_devices(self, devices: list) -> None:
        current      = self._combo_source.currentData()
        saved_index  = getattr(self, "_saved_video_index", None)
        self._combo_source.blockSignals(True)
        self._combo_source.clear()
        self._combo_source.addItem("Liftoff (screen capture)", userData={"type": "screen"})
        for idx, label in devices:
            self._combo_source.addItem(f"HDZero – {label}", userData={"type": "device", "index": idx})

        # Restore: prefer current selection → saved device index → first item
        restored = False
        if current:
            for i in range(self._combo_source.count()):
                if self._combo_source.itemData(i) == current:
                    self._combo_source.setCurrentIndex(i)
                    restored = True
                    break
        if not restored and saved_index is not None:
            for i in range(self._combo_source.count()):
                d = self._combo_source.itemData(i)
                if d and d.get("type") == "device" and d.get("index") == saved_index:
                    self._combo_source.setCurrentIndex(i)
                    break
        self._combo_source.blockSignals(False)
        # Persist the device index so future restarts can find it
        d = self._combo_source.currentData()
        if d and d.get("type") == "device":
            ui_settings.update("video_source_index", d["index"])

    def _on_start(self) -> None:
        if self._recording:
            return
        pilot_data  = self._combo_pilot.currentData()
        drone_data  = self._combo_drone.currentData()
        track_data  = self._combo_track.currentData()
        rate_data   = self._combo_rate.currentData()
        camera_data = self._combo_camera.currentData()

        missing = []
        if not pilot_data: missing.append("Pilot")
        if not drone_data: missing.append("Drone")
        if not track_data: missing.append("Track")

        mode = self._combo_ds_mode.currentData() or _SRC_LIFTOFF
        if mode in (_SRC_RC, _SRC_BOTH) and not self._combo_com.currentText():
            missing.append("RC COM port")

        if self._chk_localizer.isChecked():
            if not self._localizer_ref_path:
                QMessageBox.warning(
                    self, "Localizer",
                    "Включён локализатор, но не выбран файл reference.npz.",
                )
                return
            if not Path(self._localizer_ref_path).is_file():
                QMessageBox.warning(
                    self, "Localizer",
                    f"Файл эталона не найден:\n{self._localizer_ref_path}",
                )
                return
            if mode == _SRC_RC:
                QMessageBox.warning(
                    self, "Localizer",
                    "Режим «RC only» не содержит стиков Liftoff — локализатор "
                    "будет отключён для этой сессии.",
                )

        if missing:
            QMessageBox.warning(
                self, "Session config incomplete",
                "Please select the following before starting:\n\n  • " + "\n  • ".join(missing),
            )
            return

        track_path = Path("tracks") / f"{track_data.get('id', track_data['name'])}.json"
        if not track_path.exists():
            for p in Path("tracks").glob("*.json"):
                d = _load_json(p)
                if d.get("name") == track_data["name"]:
                    track_path = p
                    break

        loc_enabled = (
            self._chk_localizer.isChecked()
            and bool(self._localizer_ref_path)
            and Path(self._localizer_ref_path).is_file()
            and mode in (_SRC_LIFTOFF, _SRC_BOTH)
        )
        cfg = {
            "pilot":        pilot_data.get("nickname", pilot_data.get("id", "?")),
            "drone":        drone_data.get("id", drone_data["name"]),
            "track":        track_data.get("id", track_data["name"]),
            "purpose":      "training",
            "track_path":   str(track_path) if track_path.exists() else None,
            "rate":         rate_data,
            "camera":       camera_data,
            "video_source": self._combo_source.currentData() or {"type": "screen"},
            "data_source":  mode,
            "rc_port":      self._combo_com.currentText() or None,
            "sessions_dir": self.get_sessions_dir(),
            "localizer_enabled":      loc_enabled,
            "localizer_reference_path": self._localizer_ref_path if loc_enabled else None,
        }
        self.start_requested.emit(cfg)

    def _set_buttons_recording(self, active: bool) -> None:
        self._btn_start.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        self._btn_lap.setEnabled(active)
        self._btn_gate.setEnabled(active)
        self._btn_sf.setEnabled(active)

    @staticmethod
    def _make_combo(placeholder: str) -> QComboBox:
        cb = QComboBox()
        cb.setPlaceholderText(placeholder)
        cb.setToolTip(placeholder)
        return cb

    @staticmethod
    def _make_btn(label: str, obj_name: str, layout: QHBoxLayout) -> QPushButton:
        btn = QPushButton(label)
        if obj_name:
            btn.setObjectName(obj_name)
        layout.addWidget(btn)
        return btn

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[dict], label_key: str) -> None:
        combo.clear()
        for item in items:
            combo.addItem(item.get(label_key, "?"), userData=item)

"""Bottom control bar for RECORD mode.

Layout (horizontal):
  [Config group: pilot/drone/rate/camera/track combos]
  [Status panel]
  [Buttons: 1-START  2-STOP  3-LAP  4-GATE  5-SF]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from dct.gui import theme
from dct.gui.global_hotkeys import GlobalHotkeyManager
from dct.gui.widgets.status_panel import StatusPanel

_PROFILES_DIR = Path("profiles")


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
    # Emitted with session config dict when Start is pressed
    start_requested = pyqtSignal(dict)
    stop_requested  = pyqtSignal()
    lap_requested   = pyqtSignal()
    gate_requested  = pyqtSignal()   # nearest gate
    sf_requested    = pyqtSignal()   # start/finish gate

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recording = False
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(6, 4, 6, 4)

        # ── Config group ──────────────────────────────────────────────────
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

        # Qt-шорткаты (резервные, работают когда DCT в фокусе)
        self._shortcuts = []
        for key, slot in [("6", self._on_start), ("7", self.stop_requested),
                          ("8", self.lap_requested), ("9", self.gate_requested),
                          ("0", self.sf_requested)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

        # Глобальные хоткеи через keyboard lib (работают из симулятора)
        self._hotkeys = GlobalHotkeyManager()
        for key, slot in [("6", self._on_start), ("7", self._stop_safe),
                          ("8", self._lap_safe), ("9", self._gate_safe),
                          ("0", self._sf_safe)]:
            self._hotkeys.register(key, slot)

        self._set_buttons_recording(False)
        self.reload_profiles()

    # ── public API ─────────────────────────────────────────────────────────

    def reload_profiles(self) -> None:
        self._fill_combo(self._combo_pilot,  _pilots(),      "nickname")
        self._fill_combo(self._combo_drone,  _scan_dir("drones"),  "name")
        self._fill_combo(self._combo_rate,   _scan_dir("rates"),   "name")
        self._fill_combo(self._combo_camera, _scan_dir("cameras"), "name")
        self._fill_combo(self._combo_track,  _scan_tracks(),       "name")

    def set_recording(self, active: bool) -> None:
        self._recording = active
        self._set_buttons_recording(active)
        self.status.set_recording(active)
        for combo in (self._combo_pilot, self._combo_drone,
                      self._combo_rate, self._combo_camera, self._combo_track):
            combo.setEnabled(not active)

    def cleanup(self) -> None:
        """Снимает глобальные хоткеи — вызывать при закрытии окна."""
        self._hotkeys.unregister_all()

    # Обёртки для глобальных хоткеев: фильтруем лишние события
    def _stop_safe(self) -> None:
        if self._recording:
            self.stop_requested.emit()

    def _lap_safe(self) -> None:
        if self._recording:
            self.lap_requested.emit()

    def _gate_safe(self) -> None:
        if self._recording:
            self.gate_requested.emit()

    def _sf_safe(self) -> None:
        if self._recording:
            self.sf_requested.emit()

    # ── internal ───────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._recording:
            return
        pilot_data  = self._combo_pilot.currentData()
        drone_data  = self._combo_drone.currentData()
        track_data  = self._combo_track.currentData()
        rate_data   = self._combo_rate.currentData()
        camera_data = self._combo_camera.currentData()

        missing = []
        if not pilot_data:  missing.append("Pilot")
        if not drone_data:  missing.append("Drone")
        if not track_data:  missing.append("Track")
        if missing:
            QMessageBox.warning(
                self, "Session config incomplete",
                "Please select the following before starting:\n\n  • " + "\n  • ".join(missing),
            )
            return

        track_path = Path("tracks") / f"{track_data.get('id', track_data['name'])}.json"
        if not track_path.exists():
            # Try finding by scanning
            for p in Path("tracks").glob("*.json"):
                d = _load_json(p)
                if d.get("name") == track_data["name"]:
                    track_path = p
                    break

        cfg = {
            "pilot":       pilot_data.get("nickname", pilot_data.get("id", "?")),
            "drone":       drone_data.get("id", drone_data["name"]),
            "track":       track_data.get("id", track_data["name"]),
            "purpose":     "training",
            "track_path":  str(track_path) if track_path.exists() else None,
            "rate":        rate_data,
            "camera":      camera_data,
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

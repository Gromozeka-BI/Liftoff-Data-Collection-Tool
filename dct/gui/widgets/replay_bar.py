"""Bottom control bar for REPLAY mode.

Layout:
  [Session combo]  [<< Play/Pause >>]  [scrubber ─●──]  [time / lap]  [0.5x 1x 2x]
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget,
)

from dct.gui import theme
from dct.gui.widgets.status_panel import StatusPanel


def _session_dirs() -> list[Path]:
    base = Path("sessions")
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])


class ReplayBar(QWidget):
    session_selected  = pyqtSignal(str)   # session directory path
    play_pause        = pyqtSignal()
    seek_fraction     = pyqtSignal(float) # 0.0 – 1.0
    speed_changed     = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(6, 4, 6, 4)

        # Session picker
        self._combo = QComboBox()
        self._combo.setMinimumWidth(260)
        self._combo.setToolTip("Select a recorded session to replay")
        self._combo.currentIndexChanged.connect(self._on_session_changed)
        root.addWidget(self._combo)

        # Transport buttons
        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setToolTip("Play / Pause  (Space)")
        self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self.play_pause)
        root.addWidget(self._btn_play)

        # Scrubber
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setToolTip("Scrub through the session timeline")
        self._slider.sliderMoved.connect(lambda v: self.seek_fraction.emit(v / 1000.0))
        self._slider.setMinimumWidth(200)
        root.addWidget(self._slider, stretch=1)

        # Time label
        self._lbl_time = QLabel("0:00 / 0:00")
        self._lbl_time.setStyleSheet(f"color: {theme.DIM}; font-family: monospace;")
        self._lbl_time.setMinimumWidth(100)
        root.addWidget(self._lbl_time)

        # Lap label
        self._lbl_lap = QLabel("")
        self._lbl_lap.setStyleSheet(f"color: {theme.OK};")
        root.addWidget(self._lbl_lap)

        # Speed buttons
        for label, speed in [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)]:
            btn = QPushButton(label)
            btn.setMaximumWidth(44)
            btn.setToolTip(f"Playback speed {label}")
            btn.clicked.connect(lambda _checked, s=speed: self.speed_changed.emit(s))
            root.addWidget(btn)

        # Status (compact)
        self.status = StatusPanel()
        root.addWidget(self.status)

        self.reload_sessions()

    # ── public API ─────────────────────────────────────────────────────────

    def reload_sessions(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        for d in _session_dirs():
            self._combo.addItem(d.name, userData=str(d))
        self._combo.blockSignals(False)
        # Trigger load of last session if any
        if self._combo.count():
            self._on_session_changed(self._combo.count() - 1)
            self._combo.setCurrentIndex(self._combo.count() - 1)

    def update_progress(self, current_s: float, total_s: float) -> None:
        if total_s > 0:
            self._slider.setValue(int(current_s / total_s * 1000))
        self._lbl_time.setText(f"{_fmt(current_s)} / {_fmt(total_s)}")

    def set_playing(self, playing: bool) -> None:
        self._btn_play.setChecked(playing)
        self._btn_play.setText("⏸  Pause" if playing else "▶  Play")

    def set_lap(self, lap: int, total: int) -> None:
        self._lbl_lap.setText(f"Lap {lap}/{total}" if total else "")

    # ── internal ───────────────────────────────────────────────────────────

    def _on_session_changed(self, idx: int) -> None:
        path = self._combo.itemData(idx)
        if path:
            self.session_selected.emit(path)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

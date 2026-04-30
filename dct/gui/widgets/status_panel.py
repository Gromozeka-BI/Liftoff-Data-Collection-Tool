"""Status panel: recording indicator + telemetry stats with tooltips."""
from __future__ import annotations

import math
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget

from dct.gui import theme

_TOOLTIPS = {
    "rec":      "Session recording status",
    "duration": "Elapsed recording time (seconds)",
    "packets":  "Total UDP packets received from LiftOff simulator",
    "hz":       "Actual telemetry sample rate (target: 100 Hz)",
    "laps":     "Number of completed laps detected by mock RotorHazard",
    "dropped":  "UDP packets dropped due to full receive queue",
    "speed":    "3D drone speed magnitude: sqrt(vx^2 + vy^2 + vz^2)  [m/s]",
    "alt":      "Drone altitude — Y axis in LiftOff world coordinates [m]",
    "bat":      "Battery voltage reported by LiftOff (requires battery sim) [V]",
    "pos":      "Drone world position in LiftOff coordinate system [m]",
}


def _lbl(text: str, tooltip_key: str, style: str = "") -> QLabel:
    w = QLabel(text)
    w.setToolTip(_TOOLTIPS.get(tooltip_key, ""))
    if style:
        w.setStyleSheet(style)
    return w


class StatusPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(6)
        grid.setContentsMargins(4, 4, 4, 4)

        # Row 0 — recording state + duration
        self._lbl_rec  = _lbl("● IDLE", "rec",
                               f"color: {theme.DIM}; font-weight: 700; font-size: 13px;")
        self._lbl_dur  = _lbl("0.0 s",  "duration",
                               f"color: {theme.DIM}; font-size: 12px;")
        grid.addWidget(self._lbl_rec, 0, 0, 1, 2)
        grid.addWidget(self._lbl_dur, 0, 2, 1, 2, Qt.AlignmentFlag.AlignRight)

        # Row 1 — packet stats
        grid.addWidget(QLabel("Pkts"), 1, 0)
        self._lbl_pkts = _lbl("0",    "packets"); grid.addWidget(self._lbl_pkts, 1, 1)
        grid.addWidget(QLabel("Hz"),   1, 2)
        self._lbl_hz   = _lbl("0.0",  "hz");      grid.addWidget(self._lbl_hz,   1, 3)

        # Row 2 — laps + dropped
        grid.addWidget(QLabel("Laps"), 2, 0)
        self._lbl_laps = _lbl("0",    "laps");    grid.addWidget(self._lbl_laps, 2, 1)
        grid.addWidget(QLabel("Drop"), 2, 2)
        self._lbl_drop = _lbl("0",    "dropped"); grid.addWidget(self._lbl_drop, 2, 3)

        # Row 3 — telemetry values
        grid.addWidget(QLabel("Spd"),  3, 0)
        self._lbl_spd  = _lbl("— m/s","speed");   grid.addWidget(self._lbl_spd,  3, 1)
        grid.addWidget(QLabel("Alt"),  3, 2)
        self._lbl_alt  = _lbl("— m",  "alt");     grid.addWidget(self._lbl_alt,  3, 3)

        # Row 4 — battery + position
        grid.addWidget(QLabel("Bat"),  4, 0)
        self._lbl_bat  = _lbl("— V",  "bat");     grid.addWidget(self._lbl_bat,  4, 1)
        grid.addWidget(QLabel("Pos"),  4, 2)
        self._lbl_pos  = _lbl("—",    "pos");     grid.addWidget(self._lbl_pos,  4, 3)

        # Подписи-заголовки: dim + тот же тултип, что и у соответствующего значения
        _static_key = {"Pkts": "packets", "Hz": "hz", "Laps": "laps", "Drop": "dropped",
                       "Spd": "speed", "Alt": "alt", "Bat": "bat", "Pos": "pos"}
        for i in range(grid.count()):
            w = grid.itemAt(i).widget()
            if isinstance(w, QLabel) and w.text() in _static_key:
                w.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
                w.setToolTip(_TOOLTIPS.get(_static_key[w.text()], ""))

        # Фиксируем минимальные ширины динамических меток — предотвращает скачки панели
        for lbl in (self._lbl_pkts, self._lbl_hz, self._lbl_laps, self._lbl_drop):
            lbl.setMinimumWidth(52)
        self._lbl_spd.setMinimumWidth(70)
        self._lbl_alt.setMinimumWidth(52)
        self._lbl_bat.setMinimumWidth(52)
        self._lbl_pos.setMinimumWidth(140)
        self._lbl_dur.setMinimumWidth(60)

    # ── public API ─────────────────────────────────────────────────────────

    def set_recording(self, active: bool) -> None:
        if active:
            self._lbl_rec.setText("● REC")
            self._lbl_rec.setStyleSheet(
                f"color: {theme.ERR}; font-weight: 700; font-size: 13px;"
            )
        else:
            self._lbl_rec.setText("● IDLE")
            self._lbl_rec.setStyleSheet(
                f"color: {theme.DIM}; font-weight: 700; font-size: 13px;"
            )

    def update_stats(self, stats: dict[str, Any]) -> None:
        self._lbl_dur.setText(f"{stats.get('duration', 0):.1f} s")
        self._lbl_pkts.setText(str(stats.get("packets", 0)))
        self._lbl_hz.setText(f"{stats.get('hz', 0):.1f}")
        self._lbl_laps.setText(str(stats.get("laps", 0)))
        drop = stats.get("dropped", 0)
        self._lbl_drop.setText(str(drop))
        self._lbl_drop.setStyleSheet(
            f"color: {theme.ERR};" if drop > 0 else f"color: {theme.TEXT};"
        )

    def update_telemetry(self, frame: dict[str, Any]) -> None:
        spd = math.sqrt(frame["vel_x"]**2 + frame["vel_y"]**2 + frame["vel_z"]**2)
        self._lbl_spd.setText(f"{spd:.1f} m/s")
        self._lbl_alt.setText(f"{frame['pos_y']:.1f} m")
        self._lbl_bat.setText(f"{frame['bat_v']:.1f} V")
        self._lbl_pos.setText(
            f"X{frame['pos_x']:.1f} Y{frame['pos_y']:.1f} Z{frame['pos_z']:.1f}"
        )

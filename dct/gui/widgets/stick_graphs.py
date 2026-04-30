"""Five real-time stick graphs: T / Y / P / R + combined, sliding 10-second window."""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from dct.gui import theme

_CHANNELS = [
    ("T", "in_throttle", theme.STICK_T),
    ("Y", "in_yaw",      theme.STICK_Y),
    ("P", "in_pitch",    theme.STICK_P),
    ("R", "in_roll",     theme.STICK_R),
]
_WINDOW = 10.0   # секунд истории в окне
_MAXPTS = 1200   # 10 s × 120 Hz с запасом


class StickGraphsWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self._ts: deque[float]             = deque(maxlen=_MAXPTS)
        self._buf: dict[str, deque[float]] = {f: deque(maxlen=_MAXPTS) for _, f, _ in _CHANNELS}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._combined: dict[str, pg.PlotDataItem] = {}
        self._last_redraw = 0.0   # throttle: рисуем не чаще 20fps
        self._t_zero: float = 0.0  # ts_wall начала сессии; 0 = не задан

        # Отдельные графики каналов
        for label, field, color in _CHANNELS:
            pw = self._make_plot(label, color, show_x=False)
            curve = pw.plot([], [], pen=pg.mkPen(color, width=1.5))
            self._curves[field] = curve
            layout.addWidget(pw)

        # Совмещённый график (все 4 канала, ось X видна)
        pw_comb = self._make_plot("all", theme.DIM, show_x=True)
        for label, field, color in _CHANNELS:
            c = pw_comb.plot([], [], pen=pg.mkPen(color, width=1.2))
            self._combined[field] = c
        layout.addWidget(pw_comb)

    # ── public ─────────────────────────────────────────────────────────────

    def set_time_zero(self, ts_wall: float) -> None:
        """Устанавливает t=0 для оси времени. Вызывать при старте/выборе сессии."""
        self._t_zero = ts_wall

    def update_batch(self, frames: list[dict[str, Any]]) -> None:
        for frame in frames:
            self._ts.append(frame["ts_wall"])
            for _, field, _ in _CHANNELS:
                self._buf[field].append(frame.get(field, 0.0))
        # Throttle: не чаще 20fps
        now = time.monotonic()
        if now - self._last_redraw >= 0.050:
            self._redraw()
            self._last_redraw = now

    def clear(self) -> None:
        self._ts.clear()
        for buf in self._buf.values():
            buf.clear()
        self._last_redraw = 0.0
        self._t_zero = 0.0
        self._redraw()

    # ── internal ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if not self._ts:
            for c in list(self._curves.values()) + list(self._combined.values()):
                c.setData([], [])
            return

        ts_arr = np.array(self._ts)

        # Если t_zero не задан — берём первую точку буфера
        t_zero = self._t_zero if self._t_zero > 0 else ts_arr[0]
        t_abs = ts_arr - t_zero  # время в секундах от начала сессии, [0 .. N]

        # Скользящее окно: последние _WINDOW секунд
        t_now = t_abs[-1]
        mask = t_abs >= (t_now - _WINDOW)
        t_plot = t_abs[mask]  # всегда ≥ 0, растёт слева направо

        for _, field, _ in _CHANNELS:
            data = np.array(self._buf[field])[mask]
            self._curves[field].setData(t_plot, data)
            self._combined[field].setData(t_plot, data)

    @staticmethod
    def _make_plot(label: str, color: str, show_x: bool) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(theme.PANEL)
        pw.setMinimumHeight(60)
        pw.setMaximumHeight(90)
        pw.showGrid(x=False, y=True, alpha=0.15)

        # Фиксируем Y-диапазон и отключаем автомасштаб/мышиный зум
        pw.setYRange(-1.05, 1.05, padding=0)
        pw.enableAutoRange(axis="y", enable=False)
        pw.setMouseEnabled(x=False, y=False)
        pw.setMenuEnabled(False)

        ax_left = pw.getPlotItem().getAxis("left")
        ax_left.setLabel(label, color=color)
        ax_left.setWidth(28)
        ax_left.setTextPen(pg.mkPen(color))
        ax_left.setTicks([[(1, "1"), (0, "0"), (-1, "-1")]])

        if not show_x:
            pw.getPlotItem().hideAxis("bottom")
        else:
            ax_bot = pw.getPlotItem().getAxis("bottom")
            ax_bot.setLabel("t (s)", color=theme.DIM)
            ax_bot.setTextPen(pg.mkPen(theme.DIM))

        return pw

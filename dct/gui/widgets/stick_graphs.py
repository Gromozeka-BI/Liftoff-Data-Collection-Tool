"""Stick graphs: Liftoff (T/Y/P/R) + RC (ch1-4) + switch indicators (ch5-8).

Layout
------
[⇔ Merge Liftoff+RC]          ← full-width toggle button
[T  |  Thr ]  ←─ row 0        ← 4 rows, same height both sides
[Y  |  Yaw ]  ←─ row 1
[P  |  Pit ]  ←─ row 2
[R  |  Rol ]  ←─ row 3
[  Combined (all LF+RC)  ]    ← full-width, bottom axis visible
[Ch5 Ch6 Ch7 Ch8]             ← full-width switch indicators

Merge mode: RC column hidden; RC curves overlaid (dashed) on LF plots.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from dct.gui import theme

# ── Channel tables ─────────────────────────────────────────────────────────────

# LF channels: (label, frame_key, color)
_LF = [
    ("T", "in_throttle", theme.STICK_T),
    ("Y", "in_yaw",      theme.STICK_Y),
    ("P", "in_pitch",    theme.STICK_P),
    ("R", "in_roll",     theme.STICK_R),
]

# RC channels paired with LF by meaning: T↔ch3, Y↔ch4, P↔ch2, R↔ch1
_RC = [
    ("Thr", "ch3", "#ff9955"),
    ("Yaw", "ch4", "#99ddff"),
    ("Pit", "ch2", "#99ff99"),
    ("Rol", "ch1", "#ffdd99"),
]

_SW_LOW  = "#cc3333"   # < 1300
_SW_MID  = "#ccaa00"   # 1300–1700
_SW_HIGH = "#33aa33"   # > 1700

_WINDOW    = 10.0
_MAXPTS    = 1200
_RC_Y_MIN  = 800
_RC_Y_MAX  = 2200
_RC_CENTER = 1500.0
_RC_HALF   = 500.0
_PLOT_H    = 70        # fixed height for every individual plot row


class StickGraphsWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # ── Data buffers ────────────────────────────────────────────────────
        self._ts:    deque[float]            = deque(maxlen=_MAXPTS)
        self._rc_ts: deque[float]            = deque(maxlen=_MAXPTS)
        self._buf:    dict[str, deque[float]] = {f: deque(maxlen=_MAXPTS) for _, f, _ in _LF}
        self._rc_buf: dict[str, deque[float]] = {f: deque(maxlen=_MAXPTS) for _, f, _ in _RC}
        self._sw:     dict[str, int]          = {f"ch{i}": 1500 for i in range(5, 9)}

        # ── Plot / curve registries ──────────────────────────────────────────
        self._lf_plots:   list[pg.PlotWidget]       = []   # 4 individual LF
        self._rc_plots:   list[pg.PlotWidget]       = []   # 4 individual RC
        self._all_plots:  list[pg.PlotWidget]       = []   # everything (for XRange sync)
        self._lf_curves:  dict[str, pg.PlotDataItem] = {}  # individual LF
        self._rc_curves:  dict[str, pg.PlotDataItem] = {}  # individual RC
        self._comb_lf:    dict[str, pg.PlotDataItem] = {}  # LF on combined plot
        self._comb_rc:    dict[str, pg.PlotDataItem] = {}  # RC on combined plot (dashed)
        self._rc_merged:  dict[str, pg.PlotDataItem] = {}  # RC on LF plots (merge mode)

        self._merged      = False
        self._last_redraw = 0.0
        self._t_zero:     float = 0.0

        # ── Root layout ─────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Row 0: Merge button (full width) ─────────────────────────────
        self._btn_merge = QPushButton("⇔ Merge Liftoff + RC")
        self._btn_merge.setCheckable(True)
        self._btn_merge.setFixedHeight(24)
        self._btn_merge.setToolTip("Overlay RC channels on Liftoff plots (normalised to −1…1)")
        self._btn_merge.clicked.connect(self._toggle_merge)
        root.addWidget(self._btn_merge)

        # ── Row 1: 4×(LF | RC) pairs ─────────────────────────────────────
        pairs_row = QHBoxLayout()
        pairs_row.setSpacing(4)
        pairs_row.setContentsMargins(0, 0, 0, 0)

        lf_col = QVBoxLayout()
        lf_col.setSpacing(2)
        lf_col.setContentsMargins(0, 0, 0, 0)

        self._rc_col_widget = QWidget()
        rc_col = QVBoxLayout(self._rc_col_widget)
        rc_col.setSpacing(2)
        rc_col.setContentsMargins(0, 0, 0, 0)

        # Build 4 paired rows
        for i, ((lf_lbl, lf_fld, lf_clr), (rc_lbl, rc_fld, rc_clr)) in enumerate(zip(_LF, _RC)):
            # LF plot (no bottom axis)
            lf_pw = self._make_plot(_PLOT_H, y_lf=True, show_x=False)
            lf_c  = lf_pw.plot([], [], pen=pg.mkPen(lf_clr, width=1.5))
            self._lf_curves[lf_fld] = lf_c
            self._lf_plots.append(lf_pw)
            self._all_plots.append(lf_pw)
            lf_col.addWidget(lf_pw)
            _set_axis_label(lf_pw, lf_lbl, lf_clr, side="left",
                            ticks=[(1, "1"), (0, "0"), (-1, "-1")])

            # RC overlay curve on this LF plot (hidden by default)
            rc_m = lf_pw.plot([], [],
                              pen=pg.mkPen(rc_clr, width=1.5, style=Qt.PenStyle.DashLine),
                              name=rc_lbl)
            rc_m.setVisible(False)
            self._rc_merged[rc_fld] = rc_m

            # RC plot
            rc_pw = self._make_plot(_PLOT_H, y_lf=False, show_x=False)
            rc_c  = rc_pw.plot([], [], pen=pg.mkPen(rc_clr, width=1.5))
            self._rc_curves[rc_fld] = rc_c
            self._rc_plots.append(rc_pw)
            self._all_plots.append(rc_pw)
            rc_col.addWidget(rc_pw)
            _set_axis_label(rc_pw, rc_lbl, rc_clr, side="left",
                            ticks=[(2000, "2k"), (1500, "ctr"), (1000, "1k")])

        pairs_row.addLayout(lf_col, stretch=1)
        pairs_row.addWidget(self._rc_col_widget, stretch=1)
        root.addLayout(pairs_row)

        # ── Row 2: Combined plot (full width, bottom axis visible) ───────
        self._comb_pw = self._make_plot(_PLOT_H + 10, y_lf=True, show_x=True)
        for _, fld, clr in _LF:
            self._comb_lf[fld] = self._comb_pw.plot([], [], pen=pg.mkPen(clr, width=1.2))
        for _, fld, clr in _RC:
            c = self._comb_pw.plot([], [],
                                    pen=pg.mkPen(clr, width=1.2, style=Qt.PenStyle.DashLine))
            c.setVisible(False)
            self._comb_rc[fld] = c
        self._all_plots.append(self._comb_pw)
        _set_axis_label(self._comb_pw, "all", theme.DIM, side="left", ticks=None)
        ab = self._comb_pw.getPlotItem().getAxis("bottom")
        ab.setLabel("t (s)", color=theme.DIM)
        ab.setTextPen(pg.mkPen(theme.DIM))
        root.addWidget(self._comb_pw)

        # ── Row 3: Switch indicators (full width) ────────────────────────
        sw_row = QHBoxLayout()
        sw_row.setSpacing(8)
        sw_row.setContentsMargins(4, 2, 4, 2)
        self._sw_dots: dict[str, QLabel] = {}
        for ch in ("ch5", "ch6", "ch7", "ch8"):
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(f"color: {_SW_MID}; font-size: 18px;")
            dot.setToolTip(ch)
            name = QLabel(ch)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
            cell = QVBoxLayout()
            cell.setSpacing(0)
            cell.addWidget(dot)
            cell.addWidget(name)
            sw_row.addLayout(cell)
            self._sw_dots[ch] = dot
        root.addLayout(sw_row)

    # ── public ─────────────────────────────────────────────────────────────

    def set_time_zero(self, ts_wall: float) -> None:
        self._t_zero = ts_wall

    def update_batch(self, frames: list[dict[str, Any]]) -> None:
        for frame in frames:
            self._ts.append(frame["ts_wall"])
            for _, fld, _ in _LF:
                self._buf[fld].append(frame.get(fld, 0.0))
        self._throttled_redraw()

    def update_rc_batch(self, frames: list[dict]) -> None:
        for frame in frames:
            self._rc_ts.append(frame["ts_wall"])
            for _, fld, _ in _RC:
                self._rc_buf[fld].append(float(frame.get(fld, 1500)))
            for ch in ("ch5", "ch6", "ch7", "ch8"):
                self._sw[ch] = int(frame.get(ch, 1500))
        self._update_switches()
        self._throttled_redraw()

    def clear(self) -> None:
        self._ts.clear()
        self._rc_ts.clear()
        for buf in list(self._buf.values()) + list(self._rc_buf.values()):
            buf.clear()
        self._last_redraw = 0.0
        self._t_zero      = 0.0
        for pw in self._all_plots:
            pw.getPlotItem().getViewBox().setRange(xRange=(0, _WINDOW), padding=0)
        self._redraw()
        self._update_switches()

    # ── internal ───────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-apply xRange after any geometry change (fullscreen, monitor change)
        # so pyqtgraph doesn't reset or auto-scale the viewport.
        self._redraw()

    def _throttled_redraw(self) -> None:
        now = time.monotonic()
        if now - self._last_redraw >= 0.050:
            self._redraw()
            self._last_redraw = now

    def _toggle_merge(self) -> None:
        self._merged = self._btn_merge.isChecked()
        self._rc_col_widget.setVisible(not self._merged)
        for c in self._rc_merged.values():
            c.setVisible(self._merged)
        for c in self._comb_rc.values():
            c.setVisible(self._merged)
        self._redraw()

    def _update_switches(self) -> None:
        for ch, dot in self._sw_dots.items():
            v     = self._sw.get(ch, 1500)
            color = _SW_LOW if v < 1300 else (_SW_HIGH if v > 1700 else _SW_MID)
            dot.setStyleSheet(f"color: {color}; font-size: 18px;")

    def _redraw(self) -> None:
        # ── Single reference time for both sources ────────────────────────
        # Use _t_zero if set; otherwise fall back to earliest available sample.
        t0 = self._t_zero
        if t0 <= 0:
            candidates = []
            if self._ts:    candidates.append(self._ts[0])
            if self._rc_ts: candidates.append(self._rc_ts[0])
            t0 = min(candidates) if candidates else 0.0

        # ── Unified time window (latest timestamp across both sources) ────
        t_now = 0.0
        if self._ts:
            t_now = max(t_now, self._ts[-1] - t0)
        if self._rc_ts:
            t_now = max(t_now, self._rc_ts[-1] - t0)
        x_lo = max(0.0, t_now - _WINDOW)
        x_hi = max(t_now, _WINDOW)

        for pw in self._all_plots:
            pw.getPlotItem().getViewBox().setRange(xRange=(x_lo, x_hi), padding=0)

        # ── LF curves ─────────────────────────────────────────────────────
        if self._ts:
            ts_arr = np.array(self._ts)
            t_abs  = ts_arr - t0
            mask   = t_abs >= x_lo
            t_plt  = t_abs[mask]
            for _, fld, _ in _LF:
                d = np.array(self._buf[fld])[mask]
                self._lf_curves[fld].setData(t_plt, d)
                self._comb_lf[fld].setData(t_plt, d)
        else:
            for c in list(self._lf_curves.values()) + list(self._comb_lf.values()):
                c.setData([], [])

        # ── RC curves ─────────────────────────────────────────────────────
        if self._rc_ts:
            rc_arr  = np.array(self._rc_ts)
            rc_t    = rc_arr - t0
            rc_mask = rc_t >= x_lo
            rc_plt  = rc_t[rc_mask]
            for _, fld, _ in _RC:
                raw    = np.array(self._rc_buf[fld])[rc_mask]
                self._rc_curves[fld].setData(rc_plt, raw)
                normed = (raw - _RC_CENTER) / _RC_HALF
                self._rc_merged[fld].setData(rc_plt if self._merged else [], normed if self._merged else [])
                self._comb_rc[fld].setData(rc_plt if self._merged else [], normed if self._merged else [])
        else:
            for c in (list(self._rc_curves.values()) +
                      list(self._rc_merged.values()) +
                      list(self._comb_rc.values())):
                c.setData([], [])

    # ── plot factory ───────────────────────────────────────────────────────

    @staticmethod
    def _make_plot(height: int, y_lf: bool, show_x: bool) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(theme.PANEL)
        pw.setFixedHeight(height)
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.setMouseEnabled(x=False, y=False)
        pw.setMenuEnabled(False)
        vb = pw.getPlotItem().getViewBox()
        vb.disableAutoRange()
        vb.setAutoVisible(x=False, y=False)
        y_range = (-1.05, 1.05) if y_lf else (_RC_Y_MIN, _RC_Y_MAX)
        vb.setRange(yRange=y_range, xRange=(0, _WINDOW), padding=0)
        if not show_x:
            pw.getPlotItem().hideAxis("bottom")
        return pw


def _set_axis_label(
    pw: pg.PlotWidget,
    label: str,
    color: str,
    side: str,
    ticks: list | None,
) -> None:
    ax = pw.getPlotItem().getAxis(side)
    ax.setLabel(label, color=color)
    ax.setWidth(28)
    ax.setTextPen(pg.mkPen(color))
    if ticks:
        ax.setTicks([ticks])

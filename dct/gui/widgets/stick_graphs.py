"""Stick graphs: Liftoff (T/Y/P/R) + RC (ch1-4) + switch indicators.

Layout
------
[⇔ Merge Liftoff+RC]                              ← full-width toggle
["Liftoff" header]  ["RC" header]  [Reverse panel]
[T plot ]  [Thr plot]              [T ☐] [Thr ☐]
[Y plot ]  [Yaw plot]              [Y ☐] [Yaw ☐]
[P plot ]  [Pit plot]              [P ☐] [Pit ☐]
[R plot ]  [Rol plot]              [R ☐] [Rol ☐]
[  Combined (all LF+RC)  ]         ← full-width
[CH5-Arm  CH6-Turtle  CH7-Option  CH8-Rate]

Merge mode: RC plot column hidden; RC curves overlaid (dashed) on LF plots.
Both LF and RC reverse checkboxes remain visible in merge mode.
Invert checkboxes flip the DISPLAYED sign only; recorded data is unchanged.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from dct.gui import theme
from dct.gui import ui_settings

# ── Channel tables ─────────────────────────────────────────────────────────────

_LF = [
    ("T", "in_throttle", theme.STICK_T),
    ("Y", "in_yaw",      theme.STICK_Y),
    ("P", "in_pitch",    theme.STICK_P),
    ("R", "in_roll",     theme.STICK_R),
]

_RC = [
    ("Thr", "ch3", "#ff9955"),
    ("Yaw", "ch4", "#99ddff"),
    ("Pit", "ch2", "#99ff99"),
    ("Rol", "ch1", "#ffdd99"),
]

_SW_CHANNELS = [
    ("ch5", "CH5 - Arm"),
    ("ch6", "CH6 - Turtle"),
    ("ch7", "CH7 - Option"),
    ("ch8", "CH8 - Rate"),
]

_SW_LOW  = "#cc3333"
_SW_MID  = "#ccaa00"
_SW_HIGH = "#33aa33"

_WINDOW    = 10.0
_MAXPTS    = 1200
_RC_Y_MIN  = 800
_RC_Y_MAX  = 2200
_RC_CENTER = 1500.0
_RC_HALF   = 500.0
_PLOT_H    = 70


class StickGraphsWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)   # prevent layout collapse on narrow windows

        # ── Data buffers ────────────────────────────────────────────────────
        self._ts:    deque[float]             = deque(maxlen=_MAXPTS)
        self._rc_ts: deque[float]             = deque(maxlen=_MAXPTS)
        self._buf:    dict[str, deque[float]] = {f: deque(maxlen=_MAXPTS) for _, f, _ in _LF}
        self._rc_buf: dict[str, deque[float]] = {f: deque(maxlen=_MAXPTS) for _, f, _ in _RC}
        self._sw:     dict[str, int]          = {ch: 1500 for ch, _ in _SW_CHANNELS}

        # ── Registries ──────────────────────────────────────────────────────
        self._lf_plots:   list[pg.PlotWidget]        = []
        self._rc_plots:   list[pg.PlotWidget]        = []
        self._all_plots:  list[pg.PlotWidget]        = []
        self._lf_curves:  dict[str, pg.PlotDataItem] = {}
        self._rc_curves:  dict[str, pg.PlotDataItem] = {}
        self._comb_lf:    dict[str, pg.PlotDataItem] = {}
        self._comb_rc:    dict[str, pg.PlotDataItem] = {}
        self._rc_merged:  dict[str, pg.PlotDataItem] = {}
        self._lf_inv:     dict[str, QCheckBox]       = {}
        self._rc_inv:     dict[str, QCheckBox]       = {}

        self._merged      = False
        self._last_redraw = 0.0
        self._t_zero:     float = 0.0

        # ── Root layout ─────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Merge toggle ─────────────────────────────────────────────────
        self._btn_merge = QPushButton("⇔ Merge Liftoff + RC")
        self._btn_merge.setCheckable(True)
        self._btn_merge.setFixedHeight(24)
        self._btn_merge.setToolTip("Overlay RC channels on Liftoff plots (normalised to −1…1)")
        self._btn_merge.clicked.connect(self._toggle_merge)
        root.addWidget(self._btn_merge)

        # ── Main row: [LF col] [RC col] [Reverse panel] ──────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(4)
        main_row.setContentsMargins(0, 0, 0, 0)

        # ── LF column ────────────────────────────────────────────────────
        lf_outer = QWidget()
        lf_outer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lf_vbox = QVBoxLayout(lf_outer)
        lf_vbox.setSpacing(2)
        lf_vbox.setContentsMargins(0, 0, 0, 0)

        lf_hdr = QLabel("Liftoff")
        lf_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lf_hdr.setStyleSheet(f"color: {theme.DIM}; font-size: 10px; font-weight: bold;")
        lf_hdr.setFixedHeight(14)
        lf_vbox.addWidget(lf_hdr)

        # ── RC column (hideable in merge mode) ────────────────────────────
        self._rc_col_widget = QWidget()
        self._rc_col_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        rc_vbox = QVBoxLayout(self._rc_col_widget)
        rc_vbox.setSpacing(2)
        rc_vbox.setContentsMargins(0, 0, 0, 0)

        rc_hdr = QLabel("RC")
        rc_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rc_hdr.setStyleSheet(f"color: {theme.DIM}; font-size: 10px; font-weight: bold;")
        rc_hdr.setFixedHeight(14)
        rc_vbox.addWidget(rc_hdr)

        # ── Build 4 paired plot rows ──────────────────────────────────────
        for (lf_lbl, lf_fld, lf_clr), (rc_lbl, rc_fld, rc_clr) in zip(_LF, _RC):
            # LF plot
            lf_pw = self._make_plot(_PLOT_H, y_lf=True, show_x=False)
            lf_c  = lf_pw.plot([], [], pen=pg.mkPen(lf_clr, width=1.5))
            self._lf_curves[lf_fld] = lf_c
            self._lf_plots.append(lf_pw)
            self._all_plots.append(lf_pw)
            _set_axis_label(lf_pw, lf_lbl, lf_clr, side="left",
                            ticks=[(1, "1"), (0, "0"), (-1, "-1")])
            rc_m = lf_pw.plot([], [],
                              pen=pg.mkPen(rc_clr, width=1.5, style=Qt.PenStyle.DashLine),
                              name=rc_lbl)
            rc_m.setVisible(False)
            self._rc_merged[rc_fld] = rc_m
            lf_vbox.addWidget(lf_pw)

            # RC plot
            rc_pw = self._make_plot(_PLOT_H, y_lf=False, show_x=False)
            rc_c  = rc_pw.plot([], [], pen=pg.mkPen(rc_clr, width=1.5))
            self._rc_curves[rc_fld] = rc_c
            self._rc_plots.append(rc_pw)
            self._all_plots.append(rc_pw)
            _set_axis_label(rc_pw, rc_lbl, rc_clr, side="left",
                            ticks=[(2000, "2k"), (1500, "ctr"), (1000, "1k")])
            rc_vbox.addWidget(rc_pw)

        main_row.addWidget(lf_outer,            stretch=1)
        main_row.addWidget(self._rc_col_widget, stretch=1)

        # ── Reverse panel (always visible, both LF and RC checkboxes) ────
        inv_box = QGroupBox("Реверс")
        inv_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        inv_lay = QVBoxLayout(inv_box)
        inv_lay.setSpacing(3)
        inv_lay.setContentsMargins(6, 4, 6, 4)

        # Sub-header
        hdr_row = QHBoxLayout()
        for txt in ("LF", "RC"):
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
            hdr_row.addWidget(lbl)
        inv_lay.addLayout(hdr_row)
        inv_lay.addSpacing(2)

        for (lf_lbl, lf_fld, _), (rc_lbl, rc_fld, _) in zip(_LF, _RC):
            row = QHBoxLayout()
            row.setSpacing(4)

            lf_chk = QCheckBox(lf_lbl)
            lf_chk.setToolTip(f"Инвертировать {lf_lbl} (только отображение)")
            lf_chk.stateChanged.connect(self._on_invert_changed)
            self._lf_inv[lf_fld] = lf_chk

            rc_chk = QCheckBox(rc_lbl)
            rc_chk.setToolTip(f"Инвертировать {rc_lbl} (только отображение)")
            rc_chk.stateChanged.connect(self._on_invert_changed)
            self._rc_inv[rc_fld] = rc_chk

            row.addWidget(lf_chk)
            row.addWidget(rc_chk)
            inv_lay.addLayout(row)

        inv_lay.addStretch(1)
        main_row.addWidget(inv_box)
        root.addLayout(main_row)

        # ── Combined plot ────────────────────────────────────────────────
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

        # ── Switch indicators ────────────────────────────────────────────
        sw_row = QHBoxLayout()
        sw_row.setSpacing(8)
        sw_row.setContentsMargins(4, 2, 4, 2)
        self._sw_dots: dict[str, QLabel] = {}
        for ch_id, ch_label in _SW_CHANNELS:
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(f"color: {_SW_MID}; font-size: 18px;")
            dot.setToolTip(ch_label)
            name = QLabel(ch_label)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
            cell = QVBoxLayout()
            cell.setSpacing(0)
            cell.addWidget(dot)
            cell.addWidget(name)
            sw_row.addLayout(cell)
            self._sw_dots[ch_id] = dot
        root.addLayout(sw_row)

        # ── Restore persisted invert state ───────────────────────────────
        self.set_invert_state(ui_settings.load().get("invert", {}))

    # ── public API ─────────────────────────────────────────────────────────

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
            for ch_id, _ in _SW_CHANNELS:
                self._sw[ch_id] = int(frame.get(ch_id, 1500))
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

    # ── invert API ─────────────────────────────────────────────────────────

    def get_invert_state(self) -> dict:
        return {
            "lf": {fld: chk.isChecked() for fld, chk in self._lf_inv.items()},
            "rc": {fld: chk.isChecked() for fld, chk in self._rc_inv.items()},
        }

    def set_invert_state(self, state: dict) -> None:
        lf = state.get("lf", {})
        rc = state.get("rc", {})
        for fld, chk in self._lf_inv.items():
            chk.blockSignals(True)
            chk.setChecked(bool(lf.get(fld, False)))
            chk.blockSignals(False)
        for fld, chk in self._rc_inv.items():
            chk.blockSignals(True)
            chk.setChecked(bool(rc.get(fld, False)))
            chk.blockSignals(False)
        self._redraw()

    # ── internal ───────────────────────────────────────────────────────────

    def _on_invert_changed(self) -> None:
        self._redraw()
        ui_settings.update("invert", self.get_invert_state())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
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
        for ch_id, dot in self._sw_dots.items():
            v     = self._sw.get(ch_id, 1500)
            color = _SW_LOW if v < 1300 else (_SW_HIGH if v > 1700 else _SW_MID)
            dot.setStyleSheet(f"color: {color}; font-size: 18px;")

    def _redraw(self) -> None:
        # ── Unified reference time ─────────────────────────────────────────
        t0 = self._t_zero
        if t0 <= 0:
            candidates = []
            if self._ts:    candidates.append(self._ts[0])
            if self._rc_ts: candidates.append(self._rc_ts[0])
            t0 = min(candidates) if candidates else 0.0

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
                if self._lf_inv[fld].isChecked():
                    d = -d
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
                raw = np.array(self._rc_buf[fld])[rc_mask]
                if self._rc_inv[fld].isChecked():
                    raw = 2 * _RC_CENTER - raw   # mirror around 1500
                self._rc_curves[fld].setData(rc_plt, raw)
                normed = (raw - _RC_CENTER) / _RC_HALF
                self._rc_merged[fld].setData(rc_plt if self._merged else [],
                                             normed if self._merged else [])
                self._comb_rc[fld].setData(rc_plt if self._merged else [],
                                           normed if self._merged else [])
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
        pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

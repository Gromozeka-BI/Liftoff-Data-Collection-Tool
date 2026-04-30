"""2-D top-down track map: gates, start point, bounds, drone arrow, trail."""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from dct.gui import theme


class TrackMapWidget(pg.PlotWidget):
    TRAIL_SECS = 3.0
    TRAIL_MAX  = 400   # ~100 Hz * 4 s headroom

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_plot()
        self._gate_items: list[pg.PlotDataItem] = []
        self._gate_label_items: list[pg.TextItem] = []
        self._bounds_item: pg.PlotDataItem | None = None
        self._start_item: pg.ScatterPlotItem | None = None

        # Trail
        self._trail_x: deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_z: deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_ts: deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_item: pg.PlotDataItem = self.plot(
            [], [], pen=pg.mkPen(theme.TRAIL, width=2)
        )

        # Drone arrow (ArrowItem: angle=90 → points up in plot coords)
        self._arrow = pg.ArrowItem(
            angle=90, tipAngle=35, headLen=14, tailLen=10,
            tailWidth=4, brush=pg.mkBrush(theme.DRONE), pen=None,
        )
        self.addItem(self._arrow)
        self._arrow.setPos(0, 0)
        self._has_track = False

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup_plot(self) -> None:
        self.setBackground(theme.PANEL)
        pi = self.getPlotItem()
        pi.getAxis("bottom").setPen(pg.mkPen(theme.BORDER))
        pi.getAxis("left").setPen(pg.mkPen(theme.BORDER))
        pi.getAxis("bottom").setTextPen(pg.mkPen(theme.DIM))
        pi.getAxis("left").setTextPen(pg.mkPen(theme.DIM))
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setLabel("bottom", "X (m)", color=theme.DIM)
        pi.setLabel("left",   "Z (m)", color=theme.DIM)
        self.setAspectLocked(True)

    # ── public API ─────────────────────────────────────────────────────────

    def setup_track(self, track_data: dict[str, Any]) -> None:
        for item in self._gate_items:
            self.removeItem(item)
        for item in self._gate_label_items:
            self.removeItem(item)
        self._gate_items.clear()
        self._gate_label_items.clear()
        if self._bounds_item:
            self.removeItem(self._bounds_item)
            self._bounds_item = None
        if self._start_item:
            self.removeItem(self._start_item)
            self._start_item = None

        gates = track_data.get("gates", [])
        for gate in gates:
            is_sf = gate.get("is_start_finish", False)
            color = theme.GATE_SF if is_sf else theme.GATE
            xs, zs = self._gate_rect(gate)
            item = self.plot(xs, zs, pen=pg.mkPen(color, width=2.5))
            self._gate_items.append(item)

            # Gate label
            gx, _, gz = gate["position"]
            lbl = pg.TextItem(
                text=str(gate["id"]), color=color, anchor=(0.5, 0.5)
            )
            lbl.setPos(gx, gz)
            self.addItem(lbl)
            self._gate_label_items.append(lbl)

        # Bounds rectangle (dashed)
        bounds = track_data.get("bounds")
        bx = bz = ox = oz = 0.0
        if bounds:
            bx = float(bounds.get("x", 0))
            bz = float(bounds.get("y", bounds.get("z", 0)))
            ox = float(bounds.get("origin_x", 0.0))
            oz = float(bounds.get("origin_z", 0.0))
            bxs = [ox, ox+bx, ox+bx, ox, ox]
            bzs = [oz, oz, oz+bz, oz+bz, oz]
            self._bounds_item = self.plot(
                bxs, bzs,
                pen=pg.mkPen(theme.BORDER, width=1, style=Qt.PenStyle.DashLine),
            )

        # Start point marker (star)
        # Поддерживаем два формата: {"x":…, "z":…} и {"position": [x, y, z]}
        sp = track_data.get("start_point")
        sp_x: float | None = None
        sp_z: float | None = None
        if sp:
            if "position" in sp:
                sp_x, _, sp_z = sp["position"]
            else:
                sp_x = float(sp.get("x", 0))
                sp_z = float(sp.get("z", sp.get("y", 0)))
            self._start_item = pg.ScatterPlotItem(
                [sp_x], [sp_z], symbol="star", size=18,
                brush=pg.mkBrush(theme.GATE_SF), pen=pg.mkPen(None),
            )
            self.addItem(self._start_item)

        # Auto-fit view: use explicit bounds if provided, else gate extents
        fit_xs: list[float] = []
        fit_zs: list[float] = []
        if bounds and bx > 0 and bz > 0:
            fit_xs = [ox, ox + bx]
            fit_zs = [oz, oz + bz]
        elif gates:
            fit_xs = [g["position"][0] for g in gates]
            fit_zs = [g["position"][2] for g in gates]
        if sp_x is not None and sp_z is not None:
            fit_xs.append(sp_x)
            fit_zs.append(sp_z)
        if fit_xs and fit_zs:
            pad = max(2.0, (max(fit_xs) - min(fit_xs)) * 0.08)
            self.setXRange(min(fit_xs) - pad, max(fit_xs) + pad, padding=0)
            self.setYRange(min(fit_zs) - pad, max(fit_zs) + pad, padding=0)
        self._has_track = True

    def update_drone(self, frame: dict[str, Any]) -> None:
        px = frame["pos_x"]
        pz = frame["pos_z"]
        ts = frame["ts_wall"]

        # Append to trail
        self._trail_x.append(px)
        self._trail_z.append(pz)
        self._trail_ts.append(ts)

        # Time-based trim
        cutoff = ts - self.TRAIL_SECS
        while self._trail_ts and self._trail_ts[0] < cutoff:
            self._trail_x.popleft()
            self._trail_z.popleft()
            self._trail_ts.popleft()

        self._trail_item.setData(list(self._trail_x), list(self._trail_z))

        # Drone arrow
        yaw_deg = self._quat_yaw(
            frame["att_x"], frame["att_y"], frame["att_z"], frame["att_w"]
        )
        # pyqtgraph ArrowItem: angle=0 → right, angle=90 → up
        # LiftOff yaw=0 → drone faces +Z (up in our XZ plot)
        self._arrow.setPos(px, pz)
        self._arrow.setStyle(angle=90 - yaw_deg)

    def clear_trail(self) -> None:
        self._trail_x.clear()
        self._trail_z.clear()
        self._trail_ts.clear()
        self._trail_item.setData([], [])

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _gate_rect(gate: dict[str, Any]) -> tuple[list[float], list[float]]:
        px, _py, pz = gate["position"]
        ry = math.radians(gate["rotation"][1])
        hw = gate["size"][0] / 2          # half-width
        depth = max(0.12, gate["size"][0] * 0.06)  # thin visual depth
        corners = [(-hw, -depth), (hw, -depth), (hw, depth), (-hw, depth), (-hw, -depth)]
        cos_r, sin_r = math.cos(ry), math.sin(ry)
        xs = [px + cx * cos_r - cz * sin_r for cx, cz in corners]
        zs = [pz + cx * sin_r + cz * cos_r for cx, cz in corners]
        return xs, zs

    @staticmethod
    def _quat_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract yaw (degrees) around Y-up axis from Unity quaternion."""
        yaw_rad = math.atan2(
            2.0 * (qw * qy + qx * qz),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return math.degrees(yaw_rad)

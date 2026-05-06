"""Thin timeline strip showing coloured event markers for the Replay editor.

Now also supports dragging editable markers along the timeline with a
configurable snap policy (telemetry sample / RC sample / playhead) plus
modifier overrides (Shift/Ctrl/Alt).
"""
from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt6.QtWidgets import QWidget, QToolTip

from dct.gui.replay_snap import snap_ts

_HEIGHT = 22
_HIT_PX = 8
_S = 7
_DRAG_THRESHOLD_PX = 4

_EDITABLE_TYPES = {"button_lap", "button_gate", "rh_lap", "rh_gate"}

_COLORS: dict[str, str] = {
    "session_start": "#00cc44",
    "session_stop":  "#dd2200",
    "button_lap":    "#3399ff",
    "button_gate":   "#ffcc00",
    "rh_lap":        "#ff8800",
    "rh_gate":       "#cc44ff",
}
_DEFAULT_COLOR = "#888888"

_LEGEND_HTML = (
    "<b>Event legend</b><br>"
    "<span style='color:#00cc44;'>▲</span> session_start &nbsp; "
    "<span style='color:#dd2200;'>▼</span> session_stop<br>"
    "<span style='color:#3399ff;'>●</span> button_lap &nbsp; "
    "<span style='color:#ffcc00;'>◆</span> button_gate<br>"
    "<span style='color:#ff8800;'>▲</span> rh_lap (S/F) &nbsp; "
    "<span style='color:#cc44ff;'>◇</span> rh_gate"
)


def _color(event_type: str) -> QColor:
    return QColor(_COLORS.get(event_type, _DEFAULT_COLOR))


class EventStrip(QWidget):
    marker_clicked        = pyqtSignal(dict)
    marker_selected       = pyqtSignal(object)   # dict | None
    marker_drag_started   = pyqtSignal(dict)
    marker_dragging       = pyqtSignal(int, float)
    marker_drag_ended     = pyqtSignal(int, float)
    marker_drag_cancelled = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(_HEIGHT)
        self.setMouseTracking(True)
        self._events:  list[dict] = []
        self._t0 = 0.0
        self._t1 = 1.0
        self._playhead_ts: float | None = None
        self._selected_seq: int | None  = None

        # Drag state
        self._drag_seq: int | None = None
        self._drag_start_x: int = 0
        self._dragging: bool = False
        self._drag_orig_ts: float = 0.0
        self._tl_ts: list[float] | None = None
        self._rc_ts: list[float] | None = None

    # ── public API ─────────────────────────────────────────────────────────

    def set_events(self, events: list[dict], t0: float, t1: float) -> None:
        self._events = list(events)
        self._t0 = t0
        self._t1 = max(t1, t0 + 0.001)
        self.update()

    def set_playhead(self, ts_wall: float) -> None:
        self._playhead_ts = ts_wall
        self.update()

    def set_snap_sources(
        self,
        *,
        tl_ts: Sequence[float] | None = None,
        rc_ts: Sequence[float] | None = None,
    ) -> None:
        self._tl_ts = list(tl_ts) if tl_ts is not None else None
        self._rc_ts = list(rc_ts) if rc_ts is not None else None

    def selected_event(self) -> dict | None:
        if self._selected_seq is None:
            return None
        for ev in self._events:
            if ev.get("seq") == self._selected_seq:
                return ev
        return None

    def clear_selection(self) -> None:
        self._selected_seq = None
        self.update()

    # ── internal ───────────────────────────────────────────────────────────

    def _x(self, ts: float) -> float:
        if self._t1 <= self._t0:
            return 0.0
        return (ts - self._t0) / (self._t1 - self._t0) * self.width()

    def _ts_for_x(self, px: float) -> float:
        if self.width() <= 0:
            return self._t0
        frac = max(0.0, min(1.0, px / self.width()))
        return self._t0 + frac * (self._t1 - self._t0)

    def _event_at(self, px: int) -> dict | None:
        best: dict | None = None
        best_d = _HIT_PX
        for ev in self._events:
            if ev.get("event_type") in {"session_start", "session_stop"}:
                continue
            d = abs(self._x(ev["ts_wall"]) - px)
            if d < best_d:
                best_d = d
                best = ev
        return best

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = int(event.position().x())
        ev = self._event_at(x)
        self._drag_seq = None
        self._dragging = False
        if ev is not None:
            self._selected_seq = ev.get("seq")
            self._drag_start_x = x
            self._drag_orig_ts = float(ev["ts_wall"])
            if ev.get("event_type") in _EDITABLE_TYPES:
                self._drag_seq = int(ev.get("seq", -1))
            self.marker_clicked.emit(ev)
            self.marker_selected.emit(ev)
        else:
            self._selected_seq = None
            self.marker_selected.emit(None)
        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        x = int(event.position().x())

        if self._drag_seq is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if not self._dragging and abs(x - self._drag_start_x) >= _DRAG_THRESHOLD_PX:
                ev = self._event_for_seq(self._drag_seq)
                if ev is not None:
                    self._dragging = True
                    self.marker_drag_started.emit(dict(ev))
            if self._dragging:
                ev = self._event_for_seq(self._drag_seq)
                if ev is None:
                    return
                ts = self._ts_for_x(x)
                ts = snap_ts(
                    ts,
                    event_type=ev.get("event_type", ""),
                    modifiers=event.modifiers(),
                    tl_ts=self._tl_ts,
                    rc_ts=self._rc_ts,
                    playhead_ts=self._playhead_ts,
                    clamp_min=self._t0,
                    clamp_max=self._t1,
                )
                ev["ts_wall"] = ts
                self.marker_dragging.emit(self._drag_seq, ts)
                ms_delta = int((ts - self._drag_orig_ts) * 1000)
                local_s = ts - self._t0
                m, s = divmod(int(local_s), 60)
                ms = int((local_s % 1) * 1000)
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"{m}:{s:02d}.{ms:03d}  Δ {ms_delta:+d} ms",
                    self,
                )
                self.update()
                return

        ev = self._event_at(x)
        if ev:
            if ev.get("event_type") in _EDITABLE_TYPES:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            gate = ev.get("gate_id", -1)
            lap  = ev.get("lap_num", -1)
            gate_str = f"  gate={gate}" if gate != -1 else ""
            lap_str  = f"  lap={lap}"  if lap  != -1 else ""
            ts_s = ev["ts_wall"]
            total_s = ts_s - self._t0
            m, s = divmod(int(total_s), 60)
            ms = int((total_s % 1) * 1000)
            time_str = f"{m}:{s:02d}.{ms:03d}"
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"<b>{ev['event_type']}</b>  {time_str}{gate_str}{lap_str}<hr>{_LEGEND_HTML}",
                self,
            )
        else:
            self.unsetCursor()
            QToolTip.showText(
                event.globalPosition().toPoint(), _LEGEND_HTML, self,
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging and self._drag_seq is not None:
                ev = self._event_for_seq(self._drag_seq)
                if ev is not None:
                    self.marker_drag_ended.emit(
                        int(self._drag_seq), float(ev["ts_wall"]),
                    )
            self._drag_seq = None
            self._dragging = False
            QToolTip.hideText()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._dragging:
            seq = int(self._drag_seq or -1)
            ev = self._event_for_seq(seq)
            if ev is not None:
                ev["ts_wall"] = self._drag_orig_ts
            self._drag_seq = None
            self._dragging = False
            self.marker_drag_cancelled.emit(seq)
            self.update()
            return
        super().keyPressEvent(event)

    def _event_for_seq(self, seq: int) -> dict | None:
        for ev in self._events:
            if ev.get("seq") == seq:
                return ev
        return None

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(self.rect(), QColor("#1a1a1a"))

        p.setPen(QPen(QColor("#333333"), 1))
        cy = self.height() // 2
        p.drawLine(0, cy, self.width(), cy)

        if self._playhead_ts is not None:
            ph_x = int(self._x(self._playhead_ts))
            p.setPen(QPen(QColor("#ffffff60"), 1))
            p.drawLine(ph_x, 0, ph_x, self.height())

        for ev in self._events:
            x = self._x(ev["ts_wall"])
            etype = ev.get("event_type", "")
            selected = ev.get("seq") == self._selected_seq
            col = _color(etype)

            if selected:
                p.setPen(QPen(QColor("white"), 2))
            else:
                p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))

            self._draw_shape(p, etype, x, cy, selected)

    def _draw_shape(self, p: QPainter, etype: str, cx: float, cy: float, selected: bool) -> None:
        s = _S

        if etype == "session_start":
            path = QPainterPath()
            path.moveTo(cx,          cy - s)
            path.lineTo(cx + s * 0.8, cy + s * 0.6)
            path.lineTo(cx - s * 0.8, cy + s * 0.6)
            path.closeSubpath()
            p.drawPath(path)

        elif etype == "session_stop":
            path = QPainterPath()
            path.moveTo(cx,          cy + s)
            path.lineTo(cx + s * 0.8, cy - s * 0.6)
            path.lineTo(cx - s * 0.8, cy - s * 0.6)
            path.closeSubpath()
            p.drawPath(path)

        elif etype == "button_lap":
            p.drawEllipse(QPointF(cx, cy), s * 0.75, s * 0.75)

        elif etype in ("button_gate", "rh_gate"):
            path = QPainterPath()
            path.moveTo(cx,          cy - s)
            path.lineTo(cx + s * 0.7, cy)
            path.lineTo(cx,          cy + s)
            path.lineTo(cx - s * 0.7, cy)
            path.closeSubpath()
            if etype == "rh_gate":
                col = _color(etype)
                p.setPen(QPen(col, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        elif etype == "rh_lap":
            path = QPainterPath()
            path.moveTo(cx,              cy - s * 1.15)
            path.lineTo(cx + s * 1.0,   cy + s * 0.7)
            path.lineTo(cx - s * 1.0,   cy + s * 0.7)
            path.closeSubpath()
            p.drawPath(path)

        else:
            p.drawRect(int(cx - s * 0.5), int(cy - s * 0.5), s, s)

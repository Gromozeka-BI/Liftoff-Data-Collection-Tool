"""Thin timeline strip showing coloured event markers for the Replay editor."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt6.QtWidgets import QWidget, QToolTip

_HEIGHT = 20
_HIT_PX = 8   # click detection radius in pixels
_S = 7        # marker half-size in pixels

# event_type → RGB hex colour
_COLORS: dict[str, str] = {
    "session_start": "#00cc44",
    "session_stop":  "#dd2200",
    "button_lap":    "#3399ff",
    "button_gate":   "#ffcc00",
    "rh_lap":        "#ff8800",
    "rh_gate":       "#cc44ff",
}
_DEFAULT_COLOR = "#888888"


def _color(event_type: str) -> QColor:
    return QColor(_COLORS.get(event_type, _DEFAULT_COLOR))


class EventStrip(QWidget):
    """Renders event markers; emits signals on click/select.

    Signals:
        marker_clicked(dict)   – user clicked on a marker (also seeks to it)
        marker_selected(object) – selected event dict, or None when deselected
    """
    marker_clicked  = pyqtSignal(dict)
    marker_selected = pyqtSignal(object)   # dict | None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT)
        self.setMouseTracking(True)
        self._events:  list[dict] = []
        self._t0 = 0.0
        self._t1 = 1.0
        self._playhead_ts: float | None = None
        self._selected_seq: int | None  = None

    # ── public API ─────────────────────────────────────────────────────────

    def set_events(self, events: list[dict], t0: float, t1: float) -> None:
        self._events = list(events)
        self._t0 = t0
        self._t1 = max(t1, t0 + 0.001)
        self.update()

    def set_playhead(self, ts_wall: float) -> None:
        self._playhead_ts = ts_wall
        self.update()

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

    def _event_at(self, px: int) -> dict | None:
        best: dict | None = None
        best_d = _HIT_PX
        for ev in self._events:
            d = abs(self._x(ev["ts_wall"]) - px)
            if d < best_d:
                best_d = d
                best = ev
        return best

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            ev = self._event_at(int(event.position().x()))
            self._selected_seq = ev.get("seq") if ev else None
            self.update()
            if ev:
                self.marker_clicked.emit(ev)
                self.marker_selected.emit(ev)
            else:
                self.marker_selected.emit(None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        ev = self._event_at(int(event.position().x()))
        if ev:
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
                f"{ev['event_type']}  {time_str}{gate_str}{lap_str}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor("#1a1a1a"))

        # Subtle center line
        p.setPen(QPen(QColor("#333333"), 1))
        cy = self.height() // 2
        p.drawLine(0, cy, self.width(), cy)

        # Playhead
        if self._playhead_ts is not None:
            ph_x = int(self._x(self._playhead_ts))
            p.setPen(QPen(QColor("#ffffff60"), 1))
            p.drawLine(ph_x, 0, ph_x, self.height())

        # Markers
        for ev in self._events:
            x = self._x(ev["ts_wall"])
            etype = ev.get("event_type", "")
            selected = ev.get("seq") == self._selected_seq
            col = _color(etype)

            if selected:
                # White outline ring
                p.setPen(QPen(QColor("white"), 2))
            else:
                p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))

            self._draw_shape(p, etype, x, cy, selected)

    def _draw_shape(self, p: QPainter, etype: str, cx: float, cy: float, selected: bool) -> None:
        s = _S

        if etype == "session_start":
            # Upward triangle ▲
            path = QPainterPath()
            path.moveTo(cx,          cy - s)
            path.lineTo(cx + s * 0.8, cy + s * 0.6)
            path.lineTo(cx - s * 0.8, cy + s * 0.6)
            path.closeSubpath()
            p.drawPath(path)

        elif etype == "session_stop":
            # Downward triangle ▼
            path = QPainterPath()
            path.moveTo(cx,          cy + s)
            path.lineTo(cx + s * 0.8, cy - s * 0.6)
            path.lineTo(cx - s * 0.8, cy - s * 0.6)
            path.closeSubpath()
            p.drawPath(path)

        elif etype == "button_lap":
            # Circle ●
            p.drawEllipse(QPointF(cx, cy), s * 0.75, s * 0.75)

        elif etype in ("button_gate", "rh_gate"):
            # Diamond ◆  (filled for button_gate, outlined for rh_gate)
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
            # Larger filled triangle (orange S/F)
            path = QPainterPath()
            path.moveTo(cx,              cy - s * 1.15)
            path.lineTo(cx + s * 1.0,   cy + s * 0.7)
            path.lineTo(cx - s * 1.0,   cy + s * 0.7)
            path.closeSubpath()
            p.drawPath(path)

        else:
            # Generic square
            p.drawRect(int(cx - s * 0.5), int(cy - s * 0.5), s, s)

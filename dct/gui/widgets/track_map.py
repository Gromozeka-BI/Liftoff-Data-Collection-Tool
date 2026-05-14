"""2-D top-down track map with optional HUD overlay.

Форма ворот — буква Н:
  - левая / правая стойки — вертикальные линии на ±hw
  - перекладина — горизонтальная линия по центру
  - двухэтажные + два внутренних штыря у перекладины

Нумерация:
  - бейдж показывает реальный id ворот из track.json
  - бейдж нижнего/одноэтажного этажа — красный (pink fill)
  - бейдж верхнего этажа — синий (blue fill)
  - позиция: снизу-слева от входа (левая стойка, входная сторона)
  - если оба пролёта двухэтажных ворот с одной стороны — бейджи стекируются
  - бейджи масштабируются вместе с картой (только квадраты, без текста)
  - направление пролёта берётся из rotation[1] в JSON
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QTransform,
)
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent

from dct.gui import theme

_CLR_GATE    = "#0000CC"
_CLR_GATE_SF = "#CC0000"
_CLR_FLAG    = _CLR_GATE          # флаги — цвет ворот первого этажа

# Бейдж: (fill, border)
_BADGE_LOW    = ("#f8cecc", "#b85450")   # нижний этаж / одноэтажный
_BADGE_HIGH   = ("#dae8fc", "#6c8ebf")   # верхний этаж
_FILL_OFFSEQ  = "#949494"   # фон бейджа для элементов вне gate_sequence

_FLAG_ARM   = 0.35   # длина короткого луча флага (м)
_FLAG_TOP   = _FLAG_ARM * 2.0   # длина верхнего луча (вдвое длиннее)


def _is_flag(gate: dict) -> bool:
    """Флаг — элемент без поля size (нет размеров пролёта)."""
    return "size" not in gate


class BadgeItem(pg.GraphicsObject):
    """
    Бейдж с номером ворот.
    Квадрат и текст внутри масштабируются вместе с картой:
    размер шрифта вычисляется из текущего worldTransform прямо в paint().
    """

    def __init__(self, num: str, fill: str, border: str, size_meters: float = 0.4):
        super().__init__()
        self.num          = str(num)
        self.size_meters  = size_meters
        self._half        = size_meters / 2
        self._rect        = QRectF(-self._half, -self._half, size_meters, size_meters)

        self._path = QPainterPath()
        self._path.addRoundedRect(self._rect, size_meters * 0.2, size_meters * 0.2)

        self._brush = QBrush(QColor(fill))
        self._pen   = QPen(QColor(border))
        self._pen.setWidthF(size_meters * 0.06)   # толщина рамки масштабируется

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args):
        # Рисуем скруглённый прямоугольник в координатах данных (масштабируется)
        p.setBrush(self._brush)
        p.setPen(self._pen)
        p.drawPath(self._path)

        # Получаем трансформацию local → viewport ДО сброса
        t  = p.transform()
        tl = t.map(QPointF(-self._half, -self._half))
        br = t.map(QPointF( self._half,  self._half))

        screen_w   = abs(br.x() - tl.x())
        screen_h   = abs(br.y() - tl.y())
        pixel_size = min(screen_w, screen_h)
        if pixel_size < 6:
            return   # слишком мелко — текст не рисуем

        cx = (tl.x() + br.x()) / 2
        cy = (tl.y() + br.y()) / 2

        # Сбрасываем трансформацию — рисуем текст в экранных пикселях
        p.save()
        p.setWorldTransform(QTransform())
        font = QFont("Arial")
        font.setBold(True)
        font.setPixelSize(max(6, int(pixel_size * 0.60)))
        p.setFont(font)
        p.setPen(QColor(0, 0, 0))
        screen_rect = QRectF(cx - screen_w / 2, cy - screen_h / 2, screen_w, screen_h)
        p.drawText(screen_rect, Qt.AlignmentFlag.AlignCenter, self.num)
        p.restore()


class WorldTextItem(pg.GraphicsObject):
    """Текст в мировых координатах без фона, масштабируется как BadgeItem.
    font_scale — множитель относительно размера бейджа (0.60 * pixel_size)."""

    def __init__(self, text: str, color: str, size_meters: float = 0.4,
                 font_scale: float = 0.60):
        super().__init__()
        self.text        = text
        self._color      = QColor(color)
        self._half       = size_meters / 2
        self._rect       = QRectF(-self._half, -self._half, size_meters, size_meters)
        self._font_scale = font_scale

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args):
        t  = p.transform()
        tl = t.map(QPointF(-self._half, -self._half))
        br = t.map(QPointF( self._half,  self._half))

        screen_w   = abs(br.x() - tl.x())
        screen_h   = abs(br.y() - tl.y())
        pixel_size = min(screen_w, screen_h)
        if pixel_size < 4:
            return

        cx = (tl.x() + br.x()) / 2
        cy = (tl.y() + br.y()) / 2

        p.save()
        p.setWorldTransform(QTransform())
        font = QFont("Arial")
        font.setBold(True)
        font.setPixelSize(max(4, int(pixel_size * self._font_scale)))
        p.setFont(font)
        p.setPen(self._color)
        screen_rect = QRectF(cx - screen_w / 2, cy - screen_h / 2, screen_w, screen_h)
        p.drawText(screen_rect, Qt.AlignmentFlag.AlignCenter, self.text)
        p.restore()


_LOC_LABELS: list[str] = ["GT", "LF", "RC", "Legacy", "KF", "CamKF"]
_LOC_COLORS: dict[str, str] = {
    "GT":     theme.DRONE,
    "LF":     theme.LOCALIZER,
    "RC":     theme.LOCALIZER_RC,
    "Legacy": theme.LOCALIZER_LEGACY,
    "KF":     theme.LOCALIZER_KF,
    "CamKF":  theme.LOCALIZER_CAM,
}


class MapHUDItem(QGraphicsObject):
    """Translucent HUD pinned to the top-right corner of the map's viewport.

    Renders short telemetry lines (Spd / Alt / Bat / Pos) and per-localizer
    rows (LF / RC / KF) with inline visibility checkboxes.
    Clicking a localizer row toggles that marker on the map.
    """

    marker_visibility_changed = pyqtSignal(str, bool)

    PADDING  = 10
    LINE_H   = 22
    HEAD_H   = 18
    BG_ALPHA = 0.65
    CHK_SIZE = 10   # checkbox square side, px

    def __init__(self, plot_widget: "TrackMapWidget") -> None:
        super().__init__()
        self._pw = plot_widget
        self.setZValue(50)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self._tele_lines: list[tuple[str, str, str]] = []
        self._loc_data: dict[str, tuple[float, float]] = {}   # label → (prog, sigma)
        self._has_gt: bool = False
        self._marker_visible: dict[str, bool] = {
            "GT": True, "LF": True, "RC": True, "Legacy": True, "KF": True, "CamKF": False,
        }
        self._race_mode: bool = False
        self._size_w = 220
        self._size_h = 120
        # click regions in local (screen-space) coords: (y_top, y_bottom, label)
        self._click_rows: list[tuple[float, float, str]] = []

    # ── public API ────────────────────────────────────────────────────────

    def set_race_mode(self, on: bool) -> None:
        if on != self._race_mode:
            self._race_mode = on
            self.prepareGeometryChange()
            self.update()

    def set_data(
        self,
        frame: dict | None,
        locs: dict[str, tuple[float, float]] | None,
        has_gt: bool = False,
    ) -> None:
        """Update HUD content.

        Parameters
        ----------
        frame  : telemetry dict (keys vel_x/y/z, pos_y, bat_v, pos_x/z) or None
        locs   : dict of active localizer results, e.g.
                 {"LF": (progress, sigma_m), "RC": (...), "KF": (...), "Legacy": (...)}
                 Only labels present in the dict are displayed.
        has_gt : True when ground-truth drone telemetry is available (shows GT row).
        """
        tele: list[tuple[str, str, str]] = []
        if frame:
            spd = (
                frame.get("vel_x", 0.0) ** 2
                + frame.get("vel_y", 0.0) ** 2
                + frame.get("vel_z", 0.0) ** 2
            ) ** 0.5
            tele.append(("Spd", f"{spd:.1f} m/s",  theme.TEXT))
            tele.append(("Alt", f"{frame.get('pos_y', 0.0):.1f} m", theme.TEXT))
            tele.append(("Bat", f"{frame.get('bat_v', 0.0):.1f} V", theme.TEXT))
            tele.append((
                "Pos",
                f"X{frame.get('pos_x', 0.0):.1f}  Z{frame.get('pos_z', 0.0):.1f}",
                theme.DIM,
            ))
        self._tele_lines = tele
        self._loc_data   = dict(locs) if locs else {}
        self._has_gt     = bool(has_gt)
        self.prepareGeometryChange()
        self.update()

    def set_marker_visible(self, marker: str, on: bool) -> None:
        self._marker_visible[marker] = bool(on)
        self.update()

    # ── QGraphicsItem ─────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        n_tele = len(self._tele_lines)
        n_gt   = 1 if self._has_gt else 0
        n_loc  = sum(
            1
            for k in _LOC_LABELS
            if k != "GT" and (k in self._loc_data or k == "CamKF")
        )
        rows   = max(1, n_tele + n_gt + n_loc)
        line_h = self.LINE_H + (4 if self._race_mode else 0)
        h = self.PADDING * 2 + self.HEAD_H + rows * line_h
        w = 280 if self._race_mode else 240
        self._size_w, self._size_h = w, h
        return QRectF(-w, 0, w, h)

    def paint(self, p: QPainter, *args) -> None:
        if not self._tele_lines and not self._loc_data:
            return
        try:
            rect = self.boundingRect()
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            bg = QColor(theme.PANEL)
            bg.setAlphaF(self.BG_ALPHA)
            p.setBrush(QBrush(bg))
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawRoundedRect(rect, 6, 6)

            # ── header ────────────────────────────────────────────────────
            font_head = QFont("Segoe UI")
            font_head.setBold(True)
            font_head.setPixelSize(
                theme.FONT_HEAD if not self._race_mode else theme.FONT_HUD_RACE - 4,
            )
            p.setFont(font_head)
            p.setPen(QColor(theme.DIM))
            p.drawText(
                QRectF(rect.x() + self.PADDING, rect.y() + self.PADDING - 2,
                       rect.width(), self.HEAD_H),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                "DCT HUD",
            )

            font = QFont("Segoe UI")
            font.setPixelSize(theme.FONT_HUD_RACE if self._race_mode else theme.FONT_HUD)
            font.setBold(True)
            p.setFont(font)

            line_h = self.LINE_H + (4 if self._race_mode else 0)
            lbl_w  = 56   # fixed label column width

            # ── telemetry rows (no checkbox) ──────────────────────────────
            for i, (label, value, color) in enumerate(self._tele_lines):
                y = rect.y() + self.PADDING + self.HEAD_H + i * line_h
                p.setPen(QColor(theme.DIM))
                p.drawText(
                    QRectF(rect.x() + self.PADDING, y, lbl_w, line_h),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    label,
                )
                p.setPen(QColor(color))
                p.drawText(
                    QRectF(rect.x() + self.PADDING + lbl_w, y,
                           rect.width() - self.PADDING * 2 - lbl_w, line_h),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    value,
                )

            # ── marker rows (GT + localizers) with checkboxes ─────────────
            click_rows: list[tuple[float, float, str]] = []
            base_i  = len(self._tele_lines)
            row_idx = 0
            chk     = self.CHK_SIZE

            def _draw_marker_row(lbl: str, value_str: str) -> None:
                nonlocal row_idx
                i       = base_i + row_idx
                row_idx += 1
                y       = rect.y() + self.PADDING + self.HEAD_H + i * line_h
                color   = _LOC_COLORS.get(lbl, theme.TEXT)
                visible = self._marker_visible.get(lbl, True)

                chk_x    = rect.x() + self.PADDING
                chk_y    = y + (line_h - chk) / 2
                chk_rect = QRectF(chk_x, chk_y, chk, chk)
                p.setPen(QPen(QColor(color), 1.5))
                p.setBrush(QBrush(QColor(color) if visible else QColor(theme.PANEL)))
                p.drawRoundedRect(chk_rect, 2, 2)
                p.setBrush(Qt.BrushStyle.NoBrush)

                label_x = rect.x() + self.PADDING + chk + 4
                p.setPen(QColor(color if visible else theme.DIM))
                p.drawText(
                    QRectF(label_x, y, lbl_w, line_h),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    lbl,
                )
                if visible and value_str:
                    p.setPen(QColor(color))
                    p.drawText(
                        QRectF(label_x + lbl_w, y,
                               rect.width() - self.PADDING * 2 - chk - 4 - lbl_w, line_h),
                        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                        value_str,
                    )
                click_rows.append((y, y + line_h, lbl))

            # GT row (no progress/sigma — just the checkbox)
            if self._has_gt:
                _draw_marker_row("GT", "")

            # Localizer rows
            for lbl in _LOC_LABELS:
                if lbl == "GT":
                    continue
                if lbl not in self._loc_data and lbl != "CamKF":
                    continue
                if lbl in self._loc_data:
                    prog, sigma = self._loc_data[lbl]
                    value = f"{prog * 100:.0f}%  σ±{sigma:.1f} m"
                else:
                    value = "overlay"
                _draw_marker_row(lbl, value)

            self._click_rows = click_rows

        except Exception:  # noqa: BLE001 — never let a paint error crash Qt's loop
            return

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        py = event.pos().y()
        for (y0, y1, lbl) in self._click_rows:
            if y0 <= py < y1:
                new_state = not self._marker_visible.get(lbl, True)
                self._marker_visible[lbl] = new_state
                self.marker_visibility_changed.emit(lbl, new_state)
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    # ── positioning helper ────────────────────────────────────────────────

    def reposition(self) -> None:
        viewport = self._pw.viewport()
        if viewport is None:
            return
        x = viewport.width() - self.PADDING
        y = self.PADDING
        scene_pos = self._pw.mapToScene(int(x), int(y))
        self.setPos(scene_pos)


class TrackMapWidget(pg.PlotWidget):
    marker_visibility_changed = pyqtSignal(str, bool)

    TRAIL_SECS = 3.0
    TRAIL_MAX  = 400
    LOC_TRAIL_MAX = 120

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_plot()
        self._gate_items:  list = []
        self._bounds_item        = None
        self._start_items: list  = []

        self._trail_x:  deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_z:  deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_ts: deque[float] = deque(maxlen=self.TRAIL_MAX)
        self._trail_item = self.plot([], [], pen=pg.mkPen(theme.TRAIL, width=2))

        self._arrow = pg.ArrowItem(
            angle=90, tipAngle=35, headLen=14, tailLen=10,
            tailWidth=4, brush=pg.mkBrush(theme.DRONE), pen=None,
        )
        self.addItem(self._arrow)
        self._arrow.setPos(0, 0)

        # Эталонный круг (пунктир) и оценка локализатора
        self._ref_path_item = None
        self._loc_trail_x: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc_trail_z: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc_trail_item = self.plot(
            [], [], pen=pg.mkPen(theme.LOC_TRAIL, width=1.5),
        )
        self._loc_trail_item.setZValue(5)
        self._loc_arrow = pg.ArrowItem(
            angle=90, tipAngle=32, headLen=12, tailLen=8,
            tailWidth=3, brush=pg.mkBrush(theme.LOCALIZER), pen=None,
        )
        self._loc_arrow.setZValue(6)
        self._loc_arrow.setOpacity(0.0)
        self.addItem(self._loc_arrow)

        # Второй локализатор (RC) в режиме Liftoff+RC — отдельный трейл/стрелка
        self._loc2_trail_x: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc2_trail_z: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc2_trail_item = self.plot(
            [], [], pen=pg.mkPen(theme.LOC_TRAIL_RC, width=1.5, style=Qt.PenStyle.DotLine),
        )
        self._loc2_trail_item.setZValue(5)
        self._loc2_arrow = pg.ArrowItem(
            angle=90, tipAngle=32, headLen=12, tailLen=8,
            tailWidth=3, brush=pg.mkBrush(theme.LOCALIZER_RC), pen=None,
        )
        self._loc2_arrow.setZValue(6)
        self._loc2_arrow.setOpacity(0.0)
        self.addItem(self._loc2_arrow)

        # Третий локализатор (legacy sticks / тот же круг)
        self._loc3_trail_x: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc3_trail_z: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc3_trail_item = self.plot(
            [], [], pen=pg.mkPen(theme.LOC_TRAIL_LEGACY, width=1.2, style=Qt.PenStyle.DashDotLine),
        )
        self._loc3_trail_item.setZValue(5)
        self._loc3_arrow = pg.ArrowItem(
            angle=90, tipAngle=30, headLen=11, tailLen=7,
            tailWidth=3, brush=pg.mkBrush(theme.LOCALIZER_LEGACY), pen=None,
        )
        self._loc3_arrow.setZValue(6)
        self._loc3_arrow.setOpacity(0.0)
        self.addItem(self._loc3_arrow)

        # Четвёртый локализатор (RC → KF Layer 2) — красный, сплошная линия
        self._loc4_trail_x: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc4_trail_z: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc4_trail_item = self.plot(
            [], [], pen=pg.mkPen(theme.LOC_TRAIL_KF, width=2.0),
        )
        self._loc4_trail_item.setZValue(7)
        self._loc4_arrow = pg.ArrowItem(
            angle=90, tipAngle=35, headLen=14, tailLen=10,
            tailWidth=4, brush=pg.mkBrush(theme.LOCALIZER_KF), pen=None,
        )
        self._loc4_arrow.setZValue(8)
        self._loc4_arrow.setOpacity(0.0)
        self.addItem(self._loc4_arrow)

        # Пятый локализатор: camera inject + KF Layer 2 — фиолетовый
        self._loc5_trail_x: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc5_trail_z: deque[float] = deque(maxlen=self.LOC_TRAIL_MAX)
        self._loc5_trail_item = self.plot(
            [], [], pen=pg.mkPen(theme.LOC_TRAIL_CAM, width=2.0),
        )
        self._loc5_trail_item.setZValue(9)
        self._loc5_arrow = pg.ArrowItem(
            angle=90, tipAngle=35, headLen=14, tailLen=10,
            tailWidth=4, brush=pg.mkBrush(theme.LOCALIZER_CAM), pen=None,
        )
        self._loc5_arrow.setZValue(10)
        self._loc5_arrow.setOpacity(0.0)
        self.addItem(self._loc5_arrow)

        self._has_track = False

        self._show_ref_path = True
        self._show_loc_arrow = True
        self._show_loc_trail = True
        # per-marker visibility (controlled via HUD checkboxes)
        self._marker_visible: dict[str, bool] = {
            "GT": True, "LF": True, "RC": True, "Legacy": True, "KF": True, "CamKF": False,
        }

        self._hud = MapHUDItem(self)
        self.scene().addItem(self._hud)
        self._hud.reposition()
        self._hud.set_data(None, None)
        self._hud.marker_visibility_changed.connect(self._on_hud_marker_toggle)

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup_plot(self) -> None:
        self.setBackground(theme.PANEL)
        pi = self.getPlotItem()
        for axis in ("bottom", "left"):
            pi.getAxis(axis).setPen(pg.mkPen(theme.BORDER))
            pi.getAxis(axis).setTextPen(pg.mkPen(theme.DIM))
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setLabel("bottom", "X (m)", color=theme.DIM)
        pi.setLabel("left",   "Z (m)", color=theme.DIM)
        self.setAspectLocked(True)

    # ── public API ─────────────────────────────────────────────────────────

    def clear_reference_path(self) -> None:
        if self._ref_path_item is not None:
            self.removeItem(self._ref_path_item)
            self._ref_path_item = None

    def set_reference_path(self, xs, zs) -> None:
        """Нарисовать эталонную траекторию круга (X, Z в метрах), пунктир."""
        self.clear_reference_path()
        xa = list(xs)
        za = list(zs)
        if len(xa) < 2:
            return
        pen = pg.mkPen(theme.DIM, width=2, style=Qt.PenStyle.DashLine)
        self._ref_path_item = self.plot(xa, za, pen=pen)
        self._ref_path_item.setZValue(-2)
        self._ref_path_item.setVisible(self._show_ref_path)

    def clear_localizer_overlay(self) -> None:
        self._loc_trail_x.clear()
        self._loc_trail_z.clear()
        self._loc_trail_item.setData([], [])
        self._loc_arrow.setOpacity(0.0)
        self._loc2_trail_x.clear()
        self._loc2_trail_z.clear()
        self._loc2_trail_item.setData([], [])
        self._loc2_arrow.setOpacity(0.0)
        self._loc3_trail_x.clear()
        self._loc3_trail_z.clear()
        self._loc3_trail_item.setData([], [])
        self._loc3_arrow.setOpacity(0.0)
        self._loc4_trail_x.clear()
        self._loc4_trail_z.clear()
        self._loc4_trail_item.setData([], [])
        self._loc4_arrow.setOpacity(0.0)
        self._loc5_trail_x.clear()
        self._loc5_trail_z.clear()
        self._loc5_trail_item.setData([], [])
        self._loc5_arrow.setOpacity(0.0)

    def update_localizer_estimate(self, px: float, pz: float) -> None:
        """Оценка локализатора по стикам Liftoff (золотой трейл/стрелка)."""
        self._loc_trail_x.append(px)
        self._loc_trail_z.append(pz)
        vis = self._marker_visible.get("LF", True)
        if self._show_loc_trail and vis:
            self._loc_trail_item.setData(list(self._loc_trail_x), list(self._loc_trail_z))
        else:
            self._loc_trail_item.setData([], [])
        self._loc_arrow.setPos(px, pz)
        self._loc_arrow.setOpacity(1.0 if (self._show_loc_arrow and vis) else 0.0)
        if len(self._loc_trail_x) >= 2:
            dx = self._loc_trail_x[-1] - self._loc_trail_x[-2]
            dz = self._loc_trail_z[-1] - self._loc_trail_z[-2]
            if dx * dx + dz * dz > 1e-8:
                tang_deg = math.degrees(math.atan2(dz, dx))
                self._loc_arrow.setStyle(angle=90 - tang_deg)

    def update_localizer_rc_estimate(self, px: float, pz: float) -> None:
        """Оценка второго локализатора по каналам RC (синий трейл/стрелка)."""
        self._loc2_trail_x.append(px)
        self._loc2_trail_z.append(pz)
        vis = self._marker_visible.get("RC", True)
        if self._show_loc_trail and vis:
            self._loc2_trail_item.setData(list(self._loc2_trail_x), list(self._loc2_trail_z))
        else:
            self._loc2_trail_item.setData([], [])
        self._loc2_arrow.setPos(px, pz)
        self._loc2_arrow.setOpacity(1.0 if (self._show_loc_arrow and vis) else 0.0)
        if len(self._loc2_trail_x) >= 2:
            dx = self._loc2_trail_x[-1] - self._loc2_trail_x[-2]
            dz = self._loc2_trail_z[-1] - self._loc2_trail_z[-2]
            if dx * dx + dz * dz > 1e-8:
                tang_deg = math.degrees(math.atan2(dz, dx))
                self._loc2_arrow.setStyle(angle=90 - tang_deg)

    def update_localizer_legacy_estimate(self, px: float, pz: float) -> None:
        """Третья оценка PF: legacy-наблюдения (сырые стики), зелёный dash-dot."""
        self._loc3_trail_x.append(px)
        self._loc3_trail_z.append(pz)
        vis = self._marker_visible.get("Legacy", True)
        if self._show_loc_trail and vis:
            self._loc3_trail_item.setData(list(self._loc3_trail_x), list(self._loc3_trail_z))
        else:
            self._loc3_trail_item.setData([], [])
        self._loc3_arrow.setPos(px, pz)
        self._loc3_arrow.setOpacity(1.0 if (self._show_loc_arrow and vis) else 0.0)
        if len(self._loc3_trail_x) >= 2:
            dx = self._loc3_trail_x[-1] - self._loc3_trail_x[-2]
            dz = self._loc3_trail_z[-1] - self._loc3_trail_z[-2]
            if dx * dx + dz * dz > 1e-8:
                tang_deg = math.degrees(math.atan2(dz, dx))
                self._loc3_arrow.setStyle(angle=90 - tang_deg)

    def update_localizer_kf_estimate(self, px: float, pz: float) -> None:
        """Оценка KF второго контура (RC → KF Layer 2), красный."""
        self._loc4_trail_x.append(px)
        self._loc4_trail_z.append(pz)
        visible = self._marker_visible.get("KF", True)
        if self._show_loc_trail and visible:
            self._loc4_trail_item.setData(list(self._loc4_trail_x), list(self._loc4_trail_z))
        else:
            self._loc4_trail_item.setData([], [])
        self._loc4_arrow.setPos(px, pz)
        self._loc4_arrow.setOpacity(1.0 if (self._show_loc_arrow and visible) else 0.0)
        if len(self._loc4_trail_x) >= 2:
            dx = self._loc4_trail_x[-1] - self._loc4_trail_x[-2]
            dz = self._loc4_trail_z[-1] - self._loc4_trail_z[-2]
            if dx * dx + dz * dz > 1e-8:
                tang_deg = math.degrees(math.atan2(dz, dx))
                self._loc4_arrow.setStyle(angle=90 - tang_deg)

    def update_localizer_cam_estimate(self, px: float, pz: float) -> None:
        """Оценка camera-fused контура (PF + camera inject + KF), фиолетовый."""
        self._loc5_trail_x.append(px)
        self._loc5_trail_z.append(pz)
        visible = self._marker_visible.get("CamKF", False)
        if self._show_loc_trail and visible:
            self._loc5_trail_item.setData(list(self._loc5_trail_x), list(self._loc5_trail_z))
        else:
            self._loc5_trail_item.setData([], [])
        self._loc5_arrow.setPos(px, pz)
        self._loc5_arrow.setOpacity(1.0 if (self._show_loc_arrow and visible) else 0.0)
        if len(self._loc5_trail_x) >= 2:
            dx = self._loc5_trail_x[-1] - self._loc5_trail_x[-2]
            dz = self._loc5_trail_z[-1] - self._loc5_trail_z[-2]
            if dx * dx + dz * dz > 1e-8:
                tang_deg = math.degrees(math.atan2(dz, dx))
                self._loc5_arrow.setStyle(angle=90 - tang_deg)

    # ── visibility toggles ────────────────────────────────────────────────

    def set_reference_path_visible(self, on: bool) -> None:
        self._show_ref_path = bool(on)
        if self._ref_path_item is not None:
            self._ref_path_item.setVisible(self._show_ref_path)

    def set_localizer_arrow_visible(self, on: bool) -> None:
        self._show_loc_arrow = bool(on)
        mv = self._marker_visible
        self._loc_arrow.setOpacity(
            1.0 if (on and mv.get("LF", True) and len(self._loc_trail_x) > 0) else 0.0)
        self._loc2_arrow.setOpacity(
            1.0 if (on and mv.get("RC", True) and len(self._loc2_trail_x) > 0) else 0.0)
        self._loc3_arrow.setOpacity(
            1.0 if (on and mv.get("Legacy", True) and len(self._loc3_trail_x) > 0) else 0.0)
        self._loc4_arrow.setOpacity(
            1.0 if (on and mv.get("KF", True) and len(self._loc4_trail_x) > 0) else 0.0)
        self._loc5_arrow.setOpacity(
            1.0 if (on and mv.get("CamKF", False) and len(self._loc5_trail_x) > 0) else 0.0)

    def set_localizer_trail_visible(self, on: bool) -> None:
        self._show_loc_trail = bool(on)
        mv = self._marker_visible
        if not on:
            self._loc_trail_item.setData([], [])
            self._loc2_trail_item.setData([], [])
            self._loc3_trail_item.setData([], [])
            self._loc4_trail_item.setData([], [])
            self._loc5_trail_item.setData([], [])
        else:
            if mv.get("LF", True):
                self._loc_trail_item.setData(list(self._loc_trail_x), list(self._loc_trail_z))
            if mv.get("RC", True):
                self._loc2_trail_item.setData(list(self._loc2_trail_x), list(self._loc2_trail_z))
            if mv.get("Legacy", True):
                self._loc3_trail_item.setData(list(self._loc3_trail_x), list(self._loc3_trail_z))
            if mv.get("KF", True):
                self._loc4_trail_item.setData(list(self._loc4_trail_x), list(self._loc4_trail_z))
            if mv.get("CamKF", False):
                self._loc5_trail_item.setData(list(self._loc5_trail_x), list(self._loc5_trail_z))

    def set_marker_visible(self, marker: str, on: bool) -> None:
        """Показать/скрыть конкретный маркер (GT / LF / RC / Legacy / KF)."""
        self._marker_visible[marker] = bool(on)
        self._hud.set_marker_visible(marker, on)
        show       = bool(on) and self._show_loc_arrow
        trail_show = bool(on) and self._show_loc_trail
        if marker == "GT":
            self._arrow.setOpacity(1.0 if (bool(on) and len(self._trail_x) > 0) else 0.0)
            self._trail_item.setData(
                (list(self._trail_x) if bool(on) else []),
                (list(self._trail_z) if bool(on) else []),
            )
        elif marker == "LF":
            self._loc_arrow.setOpacity(1.0 if (show and len(self._loc_trail_x) > 0) else 0.0)
            self._loc_trail_item.setData(
                (list(self._loc_trail_x) if trail_show else []),
                (list(self._loc_trail_z) if trail_show else []),
            )
        elif marker == "RC":
            self._loc2_arrow.setOpacity(1.0 if (show and len(self._loc2_trail_x) > 0) else 0.0)
            self._loc2_trail_item.setData(
                (list(self._loc2_trail_x) if trail_show else []),
                (list(self._loc2_trail_z) if trail_show else []),
            )
        elif marker == "Legacy":
            self._loc3_arrow.setOpacity(1.0 if (show and len(self._loc3_trail_x) > 0) else 0.0)
            self._loc3_trail_item.setData(
                (list(self._loc3_trail_x) if trail_show else []),
                (list(self._loc3_trail_z) if trail_show else []),
            )
        elif marker == "KF":
            self._loc4_arrow.setOpacity(1.0 if (show and len(self._loc4_trail_x) > 0) else 0.0)
            self._loc4_trail_item.setData(
                (list(self._loc4_trail_x) if trail_show else []),
                (list(self._loc4_trail_z) if trail_show else []),
            )
        elif marker == "CamKF":
            self._loc5_arrow.setOpacity(1.0 if (show and len(self._loc5_trail_x) > 0) else 0.0)
            self._loc5_trail_item.setData(
                (list(self._loc5_trail_x) if trail_show else []),
                (list(self._loc5_trail_z) if trail_show else []),
            )

    def _on_hud_marker_toggle(self, marker: str, on: bool) -> None:
        self.set_marker_visible(marker, on)
        self.marker_visibility_changed.emit(marker, on)

    # ── HUD ───────────────────────────────────────────────────────────────

    def update_hud(
        self,
        frame: dict | None,
        locs: dict[str, tuple[float, float]] | None,
        has_gt: bool = False,
    ) -> None:
        self._hud.set_data(frame, locs, has_gt=has_gt)
        self._hud.reposition()

    def set_hud_visible(self, on: bool) -> None:
        self._hud.setVisible(bool(on))

    def set_hud_race_mode(self, on: bool) -> None:
        self._hud.set_race_mode(on)
        self._hud.reposition()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        hud = getattr(self, "_hud", None)
        if hud is not None:
            hud.reposition()

    def setup_track(self, track_data: dict[str, Any]) -> None:
        self.clear_reference_path()
        self.clear_localizer_overlay()

        for item in self._gate_items:
            self.removeItem(item)
        self._gate_items.clear()
        if self._bounds_item:
            self.removeItem(self._bounds_item)
            self._bounds_item = None
        for item in self._start_items:
            self.removeItem(item)
        self._start_items.clear()

        gates    = track_data.get("gates", [])
        gate_map = {g["id"]: g for g in gates}
        seq      = track_data.get("gate_sequence", [g["id"] for g in gates])
        seq_set  = set(seq)
        sf_id    = next((g["id"] for g in gates if g.get("is_start_finish")), None)

        sp     = track_data.get("start_point")
        sp_pos = sp["position"] if sp and "position" in sp else None

        # Направление пролёта для каждого gate (берётся из rotation[1])
        fly_dirs = self._get_fly_dirs_from_rotation(gate_map)

        # Двухэтажные пары
        two_level_pairs = self._find_two_level_pairs(gates)
        two_level_low   = {lo for lo, _hi in two_level_pairs}
        two_level_high  = {hi for _lo, hi in two_level_pairs}
        two_level_all   = two_level_low | two_level_high

        # Для каждой пары: одинаковое ли направление пролёта?
        same_dir_pairs: set[int] = set()   # id ворот, у которых напарник летит туда же
        for lo, hi in two_level_pairs:
            fl_lo = fly_dirs.get(lo, (0.0, 1.0))
            fl_hi = fly_dirs.get(hi, (0.0, 1.0))
            dot = fl_lo[0]*fl_hi[0] + fl_lo[1]*fl_hi[1]
            if dot > 0.7:   # угол < ~46° → считаем «одна сторона»
                same_dir_pairs.add(lo)
                same_dir_pairs.add(hi)

        # ── Формы ворот и флагов ─────────────────────────────────────────
        for gate in gates:
            gid = gate["id"]
            if _is_flag(gate):
                self._draw_flag(gate, in_seq=gid in seq_set)
                continue
            if gid in two_level_high:
                continue   # XZ совпадает с нижним — рисуем один раз
            is_sf        = gate.get("is_start_finish", False)
            is_two_level = gid in two_level_all
            color        = _CLR_GATE_SF if is_sf else _CLR_GATE
            self._draw_gate_h(gate, color, is_two_level)

        # ── Бейджи номеров ───────────────────────────────────────────────
        # Размер бейджа в метрах (квадрат масштабируется)
        badge_size = 0.4  # 40 сантиметров
        
        # Бейджи: показываем реальный id из JSON для каждого элемента
        # Порядок: сначала элементы из seq (уже включает sf_id), затем оставшиеся
        all_ids_ordered = list(dict.fromkeys(list(seq) + [g["id"] for g in gates]))
        drawn_badges: set[int] = set()

        for gid in all_ids_ordered:
            if gid not in gate_map or gid in drawn_badges:
                continue
            gate = gate_map[gid]
            if _is_flag(gate):
                continue   # флаги подписываются своим id в _draw_flag

            drawn_badges.add(gid)
            fly_dir = fly_dirs.get(gid, (0.0, 1.0))
            in_seq  = gid in seq_set
            is_high = gid in two_level_high
            fill, border = _BADGE_HIGH if is_high else _BADGE_LOW
            if not in_seq:
                fill = _FILL_OFFSEQ
            extra = 0.5 if (gid in two_level_high and gid in same_dir_pairs) else 0.0

            self._draw_badge(gate, str(gid), fly_dir, fill, border, extra, badge_size)

        # ── Границы трассы ───────────────────────────────────────────────
        bounds = track_data.get("bounds")
        bx = bz = ox = oz = 0.0
        if bounds:
            bx = float(bounds.get("x", 0))
            bz = float(bounds.get("y", bounds.get("z", 0)))
            ox = float(bounds.get("origin_x", 0.0))
            oz = float(bounds.get("origin_z", 0.0))
            self._bounds_item = self.plot(
                [ox, ox+bx, ox+bx, ox, ox],
                [oz, oz,    oz+bz, oz+bz, oz],
                pen=pg.mkPen(theme.BORDER, width=1, style=Qt.PenStyle.DashLine),
            )

        # ── Стартовая точка ──────────────────────────────────────────────
        sp_x = sp_z = None
        if sp:
            if "position" in sp:
                sp_x, _, sp_z = sp["position"]
            else:
                sp_x = float(sp.get("x", 0))
                sp_z = float(sp.get("z", sp.get("y", 0)))

            # Круг в мировых координатах, радиус в 3 раза меньше исходного
            r       = float(sp.get("radius", 1.0)) / 3.0
            angles  = [i * 2 * math.pi / 64 for i in range(65)]
            circ = self.plot(
                [sp_x + r * math.cos(a) for a in angles],
                [sp_z + r * math.sin(a) for a in angles],
                pen=pg.mkPen("#ffffff", width=2),
            )
            self._start_items.append(circ)

            # «H» — масштабируется с картой как бейдж, шрифт ×2 от бейджа
            h_lbl = WorldTextItem("H", "#ffffff",
                                  size_meters=r * 1.2, font_scale=1.20)
            h_lbl.setPos(sp_x, sp_z)
            self.addItem(h_lbl)
            self._start_items.append(h_lbl)

        # ── Auto-fit ─────────────────────────────────────────────────────
        fit_xs: list[float] = []
        fit_zs: list[float] = []
        if bounds and bx > 0 and bz > 0:
            fit_xs = [ox, ox + bx]
            fit_zs = [oz, oz + bz]
        elif gates:
            fit_xs = [g["position"][0] for g in gates]
            fit_zs = [g["position"][2] for g in gates]
        if sp_x is not None:
            fit_xs.append(sp_x)
            fit_zs.append(sp_z)
        if fit_xs and fit_zs:
            pad = max(2.0, (max(fit_xs) - min(fit_xs)) * 0.1)
            self.setXRange(min(fit_xs) - pad, max(fit_xs) + pad, padding=0)
            self.setYRange(min(fit_zs) - pad, max(fit_zs) + pad, padding=0)
        self._has_track = True

    def update_drone(self, frame: dict[str, Any]) -> None:
        px = frame["pos_x"]
        pz = frame["pos_z"]
        ts = frame["ts_wall"]
        self._trail_x.append(px)
        self._trail_z.append(pz)
        self._trail_ts.append(ts)
        cutoff = ts - self.TRAIL_SECS
        while self._trail_ts and self._trail_ts[0] < cutoff:
            self._trail_x.popleft()
            self._trail_z.popleft()
            self._trail_ts.popleft()
        gt_vis = self._marker_visible.get("GT", True)
        if gt_vis:
            self._trail_item.setData(list(self._trail_x), list(self._trail_z))
        else:
            self._trail_item.setData([], [])
        yaw_deg = self._quat_yaw(
            frame["att_x"], frame["att_y"], frame["att_z"], frame["att_w"],
        )
        self._arrow.setPos(px, pz)
        self._arrow.setOpacity(1.0 if gt_vis else 0.0)
        self._arrow.setStyle(angle=90 - yaw_deg)

    def clear_trail(self) -> None:
        self._trail_x.clear()
        self._trail_z.clear()
        self._trail_ts.clear()
        self._trail_item.setData([], [])

    # ── drawing helpers ───────────────────────────────────────────────────

    def _draw_gate_h(self, gate: dict, color: str, is_two_level: bool) -> None:
        """Рисует букву Н (две стойки + перекладина), опционально с разделителями."""
        gx, _, gz = gate["position"]
        ry        = math.radians(gate["rotation"][1])
        hw        = gate["size"][0] / 2
        depth     = hw * 0.5          # длина стоек в направлении пролёта
        cos_r, sin_r = math.cos(ry), math.sin(ry)

        def w(lx: float, lz: float) -> tuple[float, float]:
            # Вращение по часовой стрелке: 0°=горизонт, рост угла → CW
            return gx + lx * cos_r + lz * sin_r, gz - lx * sin_r + lz * cos_r

        pen = pg.mkPen(color, width=2.5)

        # Левая стойка
        p1, p2 = w(-hw,  depth / 2), w(-hw, -depth / 2)
        self._add_line([p1[0], p2[0]], [p1[1], p2[1]], pen)

        # Правая стойка
        p3, p4 = w(hw,  depth / 2), w(hw, -depth / 2)
        self._add_line([p3[0], p4[0]], [p3[1], p4[1]], pen)

        # Перекладина
        p5, p6 = w(-hw, 0), w(hw, 0)
        self._add_line([p5[0], p6[0]], [p5[1], p6[1]], pen)

        if is_two_level:
            # Два коротких внутренних штыря у перекладины
            for ix in (-hw * 0.2, hw * 0.2):
                pa = w(ix,  depth * 0.25)
                pb = w(ix, -depth * 0.25)
                self._add_line([pa[0], pb[0]], [pa[1], pb[1]], pen)

    def _draw_flag(self, gate: dict, in_seq: bool = True) -> None:
        """
        Рисует флаг в виде «+», у которого длинный луч направлен по rotation[1].
        Формула: long_dir = (cos(ry), sin(ry)) в пространстве (X, Z) карты.
          rotation=270° → (0, −1) = вниз   rotation=90°  → (0, +1) = вверх
          rotation=0°   → (+1, 0) = вправо  rotation=180° → (−1, 0) = влево
        Три коротких луча = _FLAG_ARM, длинный = _FLAG_TOP (×2).
        Подпись id — под элементом (в сторону, противоположную длинному лучу).
        Цвет — _CLR_FLAG (совпадает с воротами первого этажа).
        """
        gx, _, gz = gate["position"]
        ry = math.radians(gate["rotation"][1])

        # Единичный вектор длинного луча
        # 270° → (0,−1)=вниз  90° → (0,+1)=вверх  0° → (−1,0)=влево  180° → (+1,0)=вправо
        ldx = -math.cos(ry)   # ось X карты
        ldz =  math.sin(ry)   # ось Z карты
        # Перпендикуляр (левый/правый лучи)
        px, pz = -ldz, ldx

        a   = _FLAG_ARM
        top = _FLAG_TOP
        pen = pg.mkPen(_CLR_FLAG, width=2.0)

        # Длинный луч
        self._add_line([gx, gx + ldx * top], [gz, gz + ldz * top], pen)
        # Короткий луч — в обратную сторону
        self._add_line([gx, gx - ldx * a],   [gz, gz - ldz * a  ], pen)
        # Левый и правый лучи
        self._add_line([gx, gx + px * a],    [gz, gz + pz * a   ], pen)
        self._add_line([gx, gx - px * a],    [gz, gz - pz * a   ], pen)

        # Бейдж id — стиль как у ворот первого этажа.
        # Смещён на конец длинного луча (противоположная сторона от короткого).
        badge_size = 0.4
        bx = gx + ldx * (top + badge_size / 2 + 0.1)
        bz = gz + ldz * (top + badge_size / 2 + 0.1)
        fill, border = _BADGE_LOW
        if not in_seq:
            fill = _FILL_OFFSEQ
        badge = BadgeItem(str(gate["id"]), fill, border, size_meters=badge_size)
        self.addItem(badge)
        badge.setPos(bx, bz)
        self._gate_items.append(badge)

    def _add_line(self, xs: list, zs: list, pen) -> None:
        item = self.plot(xs, zs, pen=pen)
        self._gate_items.append(item)

    def _draw_badge(self, gate: dict, num: str,
                    fly_dir: tuple[float, float],
                    fill: str, border: str,
                    extra: float = 0.0,
                    size: float = 0.4) -> None:
        """
        Бейдж с номером: снизу-слева при взгляде со стороны входа.
        Рисуется только квадрат (текст временно отключён).
        extra — дополнительное смещение вдоль оси входа (для стекирования).
        size — размер бейджа в метрах.
        """
        gx, _, gz = gate["position"]
        hw    = gate["size"][0] / 2
        depth = hw * 0.5
        fx, fz = fly_dir          # единичный вектор пролёта
        lx, lz = -fz, fx          # левый вектор при взгляде в fly_dir

        # Точка у левой стойки со стороны входа
        bx = gx - fx * (depth / 2 + size + extra) + lx * (hw + size/2)
        bz = gz - fz * (depth / 2 + size + extra) + lz * (hw + size/2)

        badge = BadgeItem(num, fill, border, size)
        self.addItem(badge)
        badge.setPos(bx, bz)
        self._gate_items.append(badge)

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _get_fly_dirs_from_rotation(
        gate_map: dict[int, dict],
    ) -> dict[int, tuple[float, float]]:
        """
        Для каждого gate_id — направление пролёта из rotation[1].
        Возвращает единичный вектор (nx, nz) — нормаль к воротам.
        """
        dirs: dict[int, tuple[float, float]] = {}
        for gid, gate in gate_map.items():
            ry = math.radians(gate["rotation"][1])
            # Направление пролёта при CW-конвенции: (sin(ry), cos(ry))
            # 0°=через ворота по +Z, 90° CW=через по +X, и т.д.
            nx, nz = math.sin(ry), math.cos(ry)
            dirs[gid] = (nx, nz)
        return dirs

    @staticmethod
    def _find_two_level_pairs(gates: list[dict]) -> list[tuple[int, int]]:
        """
        Пары ворот с одинаковыми XZ (±0.15 м) и разными Y (> 0.1 м).
        Возвращает (lower_id, upper_id) — по значению Y.
        """
        pairs = []
        for i, g1 in enumerate(gates):
            for g2 in gates[i + 1:]:
                dx = abs(g1["position"][0] - g2["position"][0])
                dz = abs(g1["position"][2] - g2["position"][2])
                dy = abs(g1["position"][1] - g2["position"][1])
                if dx < 0.15 and dz < 0.15 and dy > 0.1:
                    lo, hi = (g1, g2) if g1["position"][1] <= g2["position"][1] else (g2, g1)
                    pairs.append((lo["id"], hi["id"]))
        return pairs

    @staticmethod
    def _quat_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.degrees(math.atan2(
            2.0 * (qw * qy + qx * qz),
            1.0 - 2.0 * (qy * qy + qz * qz),
        ))
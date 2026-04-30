"""2-D top-down track map.

Форма ворот — буква Н:
  - левая / правая стойки — вертикальные линии на ±hw
  - перекладина — горизонтальная линия по центру
  - двухэтажные + два внутренних штыря у перекладины

Нумерация:
  - SF-ворота получают номер «0»
  - остальные: 1, 2, 3, … в порядке gate_sequence
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
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainterPath, QBrush, QPen, QFont

from dct.gui import theme

_CLR_GATE    = "#0000CC"
_CLR_GATE_SF = "#CC0000"

# Бейдж: (fill, border)
_BADGE_LOW  = ("#f8cecc", "#b85450")   # нижний этаж / одноэтажный
_BADGE_HIGH = ("#dae8fc", "#6c8ebf")   # верхний этаж


class BadgeItem(pg.GraphicsObject):
    """Бейдж с номером: квадрат масштабируется вместе с картой."""
    
    def __init__(self, num: str, fill: str, border: str, size_meters: float = 0.4, view_box=None):
        super().__init__()
        self.num = str(num)
        self.fill = fill
        self.border = border
        self.size_meters = size_meters
        self.rx = size_meters * 0.2  # радиус скругления (20% от размера)
        self.view_box = view_box
        
        self._create_graphics()
    
    def _create_graphics(self):
        """Создаёт графическое представление бейджа."""
        # Создаём путь для скруглённого прямоугольника
        rect = QRectF(-self.size_meters/2, -self.size_meters/2, 
                      self.size_meters, self.size_meters)
        
        path = QPainterPath()
        path.addRoundedRect(rect, self.rx, self.rx)
        
        # Сохраняем путь для отрисовки
        self.path = path
        
        # Настройки пера и кисти
        self.brush = QBrush(QColor(self.fill))
        self.pen = QPen(QColor(self.border))
        self.pen.setWidthF(max(0.02, self.size_meters * 0.05))  # толщина границы
    
    def boundingRect(self):
        return QRectF(-self.size_meters/2, -self.size_meters/2, 
                      self.size_meters, self.size_meters)
    
    def paint(self, p, *args):
        """Рисуем только скруглённый прямоугольник (без текста)."""
        # Рисуем прямоугольник
        p.setBrush(self.brush)
        p.setPen(self.pen)
        p.drawPath(self.path)
        
        # Текст временно отключён
        # if self.view_box:
        #     # Получаем текущий размер квадрата на экране
        #     view_rect = self.view_box.viewRect()
        #     if view_rect:
        #         # Вычисляем размер квадрата в пикселях
        #         x_range = view_rect.width()
        #         y_range = view_rect.height()
        #         
        #         # Получаем размер виджета
        #         widget_size = self.view_box.size()
        #         
        #         if widget_size.width() > 0 and widget_size.height() > 0:
        #             # Приблизительный размер квадрата в пикселях
        #             pixel_size_x = (self.size_meters / x_range) * widget_size.width()
        #             pixel_size_y = (self.size_meters / y_range) * widget_size.height()
        #             pixel_size = min(pixel_size_x, pixel_size_y)
        #             
        #             # Рисуем текст только если квадрат достаточно большой
        #             if pixel_size >= 20:
        #                 # Адаптируем размер шрифта под размер квадрата
        #                 font = QFont("Arial")
        #                 font.setBold(True)
        #                 font_size = max(8, int(pixel_size * 0.6))
        #                 font.setPixelSize(font_size)
        #                 
        #                 p.setFont(font)
        #                 p.setPen(QColor(0, 0, 0))
        #                 
        #                 # Получаем размер текста
        #                 text_rect = p.fontMetrics().boundingRect(self.num)
        #                 
        #                 # Центрируем текст в квадрате
        #                 text_x = -text_rect.width() / 2
        #                 text_y = text_rect.height() / 4
        #                 
        #                 p.drawText(int(text_x), int(text_y), self.num)


class TrackMapWidget(pg.PlotWidget):
    TRAIL_SECS = 3.0
    TRAIL_MAX  = 400

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
        self._has_track = False

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

    def setup_track(self, track_data: dict[str, Any]) -> None:
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

        # ── Формы ворот ──────────────────────────────────────────────────
        for gate in gates:
            gid = gate["id"]
            if gid in two_level_high:
                continue   # XZ совпадает с нижним — рисуем один раз
            is_sf        = gate.get("is_start_finish", False)
            is_two_level = gid in two_level_all
            color        = _CLR_GATE_SF if is_sf else _CLR_GATE
            self._draw_gate_h(gate, color, is_two_level)

        # ── Бейджи номеров ───────────────────────────────────────────────
        # Размер бейджа в метрах (квадрат масштабируется)
        badge_size = 0.4  # 40 сантиметров
        
        # SF-ворота → «0»
        if sf_id is not None and sf_id in gate_map:
            sf_fly = fly_dirs.get(sf_id, (0.0, 1.0))
            self._draw_badge(gate_map[sf_id], "0", sf_fly,
                             *_BADGE_LOW, extra=0.0, size=badge_size)

        label_num  = 1
        visit_cnt: dict[int, int] = {}

        for gid in seq:
            if gid == sf_id or gid not in gate_map:
                continue
            gate = gate_map[gid]
            visit_cnt[gid] = visit_cnt.get(gid, 0) + 1
            visit = visit_cnt[gid]

            # Для каждого входа используем одно и то же направление из rotation
            fly_dir = fly_dirs.get(gid, (0.0, 1.0))

            is_high = gid in two_level_high
            fill, border = _BADGE_HIGH if is_high else _BADGE_LOW

            # Второй бейдж при одинаковом направлении сдвигаем дальше от ворот
            extra = 0.5 if (is_high and gid in same_dir_pairs) else 0.0

            self._draw_badge(gate, str(label_num), fly_dir, fill, border, extra, badge_size)
            label_num += 1

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

            circ = pg.ScatterPlotItem(
                [sp_x], [sp_z], symbol="o", size=22,
                brush=pg.mkBrush(None),
                pen=pg.mkPen("#ffffff", width=2),
            )
            self.addItem(circ)
            self._start_items.append(circ)

            h_lbl = pg.TextItem(anchor=(0.5, 0.5))
            h_lbl.setHtml(
                '<span style="color:#ffffff;font-size:10px;font-family:Arial;">H</span>'
            )
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
        self._trail_item.setData(list(self._trail_x), list(self._trail_z))
        yaw_deg = self._quat_yaw(
            frame["att_x"], frame["att_y"], frame["att_z"], frame["att_w"],
        )
        self._arrow.setPos(px, pz)
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
            return gx + lx * cos_r - lz * sin_r, gz + lx * sin_r + lz * cos_r

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

        # Создаём бейдж (только квадрат, без текста)
        badge = BadgeItem(num, fill, border, size, self.getViewBox())
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
            # Нормаль к плоскости ворот (перпендикуляр к плоскости XY)
            nx, nz = -math.sin(ry), math.cos(ry)
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
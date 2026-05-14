"""Video preview widget — показывает кадры из live-захвата или replay.

Оптимизации для минимальной нагрузки на Qt main thread:
- VideoReader уже конвертирует BGR→RGB в фоне (replay)
- Live-кадры (BGR) конвертируются здесь, но после resize до размера виджета:
  resize 1280×720 → ~300×200 уменьшает данные в 6×, что ускоряет остальные шаги
- QPixmap.scaled() не используется — масштабирование делает cv2 (быстрее)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from dct.gui import theme


class VideoPreviewWidget(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background-color: #0D0D0D; color: {theme.DIM};"
            f"border: 1px solid {theme.BORDER}; border-radius: 4px;"
        )
        self._is_rgb = False  # флаг: VideoReader шлёт RGB, ScreenRecorder — BGR
        self._overlay_enabled = False
        self._gate_overlay: list[dict[str, Any]] = []
        self._show_placeholder()

    def set_gate_overlay_enabled(self, enabled: bool) -> None:
        """Enable/disable drawing camera gate detections over the preview frame."""
        self._overlay_enabled = bool(enabled)

    def set_gate_overlay(self, detections: list[dict[str, Any]] | None) -> None:
        """Set YOLO/PnP gate detections to draw on the next preview frames.

        Expected detection fields are intentionally loose for the first Replay
        integration step:

        - `bbox_xyxy`: [x1, y1, x2, y2] in source image pixels;
        - `keypoints`: [[x, y], ...] in source image pixels;
        - `gate_id`: optional gate id or list of ids;
        - `confidence`: optional 0..1 score.
        """
        self._gate_overlay = list(detections or [])

    def update_frame(self, frame: np.ndarray, is_rgb: bool = False) -> None:
        """Принимает numpy-кадр (BGR от ScreenRecorder или RGB от VideoReader)."""
        try:
            dst_w = max(16, self.width())
            dst_h = max(16, self.height())
            src_h, src_w = frame.shape[:2]

            # Масштаб с сохранением aspect ratio (letterbox — без деформации)
            scale = min(dst_w / src_w, dst_h / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))

            if new_w != src_w or new_h != src_h:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # BGR → RGB (только для live-кадров; VideoReader уже даёт RGB)
            if not is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self._overlay_enabled and self._gate_overlay:
                self._draw_gate_overlay(frame, scale)

            h, w = frame.shape[:2]
            data = np.ascontiguousarray(frame)
            qimg = QImage(data.data, w, h, w * 3, QImage.Format.Format_RGB888)
            # fromImage делает внутреннюю копию данных — data может быть GC'd после
            self.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def clear_frame(self) -> None:
        """Сбросить в плейсхолдер (при переключении режима)."""
        self.clear()
        self._show_placeholder()

    def _draw_gate_overlay(self, frame_rgb: np.ndarray, scale: float) -> None:
        color = (197, 134, 192)  # RGB, theme.LOCALIZER_CAM
        point_color = (255, 220, 255)
        for det in self._gate_overlay:
            bbox = det.get("bbox_xyxy")
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(float(v) * scale) for v in bbox]
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)

            keypoints = det.get("keypoints")
            if keypoints is not None:
                pts = np.asarray(keypoints, dtype=float).reshape(-1, 2)
                scaled = []
                for x, y in pts:
                    px, py = int(x * scale), int(y * scale)
                    scaled.append((px, py))
                    cv2.circle(frame_rgb, (px, py), 3, point_color, -1)
                if len(scaled) >= 2:
                    cv2.polylines(frame_rgb, [np.asarray(scaled, dtype=np.int32)], True, color, 1)

            label = self._overlay_label(det)
            if label and bbox is not None and len(bbox) == 4:
                x1, y1 = int(float(bbox[0]) * scale), int(float(bbox[1]) * scale)
                cv2.putText(
                    frame_rgb,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    @staticmethod
    def _overlay_label(det: dict[str, Any]) -> str:
        parts = []
        if det.get("gate_id") is not None:
            parts.append(f"gate {det['gate_id']}")
        if det.get("confidence") is not None:
            try:
                parts.append(f"{float(det['confidence']):.2f}")
            except Exception:
                pass
        return " ".join(parts)

    def _show_placeholder(self) -> None:
        self.setText("[ Video Preview ]")

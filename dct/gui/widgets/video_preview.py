"""Video preview widget — показывает кадры из live-захвата или replay.

Оптимизации для минимальной нагрузки на Qt main thread:
- VideoReader уже конвертирует BGR→RGB в фоне (replay)
- Live-кадры (BGR) конвертируются здесь, но после resize до размера виджета:
  resize 1280×720 → ~300×200 уменьшает данные в 6×, что ускоряет остальные шаги
- QPixmap.scaled() не используется — масштабирование делает cv2 (быстрее)
"""
from __future__ import annotations

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
        self._show_placeholder()

    def update_frame(self, frame: np.ndarray, is_rgb: bool = False) -> None:
        """Принимает numpy-кадр (BGR от ScreenRecorder или RGB от VideoReader)."""
        try:
            dst_w = max(16, self.width())
            dst_h = max(16, self.height())

            # Масштабируем до размера виджета ДО конвертации цвета:
            # это уменьшает объём данных для следующих шагов в 4–10x.
            src_h, src_w = frame.shape[:2]
            if src_w != dst_w or src_h != dst_h:
                frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

            # BGR → RGB (только для live-кадров; VideoReader уже даёт RGB)
            if not is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            # np.ascontiguousarray гарантирует непрерывность памяти для QImage
            data = np.ascontiguousarray(frame)
            qimg = QImage(data.data, w, h, w * 3, QImage.Format.Format_RGB888)
            # fromImage делает внутреннюю копию, поэтому data может быть GC'd после
            self.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def clear_frame(self) -> None:
        """Сбросить в плейсхолдер (при переключении режима)."""
        self.clear()
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.setText("[ Video Preview ]")

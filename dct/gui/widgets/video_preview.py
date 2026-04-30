"""Video preview widget — показывает кадры из live-захвата или replay."""
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
        self._show_placeholder()

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Принимает numpy BGR-кадр от ScreenRecorder или cv2.VideoCapture."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.setPixmap(pixmap)
        except Exception:
            pass

    def clear_frame(self) -> None:
        """Сбросить в плейсхолдер (при переключении режима)."""
        self.clear()
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.setText("[ Video Preview ]")

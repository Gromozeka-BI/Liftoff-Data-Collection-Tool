"""Video preview placeholder — will show live capture feed in Stage 4."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from dct.gui import theme


class VideoPreviewWidget(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setText("[ Video Preview — Stage 4 ]")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(150)
        self.setStyleSheet(
            f"background-color: #0D0D0D; color: {theme.DIM};"
            f"border: 1px solid {theme.BORDER}; border-radius: 4px;"
        )

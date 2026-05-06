"""Floating video Picture-in-Picture used in Race mode."""
from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from dct.gui import theme, ui_settings
from dct.gui.widgets.video_preview import VideoPreviewWidget


class VideoPiP(QFrame):
    closed = pyqtSignal()

    DEFAULT_SIZE = (300, 170)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("video_pip")
        self.setStyleSheet(
            f"#video_pip {{ background-color: {theme.PANEL}; border: 1px solid {theme.BORDER}; }}",
        )
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(*self.DEFAULT_SIZE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 0)
        bar.setSpacing(2)
        bar.addStretch(1)
        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(20, 18)
        self._btn_close.setProperty("role", "icon")
        self._btn_close.clicked.connect(self._handle_close)
        bar.addWidget(self._btn_close)
        outer.addLayout(bar)

        self.video = VideoPreviewWidget()
        self.video.setMinimumSize(280, 140)
        outer.addWidget(self.video, stretch=1)

        self._drag_pos: QPoint | None = None
        self._restore_position()

    # ── public API ─────────────────────────────────────────────────────────

    def update_frame(self, frame, is_rgb: bool = False) -> None:
        self.video.update_frame(frame, is_rgb=is_rgb)

    # ── moving ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._drag_pos is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            new_pos = ev.globalPosition().toPoint() - self._drag_pos
            self.move(self._clamp_pos(new_pos))
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self._save_position()
        super().mouseReleaseEvent(ev)

    def _clamp_pos(self, p: QPoint) -> QPoint:
        parent = self.parentWidget()
        if parent is None:
            return p
        x = max(0, min(p.x(), parent.width() - self.width()))
        y = max(0, min(p.y(), parent.height() - self.height()))
        return QPoint(x, y)

    def _save_position(self) -> None:
        d = ui_settings.load()
        race = d.setdefault("race", {})
        race["pip_pos"] = [int(self.x()), int(self.y())]
        ui_settings.save(d)

    def _restore_position(self) -> None:
        race = ui_settings.load().get("race", {})
        pos = race.get("pip_pos")
        if isinstance(pos, list) and len(pos) == 2:
            self.move(int(pos[0]), int(pos[1]))

    def _handle_close(self) -> None:
        self.closed.emit()
        self.hide()

"""Right-hand collapsible sidebar that hosts Setup/Replay pages.

The visible page is driven entirely by the main mode (Record vs Replay)
selected from :class:`TopBar`. The sidebar itself no longer renders its own
tab bar or collapse button — those concerns moved to the top bar.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

PAGE_SETUP  = 0
PAGE_REPLAY = 1


class Sidebar(QFrame):
    page_changed      = pyqtSignal(int)
    collapsed_changed = pyqtSignal(bool)

    DEFAULT_WIDTH = 262

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        width: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar_root")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._fixed_width = int(width) if width is not None else self.DEFAULT_WIDTH
        self.setFixedWidth(self._fixed_width)
        self._collapsed = False

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

    @property
    def fixed_width(self) -> int:
        return self._fixed_width

    def set_fixed_width_px(self, width: int) -> None:
        self._fixed_width = int(width)
        self.setFixedWidth(self._fixed_width)

    # ── public API ─────────────────────────────────────────────────────────

    def add_page(self, widget: QWidget) -> int:
        return self._stack.addWidget(widget)

    def set_page(self, idx: int) -> None:
        if idx < 0 or idx >= self._stack.count():
            return
        self._stack.setCurrentIndex(idx)
        self.page_changed.emit(idx)

    def current_page(self) -> int:
        return self._stack.currentIndex()

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self.setVisible(not collapsed)
        self.collapsed_changed.emit(collapsed)

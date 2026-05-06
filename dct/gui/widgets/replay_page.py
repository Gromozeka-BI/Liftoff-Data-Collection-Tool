"""Sidebar Replay page — folder picker, sessions list, hotkey hints."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from dct.gui import ui_settings


def _session_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])


class ReplayPage(QWidget):
    session_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Folder & sessions ─────────────────────────────────────────────
        box = QGroupBox("Replay sessions")
        lay = QVBoxLayout(box)
        lay.setSpacing(4)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        self._lbl_dir = QLabel(ui_settings.load().get("replay_dir", "sessions"))
        self._lbl_dir.setProperty("role", "dim")
        self._lbl_dir.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        self._lbl_dir.setToolTip("Папка с сессиями для Replay")
        btn_browse = QPushButton("📁")
        btn_browse.setProperty("role", "icon")
        btn_browse.setFixedWidth(28)
        btn_browse.setToolTip("Выбрать папку сессий")
        btn_browse.clicked.connect(self._browse_dir)
        folder_row.addWidget(self._lbl_dir, stretch=1)
        folder_row.addWidget(btn_browse)
        lay.addLayout(folder_row)

        self._combo = QComboBox()
        self._combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._combo.setToolTip("Выбрать сессию для воспроизведения")
        self._combo.currentIndexChanged.connect(self._on_session_changed)
        lay.addWidget(self._combo)

        root.addWidget(box)

        # ── Hotkey hints ─────────────────────────────────────────────────
        hints = QGroupBox("Hotkeys")
        hl = QVBoxLayout(hints)
        hl.setSpacing(2)
        for line in [
            "Space — Play / Pause",
            "8 — +LAP    9 — +GATE    0 — +S/F",
            "Del / Backspace — удалить выбранный",
            "Drag метки → перенос (Shift/Ctrl/Alt — снэп)",
            "← / → — ±1 кадр телеметрии  (Shift+ ← → ±100 мс)",
        ]:
            lbl = QLabel(line)
            lbl.setProperty("role", "dim")
            lbl.setWordWrap(True)
            hl.addWidget(lbl)
        root.addWidget(hints)

        root.addStretch(1)
        self.reload_sessions()

    # ── public API ─────────────────────────────────────────────────────────

    def reload_sessions(self) -> None:
        base = Path(ui_settings.load().get("replay_dir", "sessions"))
        self._combo.blockSignals(True)
        self._combo.clear()
        for d in _session_dirs(base):
            self._combo.addItem(d.name, userData=str(d))
        self._combo.blockSignals(False)
        if self._combo.count():
            self._combo.setCurrentIndex(self._combo.count() - 1)
            path = self._combo.itemData(self._combo.count() - 1)
            if path:
                self.session_selected.emit(path)

    # ── private slots ──────────────────────────────────────────────────────

    def _browse_dir(self) -> None:
        current = ui_settings.load().get("replay_dir", "sessions")
        folder = QFileDialog.getExistingDirectory(
            self, "Папка с сессиями для Replay", current,
        )
        if folder:
            self._lbl_dir.setText(folder)
            ui_settings.update("replay_dir", folder)
            self.reload_sessions()

    def _on_session_changed(self, idx: int) -> None:
        path = self._combo.itemData(idx)
        if path:
            self.session_selected.emit(path)

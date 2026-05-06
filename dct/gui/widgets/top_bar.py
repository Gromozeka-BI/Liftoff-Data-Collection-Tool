"""Top bar: mode switcher, setup summary, REC indicator, sidebar toggle."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget,
)

from dct.gui import theme

_LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.png"

MODE_RECORD = 0
MODE_REPLAY = 1
MODE_RACE   = 2


class _ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class TopBar(QFrame):
    mode_changed   = pyqtSignal(int)
    summary_clicked = pyqtSignal()
    toggle_sidebar  = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "topbar")
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        if _LOGO_PATH.exists():
            logo_lbl = QLabel()
            pix = QPixmap(str(_LOGO_PATH)).scaledToHeight(
                26, Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl.setPixmap(pix)
            lay.addWidget(logo_lbl)

        self._btn_rec = QPushButton("● RECORD")
        self._btn_rep = QPushButton("▶ REPLAY")
        self._btn_race = QPushButton("⚑ RACE")
        for i, btn in enumerate((self._btn_rec, self._btn_rep, self._btn_race)):
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.setMinimumWidth(110)
            btn.clicked.connect(lambda _checked, m=i: self._on_mode_clicked(m))
            lay.addWidget(btn)
        self._btn_rec.setChecked(True)

        self._lbl_summary = _ClickableLabel("—")
        self._lbl_summary.setProperty("role", "dim")
        self._lbl_summary.setToolTip(
            "Активная конфигурация Setup. Нажмите, чтобы развернуть Sidebar.",
        )
        self._lbl_summary.clicked.connect(self.summary_clicked)
        self._lbl_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        self._lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._lbl_summary, stretch=1)

        self._lbl_rec = QLabel("● IDLE")
        self._lbl_rec.setProperty("role", "dim")
        self._lbl_rec.setMinimumWidth(64)
        lay.addWidget(self._lbl_rec)

        self._lbl_dur = QLabel("0.0 s")
        self._lbl_dur.setProperty("role", "dim")
        self._lbl_dur.setMinimumWidth(64)
        self._lbl_dur.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._lbl_dur)

        self._lbl_hz = QLabel("0 Hz")
        self._lbl_hz.setProperty("role", "dim")
        self._lbl_hz.setMinimumWidth(60)
        self._lbl_hz.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._lbl_hz)

        self._btn_sidebar = QPushButton("›")
        self._btn_sidebar.setProperty("role", "icon")
        self._btn_sidebar.setToolTip("Свернуть/развернуть боковую панель")
        self._btn_sidebar.setFixedWidth(28)
        self._btn_sidebar.clicked.connect(self.toggle_sidebar)
        lay.addWidget(self._btn_sidebar)

    # ── public API ─────────────────────────────────────────────────────────

    def set_mode(self, mode: int) -> None:
        self._btn_rec.setChecked(mode == MODE_RECORD)
        self._btn_rep.setChecked(mode == MODE_REPLAY)
        self._btn_race.setChecked(mode == MODE_RACE)

    def set_summary(self, text: str) -> None:
        self._lbl_summary.setText(text or "—")

    def set_recording(self, active: bool) -> None:
        if active:
            self._lbl_rec.setText("● REC")
            self._lbl_rec.setStyleSheet(
                f"color: {theme.ERR}; font-weight: 700;",
            )
        else:
            self._lbl_rec.setText("● IDLE")
            self._lbl_rec.setStyleSheet(
                f"color: {theme.DIM}; font-weight: 700;",
            )

    def set_stats(self, *, duration: float, hz: float) -> None:
        self._lbl_dur.setText(f"{duration:.1f} s")
        self._lbl_hz.setText(f"{hz:.0f} Hz")

    def set_sidebar_collapsed(self, collapsed: bool) -> None:
        self._btn_sidebar.setText("›" if collapsed else "‹")

    def _on_mode_clicked(self, mode: int) -> None:
        self.set_mode(mode)
        self.mode_changed.emit(mode)

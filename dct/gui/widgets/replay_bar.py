"""Bottom control bar for REPLAY mode.

Layout (three equal-proportion blocks in one row):
  ┌─ LEFT  20% ──────┬─ CENTER 60% ────────────────────────────────┬─ RIGHT 20% ─┐
  │ session selector  │ ▶ [event_strip / slider] MM:SS.fff ×speed   │  status     │
  │ folder path       │ [+LAP] [+GATE] [+S/F]  |  [DEL]  (centred)  │  legend     │
  └──────────────────┴──────────────────────────────────────────────┴─────────────┘
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from dct.gui import theme, ui_settings
from dct.gui.widgets.status_panel import StatusPanel
from dct.gui.widgets.event_strip import EventStrip


# ── helpers ────────────────────────────────────────────────────────────────────

def _session_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])


def _fmt(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m}:{s:02d}"


def _fmt_ms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s_int = divmod(int(seconds), 60)
    ms = int((seconds % 1) * 1000)
    return f"{m}:{s_int:02d}.{ms:03d}"


def _vline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setStyleSheet(f"color: {theme.DIM};")
    sep.setFixedWidth(12)
    return sep


# ── widget ─────────────────────────────────────────────────────────────────────

class ReplayBar(QWidget):
    session_selected       = pyqtSignal(str)
    play_pause             = pyqtSignal()
    seek_fraction          = pyqtSignal(float)
    speed_changed          = pyqtSignal(float)
    # Event editor
    event_add_requested    = pyqtSignal(str, int)   # (event_type, gate_id)
    event_delete_requested = pyqtSignal(dict)
    event_seek_requested   = pyqtSignal(float)       # absolute ts_wall

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._t0 = 0.0
        self._t1 = 1.0
        self._gates: list[dict] = []
        self._selected_event: dict | None = None
        self._session_active = False

        # ── Root: three side-by-side blocks ───────────────────────────────
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(6, 4, 6, 4)

        root.addWidget(self._build_left(),   stretch=1)
        root.addWidget(self._build_center(), stretch=3)
        root.addWidget(self._build_right(),  stretch=1)

        # ── Keyboard shortcuts (guarded by _session_active) ────────────────
        for key, slot in [("8", self._on_add_lap_safe),
                           ("9", self._on_add_gate_safe),
                           ("0", self._on_add_sf_safe)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)

        for key, slot in [(Qt.Key.Key_Delete, self._on_delete_safe),
                           (Qt.Key.Key_Backspace, self._on_delete_safe)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)

        self.reload_sessions()

    # ── block builders ──────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 0, 0)

        # Folder row
        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        self._lbl_replay_dir = QLabel()
        self._lbl_replay_dir.setStyleSheet(f"color: {theme.DIM}; font-size: 9px;")
        saved = ui_settings.load().get("replay_dir", "sessions")
        self._lbl_replay_dir.setText(str(saved))
        self._lbl_replay_dir.setToolTip("Папка с сессиями для Replay")
        self._lbl_replay_dir.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        btn_browse = QPushButton("📁")
        btn_browse.setFixedWidth(26)
        btn_browse.setToolTip("Выбрать папку сессий для Replay")
        btn_browse.clicked.connect(self._browse_replay_dir)
        folder_row.addWidget(self._lbl_replay_dir)
        folder_row.addWidget(btn_browse)
        lay.addLayout(folder_row)

        # Session combo
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.setToolTip("Выбрать сессию для воспроизведения")
        self._combo.currentIndexChanged.connect(self._on_session_changed)
        lay.addWidget(self._combo)

        lay.addStretch()
        return w

    def _build_center(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(3)
        lay.setContentsMargins(0, 0, 0, 0)

        # ── Transport row ────────────────────────────────────────────────
        transport = QHBoxLayout()
        transport.setSpacing(6)

        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setToolTip("Play / Pause  (Space)")
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(80)
        self._btn_play.clicked.connect(self.play_pause)
        transport.addWidget(self._btn_play)

        # Timeline column (event strip above slider)
        timeline_col = QWidget()
        timeline_lay = QVBoxLayout(timeline_col)
        timeline_lay.setSpacing(1)
        timeline_lay.setContentsMargins(0, 0, 0, 0)

        self.event_strip = EventStrip()
        self.event_strip.marker_clicked.connect(self._on_marker_clicked)
        self.event_strip.marker_selected.connect(self._on_marker_selected)
        timeline_lay.addWidget(self.event_strip)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 10000)
        self._slider.setToolTip("Scrub through the session timeline")
        self._slider.sliderMoved.connect(lambda v: self.seek_fraction.emit(v / 10000.0))
        timeline_lay.addWidget(self._slider)

        transport.addWidget(timeline_col, stretch=1)

        # Precise time label
        self._lbl_time = QLabel("0:00.000 / 0:00")
        self._lbl_time.setStyleSheet(
            f"color: {theme.DIM}; font-family: monospace; font-size: 11px;"
        )
        self._lbl_time.setMinimumWidth(130)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        transport.addWidget(self._lbl_time)

        # Lap label
        self._lbl_lap = QLabel("")
        self._lbl_lap.setStyleSheet(f"color: {theme.OK}; font-size: 10px;")
        self._lbl_lap.setMinimumWidth(60)
        transport.addWidget(self._lbl_lap)

        # Speed buttons
        for label, speed in [("½x", 0.5), ("1x", 1.0), ("2x", 2.0)]:
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.setToolTip(f"Playback speed {label}")
            btn.clicked.connect(lambda _checked, s=speed: self.speed_changed.emit(s))
            transport.addWidget(btn)

        lay.addLayout(transport)

        # ── Event editor row (centred) ────────────────────────────────────
        editor = QHBoxLayout()
        editor.setSpacing(6)
        editor.addStretch()   # push buttons to centre

        self._btn_add_lap = QPushButton("+ LAP  (8)")
        self._btn_add_lap.setToolTip("Добавить button_lap в текущей позиции  (клавиша 8)")
        self._btn_add_lap.setFixedWidth(100)
        self._btn_add_lap.clicked.connect(self._on_add_lap)
        editor.addWidget(self._btn_add_lap)

        self._btn_add_gate = QPushButton("+ GATE  (9)")
        self._btn_add_gate.setToolTip("Добавить button_gate в текущей позиции  (клавиша 9)")
        self._btn_add_gate.setFixedWidth(108)
        self._btn_add_gate.clicked.connect(self._on_add_gate)
        editor.addWidget(self._btn_add_gate)

        self._btn_add_sf = QPushButton("+ S/F  (0)")
        self._btn_add_sf.setToolTip("Добавить rh_lap в текущей позиции  (клавиша 0)")
        self._btn_add_sf.setFixedWidth(96)
        self._btn_add_sf.clicked.connect(self._on_add_sf)
        editor.addWidget(self._btn_add_sf)

        editor.addWidget(_vline())

        self._btn_del = QPushButton("🗑  Удалить  (Del)")
        self._btn_del.setToolTip("Удалить выбранный маркер события  (Del / Backspace)")
        self._btn_del.setFixedWidth(130)
        self._btn_del.setEnabled(False)
        self._btn_del.clicked.connect(self._on_delete_selected)
        editor.addWidget(self._btn_del)

        editor.addStretch()   # symmetric right padding
        lay.addLayout(editor)

        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 0, 0)

        self.status = StatusPanel()
        lay.addWidget(self.status)

        # Colour legend (vertical, compact)
        legend_items = [
            ("#00cc44", "▲ session_start"),
            ("#dd2200", "▼ session_stop"),
            ("#3399ff", "● button_lap"),
            ("#ffcc00", "◆ button_gate"),
            ("#ff8800", "▲ rh_lap  (S/F)"),
            ("#cc44ff", "◇ rh_gate"),
        ]
        for color, label in legend_items:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {color}; font-size: 8px;")
            lay.addWidget(lbl)

        lay.addStretch()
        return w

    # ── public API ──────────────────────────────────────────────────────────

    def reload_sessions(self) -> None:
        base = Path(ui_settings.load().get("replay_dir", "sessions"))
        self._combo.blockSignals(True)
        self._combo.clear()
        for d in _session_dirs(base):
            self._combo.addItem(d.name, userData=str(d))
        self._combo.blockSignals(False)
        if self._combo.count():
            self._on_session_changed(self._combo.count() - 1)
            self._combo.setCurrentIndex(self._combo.count() - 1)

    def update_progress(self, current_s: float, total_s: float) -> None:
        if total_s > 0:
            self._slider.setValue(int(current_s / total_s * 10000))
        self._lbl_time.setText(f"{_fmt_ms(current_s)} / {_fmt(total_s)}")
        self.event_strip.set_playhead(self._t0 + current_s)

    def set_session_time_range(self, t0: float, t1: float) -> None:
        self._t0 = t0
        self._t1 = t1
        self._session_active = True

    def set_events(self, events: list[dict]) -> None:
        self.event_strip.set_events(events, self._t0, self._t1)

    def set_gates(self, gates: list[dict]) -> None:
        self._gates = gates

    def set_playing(self, playing: bool) -> None:
        self._btn_play.setChecked(playing)
        self._btn_play.setText("⏸  Pause" if playing else "▶  Play")

    def set_lap(self, lap: int, total: int) -> None:
        self._lbl_lap.setText(f"Круг {lap}/{total}" if total else "")

    def deactivate(self) -> None:
        """Called when leaving Replay mode."""
        self._session_active = False
        self.event_strip.clear_selection()
        self._selected_event = None
        self._btn_del.setEnabled(False)

    # ── event strip callbacks ───────────────────────────────────────────────

    def _on_marker_clicked(self, ev: dict) -> None:
        self.event_seek_requested.emit(ev["ts_wall"])

    def _on_marker_selected(self, ev) -> None:
        self._selected_event = ev if isinstance(ev, dict) else None
        self._btn_del.setEnabled(self._selected_event is not None)

    # ── add / delete ────────────────────────────────────────────────────────

    def _on_add_lap(self) -> None:
        self.event_add_requested.emit("button_lap", -1)

    def _on_add_gate(self) -> None:
        from dct.gui.widgets.gate_pick_dialog import GatePickDialog
        dlg = GatePickDialog(self._gates, self)
        if dlg.exec():
            self.event_add_requested.emit("button_gate", dlg.selected_gate_id())

    def _on_add_sf(self) -> None:
        self.event_add_requested.emit("rh_lap", -1)

    def _on_delete_selected(self) -> None:
        if self._selected_event:
            ev, self._selected_event = self._selected_event, None
            self._btn_del.setEnabled(False)
            self.event_strip.clear_selection()
            self.event_delete_requested.emit(ev)

    # guarded (only fire when a session is active in Replay mode)
    def _on_add_lap_safe(self)  -> None:
        if self._session_active: self._on_add_lap()

    def _on_add_gate_safe(self) -> None:
        if self._session_active: self._on_add_gate()

    def _on_add_sf_safe(self)   -> None:
        if self._session_active: self._on_add_sf()

    def _on_delete_safe(self)   -> None:
        if self._session_active: self._on_delete_selected()

    # ── internal ────────────────────────────────────────────────────────────

    def _browse_replay_dir(self) -> None:
        current = ui_settings.load().get("replay_dir", "sessions")
        folder = QFileDialog.getExistingDirectory(
            self, "Папка с сессиями для Replay", current,
        )
        if folder:
            self._lbl_replay_dir.setText(folder)
            ui_settings.update("replay_dir", folder)
            self.reload_sessions()

    def _on_session_changed(self, idx: int) -> None:
        path = self._combo.itemData(idx)
        if path:
            self._session_active = False
            self.session_selected.emit(path)

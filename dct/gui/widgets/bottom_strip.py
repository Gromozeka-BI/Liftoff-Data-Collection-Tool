"""Bottom action strip: switches between record and replay modes.

Owns global hotkeys 6/7/8/9/0 and routes Replay editor signals
(EventStrip drag/click + add/delete buttons + inline editor).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from dct.gui import theme
from dct.gui.global_hotkeys import GlobalHotkeyManager
from dct.gui.widgets.event_strip import EventStrip


def _fmt_ms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s_int = divmod(int(seconds), 60)
    ms = int((seconds % 1) * 1000)
    return f"{m}:{s_int:02d}.{ms:03d}"


def _fmt(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m}:{s:02d}"


class _RecordControls(QWidget):
    start_clicked = pyqtSignal()
    stop_clicked  = pyqtSignal()
    lap_clicked   = pyqtSignal()
    gate_clicked  = pyqtSignal()
    sf_clicked    = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 4, 8, 4)
        self._btn_start = self._mk("6  START", "btn_start", lay, self.start_clicked)
        self._btn_stop  = self._mk("7  STOP",  "btn_stop",  lay, self.stop_clicked)
        lay.addStretch(1)
        self._btn_lap   = self._mk("8  LAP",   "",          lay, self.lap_clicked)
        self._btn_gate  = self._mk("9  GATE",  "",          lay, self.gate_clicked)
        self._btn_sf    = self._mk("0  S/F",   "",          lay, self.sf_clicked)

        self._set_recording(False)

    def _set_recording(self, active: bool) -> None:
        self._btn_start.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        self._btn_lap.setEnabled(active)
        self._btn_gate.setEnabled(active)
        self._btn_sf.setEnabled(active)

    @staticmethod
    def _mk(label: str, obj_name: str, lay: QHBoxLayout, sig) -> QPushButton:
        b = QPushButton(label)
        b.setProperty("role", "big")
        if obj_name:
            b.setObjectName(obj_name)
        b.clicked.connect(sig)
        lay.addWidget(b)
        return b


class _ReplayControls(QWidget):
    play_pause_clicked       = pyqtSignal()
    speed_changed            = pyqtSignal(float)
    seek_fraction            = pyqtSignal(float)
    event_add_requested      = pyqtSignal(str, int)
    event_delete_requested   = pyqtSignal(dict)
    event_seek_requested     = pyqtSignal(float)
    event_drag_started       = pyqtSignal(dict)
    event_drag_ended         = pyqtSignal(int, float)
    event_inline_apply       = pyqtSignal(int, str, float, int)  # seq, type, ts, gate_id
    nudge_frame_requested    = pyqtSignal(int)                   # ±1 telemetry sample
    nudge_ms_requested       = pyqtSignal(int)                   # ±100 ms

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t0 = 0.0
        self._t1 = 1.0
        self._gates: list[dict] = []
        self._selected_event: dict | None = None

        outer = QVBoxLayout(self)
        outer.setSpacing(2)
        outer.setContentsMargins(8, 4, 8, 4)

        # ── Transport row ────────────────────────────────────────────────
        transport = QHBoxLayout()
        transport.setSpacing(6)

        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setMinimumWidth(80)
        self._btn_play.clicked.connect(self.play_pause_clicked)
        transport.addWidget(self._btn_play)

        col = QWidget()
        col_lay = QVBoxLayout(col)
        col_lay.setSpacing(1)
        col_lay.setContentsMargins(0, 0, 0, 0)

        self.event_strip = EventStrip()
        self.event_strip.marker_clicked.connect(self._on_marker_clicked)
        self.event_strip.marker_selected.connect(self._on_marker_selected)
        self.event_strip.marker_drag_started.connect(self.event_drag_started)
        self.event_strip.marker_drag_ended.connect(self.event_drag_ended)
        col_lay.addWidget(self.event_strip)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 10000)
        self._slider.sliderMoved.connect(
            lambda v: self.seek_fraction.emit(v / 10000.0),
        )
        col_lay.addWidget(self._slider)

        transport.addWidget(col, stretch=1)

        self._lbl_time = QLabel("0:00.000 / 0:00")
        self._lbl_time.setStyleSheet(
            f"color: {theme.DIM}; font-family: monospace; font-size: {theme.FONT_DIM}px;",
        )
        self._lbl_time.setMinimumWidth(140)
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        transport.addWidget(self._lbl_time)

        self._lbl_lap = QLabel("")
        self._lbl_lap.setStyleSheet(
            f"color: {theme.OK}; font-size: {theme.FONT_DIM}px;",
        )
        self._lbl_lap.setMinimumWidth(60)
        transport.addWidget(self._lbl_lap)

        for label, speed in [("½×", 0.5), ("1×", 1.0), ("2×", 2.0)]:
            btn = QPushButton(label)
            btn.setFixedWidth(44)
            btn.clicked.connect(lambda _checked, s=speed: self.speed_changed.emit(s))
            transport.addWidget(btn)

        outer.addLayout(transport)

        # ── Editor row ───────────────────────────────────────────────────
        editor = QHBoxLayout()
        editor.setSpacing(4)
        for txt, slot, key in [
            ("+ LAP  (8)",  self._add_lap,  "lap"),
            ("+ GATE  (9)", self._add_gate, "gate"),
            ("+ S/F  (0)",  self._add_sf,   "sf"),
        ]:
            b = QPushButton(txt)
            b.clicked.connect(slot)
            editor.addWidget(b)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {theme.DIM};")
        sep.setFixedWidth(12)
        editor.addWidget(sep)

        self._btn_del = QPushButton("🗑  Удалить (Del)")
        self._btn_del.setEnabled(False)
        self._btn_del.clicked.connect(self._delete_selected)
        editor.addWidget(self._btn_del)

        editor.addWidget(QLabel("Type:"))
        self._inline_type = QComboBox()
        for t in ("button_lap", "button_gate", "rh_lap", "rh_gate"):
            self._inline_type.addItem(t)
        self._inline_type.setEnabled(False)
        editor.addWidget(self._inline_type)

        editor.addWidget(QLabel("ts (s):"))
        self._inline_ts = QDoubleSpinBox()
        self._inline_ts.setDecimals(3)
        self._inline_ts.setRange(0.0, 36_000.0)
        self._inline_ts.setSingleStep(0.05)
        self._inline_ts.setEnabled(False)
        editor.addWidget(self._inline_ts)

        editor.addWidget(QLabel("Gate:"))
        self._inline_gate = QComboBox()
        self._inline_gate.setEnabled(False)
        editor.addWidget(self._inline_gate)

        self._btn_apply = QPushButton("Apply (Ctrl+Enter)")
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply_inline)
        editor.addWidget(self._btn_apply)

        editor.addStretch(1)
        outer.addLayout(editor)

        # ── Hotkeys ─────────────────────────────────────────────────────
        for key, slot in [
            ("Ctrl+Return", self._apply_inline),
            ("Ctrl+Enter",  self._apply_inline),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)

        for key, delta in [
            (Qt.Key.Key_Left,  -1),
            (Qt.Key.Key_Right, +1),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda d=delta: self.nudge_frame_requested.emit(d))
        for key, delta in [
            ("Shift+Left",  -100),
            ("Shift+Right", +100),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda d=delta: self.nudge_ms_requested.emit(d))

    # ── public API ─────────────────────────────────────────────────────────

    def set_session_time_range(self, t0: float, t1: float) -> None:
        self._t0 = t0
        self._t1 = t1

    def set_events(self, events: list[dict]) -> None:
        self.event_strip.set_events(events, self._t0, self._t1)
        if self._selected_event is not None:
            seq = self._selected_event.get("seq")
            for ev in events:
                if ev.get("seq") == seq:
                    self._selected_event = ev
                    self._populate_inline_editor(ev)
                    break

    def set_gates(self, gates: list[dict]) -> None:
        self._gates = list(gates)
        self._inline_gate.blockSignals(True)
        self._inline_gate.clear()
        self._inline_gate.addItem("(none)", userData=-1)
        for g in self._gates:
            self._inline_gate.addItem(f"{g['id']} · {g.get('name', '')}", userData=int(g["id"]))
        self._inline_gate.blockSignals(False)

    def set_snap_sources(self, *, tl_ts=None, rc_ts=None) -> None:
        self.event_strip.set_snap_sources(tl_ts=tl_ts, rc_ts=rc_ts)

    def set_playing(self, playing: bool) -> None:
        self._btn_play.setChecked(playing)
        self._btn_play.setText("⏸  Pause" if playing else "▶  Play")

    def set_lap(self, lap: int, total: int) -> None:
        self._lbl_lap.setText(f"Круг {lap}/{total}" if total else "")

    def update_progress(self, current_s: float, total_s: float) -> None:
        if total_s > 0:
            self._slider.setValue(int(current_s / total_s * 10000))
        self._lbl_time.setText(f"{_fmt_ms(current_s)} / {_fmt(total_s)}")
        self.event_strip.set_playhead(self._t0 + current_s)

    def selected_event(self) -> dict | None:
        return self._selected_event

    # ── private slots ──────────────────────────────────────────────────────

    def _on_marker_clicked(self, ev: dict) -> None:
        self.event_seek_requested.emit(ev["ts_wall"])

    def _on_marker_selected(self, ev) -> None:
        self._selected_event = ev if isinstance(ev, dict) else None
        editable = self._selected_event is not None
        self._btn_del.setEnabled(editable)
        for w in (self._inline_type, self._inline_ts, self._inline_gate, self._btn_apply):
            w.setEnabled(editable)
        if editable:
            self._populate_inline_editor(self._selected_event)

    def _populate_inline_editor(self, ev: dict) -> None:
        self._inline_type.blockSignals(True)
        idx = self._inline_type.findText(ev.get("event_type", ""))
        if idx >= 0:
            self._inline_type.setCurrentIndex(idx)
        self._inline_type.blockSignals(False)
        local = float(ev.get("ts_wall", 0.0)) - self._t0
        self._inline_ts.blockSignals(True)
        self._inline_ts.setValue(max(0.0, local))
        self._inline_ts.blockSignals(False)
        gate_id = int(ev.get("gate_id", -1) or -1)
        idx = self._inline_gate.findData(gate_id)
        if idx < 0:
            idx = 0
        self._inline_gate.blockSignals(True)
        self._inline_gate.setCurrentIndex(idx)
        self._inline_gate.blockSignals(False)

    def _apply_inline(self) -> None:
        if not self._selected_event:
            return
        seq = int(self._selected_event.get("seq", -1))
        et = self._inline_type.currentText()
        ts_local = float(self._inline_ts.value())
        ts_abs = self._t0 + ts_local
        gid = int(self._inline_gate.currentData() if self._inline_gate.currentIndex() >= 0 else -1)
        self.event_inline_apply.emit(seq, et, ts_abs, gid)

    def _add_lap(self) -> None:
        self.event_add_requested.emit("button_lap", -1)

    def _add_gate(self) -> None:
        from dct.gui.widgets.gate_pick_dialog import GatePickDialog
        dlg = GatePickDialog(self._gates, self)
        if dlg.exec():
            self.event_add_requested.emit("button_gate", dlg.selected_gate_id())

    def _add_sf(self) -> None:
        self.event_add_requested.emit("rh_lap", -1)

    def _delete_selected(self) -> None:
        if not self._selected_event:
            return
        ev, self._selected_event = self._selected_event, None
        self._btn_del.setEnabled(False)
        self.event_strip.clear_selection()
        self.event_delete_requested.emit(ev)


class BottomStrip(QFrame):
    # Record signals
    start_clicked = pyqtSignal()
    stop_clicked  = pyqtSignal()
    lap_clicked   = pyqtSignal()
    gate_clicked  = pyqtSignal()
    sf_clicked    = pyqtSignal()

    # Replay signals
    play_pause              = pyqtSignal()
    seek_fraction           = pyqtSignal(float)
    speed_changed           = pyqtSignal(float)
    event_add_requested     = pyqtSignal(str, int)
    event_delete_requested  = pyqtSignal(dict)
    event_seek_requested    = pyqtSignal(float)
    event_drag_started      = pyqtSignal(dict)
    event_drag_ended        = pyqtSignal(int, float)
    event_inline_apply      = pyqtSignal(int, str, float, int)
    nudge_frame_requested   = pyqtSignal(int)
    nudge_ms_requested      = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "bottombar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(64)

        self._stack = QStackedWidget(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._stack)

        self._record = _RecordControls()
        self._replay = _ReplayControls()
        self._stack.addWidget(self._record)
        self._stack.addWidget(self._replay)

        self._record.start_clicked.connect(self.start_clicked)
        self._record.stop_clicked.connect(self.stop_clicked)
        self._record.lap_clicked.connect(self.lap_clicked)
        self._record.gate_clicked.connect(self.gate_clicked)
        self._record.sf_clicked.connect(self.sf_clicked)

        self._replay.play_pause_clicked.connect(self.play_pause)
        self._replay.seek_fraction.connect(self.seek_fraction)
        self._replay.speed_changed.connect(self.speed_changed)
        self._replay.event_add_requested.connect(self.event_add_requested)
        self._replay.event_delete_requested.connect(self.event_delete_requested)
        self._replay.event_seek_requested.connect(self.event_seek_requested)
        self._replay.event_drag_started.connect(self.event_drag_started)
        self._replay.event_drag_ended.connect(self.event_drag_ended)
        self._replay.event_inline_apply.connect(self.event_inline_apply)
        self._replay.nudge_frame_requested.connect(self.nudge_frame_requested)
        self._replay.nudge_ms_requested.connect(self.nudge_ms_requested)

        # Hotkeys (record mode only, plus replay 8/9/0)
        self._app_shortcuts = []
        for key, slot in [
            ("6", self._safe_start),
            ("7", self._safe_stop),
            ("8", self._key_8),
            ("9", self._key_9),
            ("0", self._key_0),
            (Qt.Key.Key_Delete, self._key_del),
            (Qt.Key.Key_Backspace, self._key_del),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._app_shortcuts.append(sc)

        self._hotkeys = GlobalHotkeyManager()
        self._recording = False
        for key, slot in [
            ("6", self._safe_start),
            ("7", self._safe_stop),
            ("8", self._safe_lap),
            ("9", self._safe_gate),
            ("0", self._safe_sf),
        ]:
            self._hotkeys.register(key, slot)

        self.set_record_mode()

    # ── public API ─────────────────────────────────────────────────────────

    def set_record_mode(self) -> None:
        self._stack.setCurrentIndex(0)

    def set_replay_mode(self) -> None:
        self._stack.setCurrentIndex(1)

    def set_recording(self, active: bool) -> None:
        self._recording = active
        self._record._set_recording(active)

    def replay_controls(self) -> _ReplayControls:
        return self._replay

    def cleanup(self) -> None:
        self._hotkeys.unregister_all()

    # ── safe (only on record mode + recording) ─────────────────────────────

    def _safe_start(self) -> None:
        if self._stack.currentIndex() == 0 and not self._recording:
            self.start_clicked.emit()

    def _safe_stop(self) -> None:
        if self._recording:
            self.stop_clicked.emit()

    def _safe_lap(self) -> None:
        if self._stack.currentIndex() == 0 and self._recording:
            self.lap_clicked.emit()
        elif self._stack.currentIndex() == 1:
            self.event_add_requested.emit("button_lap", -1)

    def _safe_gate(self) -> None:
        if self._stack.currentIndex() == 0 and self._recording:
            self.gate_clicked.emit()
        elif self._stack.currentIndex() == 1:
            self._replay._add_gate()

    def _safe_sf(self) -> None:
        if self._stack.currentIndex() == 0 and self._recording:
            self.sf_clicked.emit()
        elif self._stack.currentIndex() == 1:
            self.event_add_requested.emit("rh_lap", -1)

    # ── application shortcut wrappers ─────────────────────────────────────

    def _key_8(self) -> None:
        self._safe_lap()

    def _key_9(self) -> None:
        self._safe_gate()

    def _key_0(self) -> None:
        self._safe_sf()

    @pyqtSlot()
    def _key_del(self) -> None:
        if self._stack.currentIndex() == 1:
            sel = self._replay.selected_event()
            if sel:
                self._replay._delete_selected()

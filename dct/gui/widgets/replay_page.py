"""Sidebar Replay page — folder picker, sessions list, localizer panel, hotkey hints."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from dct.gui import ui_settings
from dct.localization import reference_builder as refbuild


def _session_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])


class ReplayPage(QWidget):
    session_selected         = pyqtSignal(str)
    localizer_settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_track_id: str = ""
        self._invert_lf: dict = {}

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

        # ── Localizer ─────────────────────────────────────────────────────
        loc_box = QGroupBox("Localizer")
        loc_lay = QVBoxLayout(loc_box)
        loc_lay.setSpacing(4)

        self._chk_localizer = QCheckBox("Enable on replay")
        self._chk_localizer.setToolTip(
            "Запускать локализатор при воспроизведении выбранной сессии.",
        )
        s = ui_settings.load()
        self._chk_localizer.setChecked(bool(s.get("replay_localizer_enabled", True)))
        self._chk_localizer.stateChanged.connect(self._on_loc_state_changed)
        loc_lay.addWidget(self._chk_localizer)

        prof_row = QHBoxLayout()
        prof_row.setSpacing(4)
        prof_row.addWidget(QLabel("Profile"))
        self._combo_loc_profile = QComboBox()
        self._combo_loc_profile.setMinimumWidth(80)
        self._combo_loc_profile.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._combo_loc_profile.currentIndexChanged.connect(self._on_loc_profile_changed)
        prof_row.addWidget(self._combo_loc_profile, stretch=1)
        self._btn_loc_build = QPushButton("Build…")
        self._btn_loc_build.setToolTip("Собрать новый эталон из сессии или папки сессий")
        self._btn_loc_build.clicked.connect(self._on_build_clicked)
        prof_row.addWidget(self._btn_loc_build)
        loc_lay.addLayout(prof_row)

        self._lbl_loc_meta = QLabel("(нет файла)")
        self._lbl_loc_meta.setProperty("role", "dim")
        self._lbl_loc_meta.setWordWrap(True)
        loc_lay.addWidget(self._lbl_loc_meta)

        viz = QFrame()
        viz_lay = QVBoxLayout(viz)
        viz_lay.setContentsMargins(0, 0, 0, 0)
        viz_lay.setSpacing(2)
        self._chk_show_path  = QCheckBox("Reference path (dashed)")
        self._chk_show_arrow = QCheckBox("Estimate arrow")
        self._chk_show_trail = QCheckBox("Estimate trail")
        for chk in (self._chk_show_path, self._chk_show_arrow, self._chk_show_trail):
            chk.setChecked(True)
            chk.stateChanged.connect(self._on_show_changed)
            viz_lay.addWidget(chk)
        loc_lay.addWidget(viz)

        action_row = QHBoxLayout()
        self._btn_loc_reset = QPushButton("Reset filter")
        self._btn_loc_reset.setToolTip("Перезапустить частичный фильтр (сбросить состояние)")
        action_row.addWidget(self._btn_loc_reset)
        action_row.addStretch()
        loc_lay.addLayout(action_row)
        root.addWidget(loc_box)

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

    def set_localizer_track(self, track_id: str) -> None:
        """Populate the profile combo for *track_id*.  Call after a session is selected."""
        self._current_track_id = track_id or ""
        self._combo_loc_profile.blockSignals(True)
        self._combo_loc_profile.clear()
        if track_id:
            for info in refbuild.find_for_track(track_id):
                self._combo_loc_profile.addItem(
                    info.profile,
                    userData={
                        "track_id": track_id,
                        "profile":  info.profile,
                        "npz_path": str(info.npz_path),
                        "meta":     info.meta,
                    },
                )
            # Restore last-used profile for this track
            preferred = ui_settings.load().get("localizer", {}).get(track_id, {}).get("profile")
            if preferred:
                idx = self._combo_loc_profile.findText(preferred)
                if idx >= 0:
                    self._combo_loc_profile.setCurrentIndex(idx)
        self._combo_loc_profile.blockSignals(False)
        self._update_loc_meta_label()

    def localizer_enabled(self) -> bool:
        return self._chk_localizer.isChecked()

    def current_localizer_path(self) -> Path | None:
        """Return the selected profile .npz path, or None if nothing selected."""
        info = self._combo_loc_profile.currentData()
        if not info:
            return None
        p = Path(info["npz_path"])
        return p if p.is_file() else None

    def localizer_show_state(self) -> dict[str, bool]:
        return {
            "path":  self._chk_show_path.isChecked(),
            "arrow": self._chk_show_arrow.isChecked(),
            "trail": self._chk_show_trail.isChecked(),
        }

    def reset_filter_button(self) -> QPushButton:
        return self._btn_loc_reset

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

    def set_invert_lf(self, invert_lf: dict) -> None:
        """Called by main_window whenever the LF invert-checkbox state changes."""
        self._invert_lf = dict(invert_lf)

    def _on_build_clicked(self) -> None:
        if not self._current_track_id:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Reference builder",
                "Нет track_id для текущей сессии.\n"
                "Убедитесь, что в папке сессии есть track.json.",
            )
            return
        from dct.gui.widgets.reference_build_dialog import ReferenceBuildDialog
        dlg = ReferenceBuildDialog(
            self._current_track_id, invert_lf=self._invert_lf, parent=self,
        )
        if dlg.exec():
            # Reload profiles and notify the main window to reinitialize
            self.set_localizer_track(self._current_track_id)
            self.localizer_settings_changed.emit()

    def _on_loc_state_changed(self) -> None:
        ui_settings.update("replay_localizer_enabled", self._chk_localizer.isChecked())
        self.localizer_settings_changed.emit()

    def _on_loc_profile_changed(self) -> None:
        track_id = self._current_track_id
        info = self._combo_loc_profile.currentData()
        if track_id and info:
            d = ui_settings.load()
            d.setdefault("localizer", {})[track_id] = {"profile": info["profile"]}
            ui_settings.save(d)
        self._update_loc_meta_label()
        self.localizer_settings_changed.emit()

    def _on_show_changed(self) -> None:
        self.localizer_settings_changed.emit()

    def _update_loc_meta_label(self) -> None:
        info = self._combo_loc_profile.currentData()
        if not info:
            self._lbl_loc_meta.setText("(нет файла)")
            return
        meta = info.get("meta") or {}
        bits = [Path(info["npz_path"]).name]
        if meta.get("built_at"):
            bits.append(str(meta["built_at"])[:10])
        if meta.get("source"):
            bits.append(f"src={Path(str(meta['source'])).name}")
        if meta.get("lap_index") is not None:
            bits.append(f"lap={meta['lap_index']}")
        self._lbl_loc_meta.setText("  ·  ".join(bits))

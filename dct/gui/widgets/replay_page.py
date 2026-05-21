"""Sidebar Replay page — folder picker, sessions list, localizer panel, hotkey hints."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from dct.gui import ui_settings
from dct.gui.widgets.mavlink_panel import (
    apply_mavlink_panel_layout,
    build_mavlink_panel,
    read_mavlink_settings,
)
from dct.localization import reference_builder as refbuild


def _session_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])


class ReplayPage(QWidget):
    session_selected         = pyqtSignal(str)
    localizer_settings_changed = pyqtSignal()
    mavlink_settings_changed = pyqtSignal()

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

        sigma_row = QHBoxLayout()
        sigma_row.setSpacing(4)
        sigma_lbl = QLabel("Obs σ:")
        sigma_lbl.setToolTip(
            "Observation noise sigma for the Particle Filter.\n"
            "Lower = tighter matching (faster lock, more sensitive to mismatch).\n"
            "Higher = looser (more robust, but slower/weaker convergence).\n"
            "Recommended: 1.0–1.5 for same-pilot reference; 2.0+ for cross-pilot."
        )
        sigma_row.addWidget(sigma_lbl)
        self._spn_obs_sigma = QDoubleSpinBox()
        self._spn_obs_sigma.setRange(0.5, 10.0)
        self._spn_obs_sigma.setSingleStep(0.5)
        self._spn_obs_sigma.setDecimals(1)
        saved_sigma = float(ui_settings.load().get("localizer_obs_sigma", 1.5))
        self._spn_obs_sigma.setValue(saved_sigma)
        self._spn_obs_sigma.setToolTip(sigma_lbl.toolTip())
        self._spn_obs_sigma.valueChanged.connect(self._on_obs_sigma_changed)
        sigma_row.addWidget(self._spn_obs_sigma)
        sigma_row.addStretch()
        loc_lay.addLayout(sigma_row)

        # ── Per-channel weights ──────────────────────────────────────────
        _CH_DEFAULTS = [1.0, 1.0, 1.0, 1.0]
        _CH_LABELS    = ["Thr", "Yaw", "Pit", "Roll"]
        _saved_ws: list = ui_settings.load().get("localizer_ch_weights", _CH_DEFAULTS)
        if len(_saved_ws) != 4:
            _saved_ws = _CH_DEFAULTS

        _W_TOOLTIP = (
            "Per-channel importance weights for the Particle Filter distance:\n"
            "  Thr  – throttle stick (drone-dependent; set 0 for cross-drone)\n"
            "  Yaw  – yaw rate (deg/s after rate curve)\n"
            "  Pit  – pitch rate\n"
            "  Roll – roll rate (most lap-consistent; raise for better lock)\n"
            "Weights are applied as: d² = Σ wᵢ·(ref_i − obs_i)²\n"
            "Then normalised by the number of active channels."
        )
        weights_header = QLabel("Weights:")
        weights_header.setToolTip(_W_TOOLTIP)
        loc_lay.addWidget(weights_header)

        weights_grid = QHBoxLayout()
        weights_grid.setSpacing(4)
        self._spn_ch_weights: list[QDoubleSpinBox] = []

        # Two columns: label | spinbox | label | spinbox
        col_a = QVBoxLayout()
        col_a.setSpacing(2)
        col_b = QVBoxLayout()
        col_b.setSpacing(2)

        for i, (lbl, val) in enumerate(zip(_CH_LABELS, _saved_ws)):
            row = QHBoxLayout()
            row.setSpacing(3)
            ch_lbl = QLabel(f"{lbl}:")
            ch_lbl.setFixedWidth(28)
            ch_lbl.setToolTip(_W_TOOLTIP)
            row.addWidget(ch_lbl)
            spn = QDoubleSpinBox()
            spn.setRange(0.0, 5.0)
            spn.setSingleStep(0.5)
            spn.setDecimals(1)
            spn.setValue(float(val))
            spn.setFixedWidth(52)
            spn.setToolTip(_W_TOOLTIP)
            spn.valueChanged.connect(self._on_ch_weights_changed)
            row.addWidget(spn)
            row.addStretch()
            self._spn_ch_weights.append(spn)
            if i < 2:
                col_a.addLayout(row)
            else:
                col_b.addLayout(row)

        weights_grid.addLayout(col_a)
        weights_grid.addLayout(col_b)
        weights_grid.addStretch()
        loc_lay.addLayout(weights_grid)

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

        # ── MAVLink telemetry ─────────────────────────────────────────────
        self._mavlink = build_mavlink_panel(
            ui_settings.load().get("mavlink", {}),
            bounds_default="Bounds: no replay track selected",
            on_changed=self._on_mavlink_changed,
        )
        root.addWidget(self._mavlink.box, 0, Qt.AlignmentFlag.AlignTop)
        apply_mavlink_panel_layout(self._mavlink)

        root.addStretch(1)
        self.reload_sessions()

    # ── public API ─────────────────────────────────────────────────────────

    def refresh_layout(self) -> None:
        """Re-apply Mavlink sizing after tab switch (same as Setup)."""
        inner = max(160, self.width() - 16)
        apply_mavlink_panel_layout(self._mavlink, inner)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.refresh_layout)

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

    def current_obs_sigma(self) -> float:
        """Return the currently selected obs_sigma for the Particle Filter."""
        return float(self._spn_obs_sigma.value())

    def channel_weights(self) -> list[float]:
        """Return current per-channel weights [Thr, Yaw, Pit, Roll]."""
        return [float(spn.value()) for spn in self._spn_ch_weights]

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

    def mavlink_settings(self) -> dict:
        return read_mavlink_settings(self._mavlink)

    def set_mavlink_track_bounds(self, bounds: dict | None) -> None:
        if not isinstance(bounds, dict):
            self._mavlink.lbl_bounds.setText("Bounds: no replay track selected")
            return
        try:
            origin_x = float(bounds.get("origin_x", 0.0))
            origin_z = float(bounds.get("origin_z", 0.0))
            size_x = float(bounds["x"])
            size_z = float(bounds.get("z", bounds["y"]))
        except (KeyError, TypeError, ValueError):
            self._mavlink.lbl_bounds.setText("Bounds: invalid track.json bounds")
            return
        self._mavlink.lbl_bounds.setText(
            f"Local anchors: 0.0=({origin_x:.2f},{origin_z:.2f}), "
            f"X.0=({origin_x + size_x:.2f},{origin_z:.2f}), "
            f"0.Z=({origin_x:.2f},{origin_z + size_z:.2f})",
        )

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

    def _on_obs_sigma_changed(self, value: float) -> None:
        ui_settings.update("localizer_obs_sigma", value)
        self.localizer_settings_changed.emit()

    def _on_ch_weights_changed(self) -> None:
        ui_settings.update("localizer_ch_weights", self.channel_weights())
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

    def _on_mavlink_changed(self) -> None:
        ui_settings.update("mavlink", self.mavlink_settings())
        self.mavlink_settings_changed.emit()

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

"""Sidebar Setup page — vertical session configuration form.

Owns combos for pilot/drone/rate/camera/track, data-source selector,
COM port, video source and the Localizer profile picker.

Emits ``cfg_changed`` whenever any selection mutates so that the top bar
summary stays in sync with the form. ``build_cfg()`` is the single entry
point used by the BottomStrip's START button.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from dct.gui import theme, ui_settings
from dct.localization import reference_builder as refbuild
from dct.rc_receiver import scan_serial_ports
from dct.screen_recorder import scan_video_devices
from dct import tracks_io

_PROFILES_DIR = Path("profiles")

_SRC_LIFTOFF = "liftoff"
_SRC_RC      = "rc"
_SRC_BOTH    = "both"


def _scan_dir(subdir: str) -> list[dict]:
    d = _PROFILES_DIR / subdir
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def _pilots() -> list[dict]:
    p = _PROFILES_DIR / "pilots.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


class SetupPage(QWidget):
    cfg_changed              = pyqtSignal()
    video_source_changed     = pyqtSignal(dict)
    localizer_settings_changed = pyqtSignal()

    _video_devices_ready = pyqtSignal(list)
    _com_ports_ready     = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._recording = False
        self._saved_com_port = ""
        self._saved_video_index: Any = None

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Session config ─────────────────────────────────────────────────
        cfg_box = QGroupBox("Session config")
        cfg_lay = QVBoxLayout(cfg_box)
        cfg_lay.setSpacing(4)
        self._combo_pilot  = self._make_combo("Pilot",  cfg_lay)
        self._combo_drone  = self._make_combo("Drone",  cfg_lay)
        self._combo_rate   = self._make_combo("Rate",   cfg_lay)
        self._combo_camera = self._make_combo("Camera", cfg_lay)
        self._combo_track  = self._make_combo("Track",  cfg_lay)
        root.addWidget(cfg_box)

        # ── Data sources ───────────────────────────────────────────────────
        ds_box = QGroupBox("Data sources")
        ds_lay = QVBoxLayout(ds_box)
        ds_lay.setSpacing(4)

        self._combo_ds_mode = QComboBox()
        self._combo_ds_mode.addItem("Liftoff only", userData=_SRC_LIFTOFF)
        self._combo_ds_mode.addItem("RC only",      userData=_SRC_RC)
        self._combo_ds_mode.addItem("Liftoff + RC", userData=_SRC_BOTH)
        self._combo_ds_mode.currentIndexChanged.connect(self._on_ds_mode_changed)
        ds_lay.addWidget(self._combo_ds_mode)

        rc_row = QHBoxLayout()
        rc_row.setSpacing(4)
        self._combo_com = QComboBox()
        self._combo_com.setMinimumWidth(110)
        self._combo_com.setToolTip("COM-порт RC-приёмника")
        self._btn_refresh_com = QPushButton("↻")
        self._btn_refresh_com.setProperty("role", "icon")
        self._btn_refresh_com.setFixedWidth(28)
        self._btn_refresh_com.setToolTip("Обновить список COM-портов")
        self._btn_refresh_com.clicked.connect(self._refresh_com_ports)
        self._lbl_rc_status = QLabel("●")
        self._lbl_rc_status.setStyleSheet(
            f"color: {theme.WARN}; font-size: {theme.FONT_HEAD}px;",
        )
        rc_row.addWidget(self._combo_com, stretch=1)
        rc_row.addWidget(self._btn_refresh_com)
        rc_row.addWidget(self._lbl_rc_status)
        ds_lay.addLayout(rc_row)
        root.addWidget(ds_box)

        # ── Video source ───────────────────────────────────────────────────
        vid_box = QGroupBox("Video source")
        vid_lay = QVBoxLayout(vid_box)
        vid_lay.setSpacing(4)

        vrow = QHBoxLayout()
        vrow.setSpacing(4)
        self._combo_source = QComboBox()
        self._combo_source.setMinimumWidth(110)
        self._btn_refresh_src = QPushButton("↻")
        self._btn_refresh_src.setProperty("role", "icon")
        self._btn_refresh_src.setFixedWidth(28)
        self._btn_refresh_src.setToolTip("Обновить список устройств захвата")
        self._btn_refresh_src.clicked.connect(self._refresh_video_sources)
        self._combo_source.currentIndexChanged.connect(self._on_video_source_changed)
        vrow.addWidget(self._combo_source, stretch=1)
        vrow.addWidget(self._btn_refresh_src)
        vid_lay.addLayout(vrow)

        save_row = QHBoxLayout()
        save_row.setSpacing(4)
        self._lbl_sessions_dir = QLabel(
            ui_settings.load().get("sessions_dir", "sessions"),
        )
        self._lbl_sessions_dir.setProperty("role", "dim")
        self._lbl_sessions_dir.setToolTip("Папка для сохранения сессий")
        self._lbl_sessions_dir.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        self._btn_browse_dir = QPushButton("📁")
        self._btn_browse_dir.setProperty("role", "icon")
        self._btn_browse_dir.setFixedWidth(28)
        self._btn_browse_dir.setToolTip("Выбрать папку сессий")
        self._btn_browse_dir.clicked.connect(self._browse_sessions_dir)
        save_row.addWidget(self._lbl_sessions_dir, stretch=1)
        save_row.addWidget(self._btn_browse_dir)
        vid_lay.addLayout(save_row)
        root.addWidget(vid_box)

        # ── Localizer ──────────────────────────────────────────────────────
        loc_box = QGroupBox("Localizer")
        loc_lay = QVBoxLayout(loc_box)
        loc_lay.setSpacing(4)

        self._chk_localizer = QCheckBox("Enable on record")
        self._chk_localizer.setToolTip(
            "Локализатор по эталону: Liftoff / Both — по стикам симулятора; "
            "RC only — по нормализованным каналам приёмника (как на графиках).",
        )
        self._chk_localizer.stateChanged.connect(self._on_loc_state_changed)
        loc_lay.addWidget(self._chk_localizer)

        prof_row = QHBoxLayout()
        prof_row.setSpacing(4)
        prof_row.addWidget(QLabel("Profile"))
        self._combo_loc_profile = QComboBox()
        self._combo_loc_profile.setMinimumWidth(110)
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

        root.addStretch(1)

        # ── Connections ────────────────────────────────────────────────────
        self._video_devices_ready.connect(self._apply_video_devices)
        self._com_ports_ready.connect(self._apply_com_ports)

        for combo in (self._combo_pilot, self._combo_drone, self._combo_rate,
                      self._combo_camera, self._combo_track):
            combo.currentIndexChanged.connect(self._save_settings)
            combo.currentIndexChanged.connect(self.cfg_changed)

        self._combo_track.currentIndexChanged.connect(self._reload_loc_profiles)
        self._combo_ds_mode.currentIndexChanged.connect(self._save_settings)
        self._combo_com.currentIndexChanged.connect(self._save_settings)

        # Initial population
        self._combo_source.addItem(
            "Liftoff (screen capture)", userData={"type": "screen"},
        )
        self.reload_profiles()
        self._on_ds_mode_changed()
        QTimer.singleShot(200, self._refresh_com_ports)

    # ── public API ─────────────────────────────────────────────────────────

    def reload_profiles(self) -> None:
        self._loading = True
        try:
            self._fill_combo(self._combo_pilot,  _pilots(),            "nickname")
            self._fill_combo(self._combo_drone,  _scan_dir("drones"),  "name")
            self._fill_combo(self._combo_rate,   _scan_dir("rates"),   "name")
            self._fill_combo(self._combo_camera, _scan_dir("cameras"), "name")
            self._fill_track_combo()
        finally:
            self._loading = False
        self._restore_settings()
        self._reload_loc_profiles()

    def get_sessions_dir(self) -> str:
        return ui_settings.load().get("sessions_dir", "sessions")

    def current_video_source(self) -> dict:
        return self._combo_source.currentData() or {"type": "screen"}

    def current_track_id(self) -> str | None:
        d = self._combo_track.currentData()
        if not d:
            return None
        return str(d.get("track_id") or d.get("id") or "")

    def current_localizer_profile(self) -> tuple[str | None, Path | None]:
        if not self._chk_localizer.isChecked():
            return None, None
        info = self._combo_loc_profile.currentData()
        if not info:
            return None, None
        return str(info.get("profile")), Path(info["npz_path"])

    def localizer_show_state(self) -> dict[str, bool]:
        return {
            "path":  self._chk_show_path.isChecked(),
            "arrow": self._chk_show_arrow.isChecked(),
            "trail": self._chk_show_trail.isChecked(),
        }

    def reset_filter_button(self) -> QPushButton:
        return self._btn_loc_reset

    def set_recording(self, active: bool) -> None:
        self._recording = active
        for w in (self._combo_pilot, self._combo_drone, self._combo_rate,
                  self._combo_camera, self._combo_track,
                  self._combo_ds_mode, self._combo_com, self._btn_refresh_com,
                  self._combo_source, self._btn_refresh_src,
                  self._chk_localizer, self._combo_loc_profile, self._btn_loc_build):
            w.setEnabled(not active)

    def set_rc_status(self, online: bool) -> None:
        if online:
            self._lbl_rc_status.setStyleSheet(
                f"color: {theme.OK}; font-size: {theme.FONT_HEAD}px;",
            )
            self._lbl_rc_status.setToolTip("RC receiver: online")
        else:
            self._lbl_rc_status.setStyleSheet(
                f"color: {theme.ERR}; font-size: {theme.FONT_HEAD}px;",
            )
            self._lbl_rc_status.setToolTip("RC receiver: offline")

    def summary_text(self) -> str:
        parts = []
        for combo in (self._combo_pilot, self._combo_drone, self._combo_track):
            t = combo.currentText().strip()
            if t:
                parts.append(t)
        ds_text = self._combo_ds_mode.currentText() or "—"
        parts.append(ds_text)
        return " · ".join(parts) if parts else "—"

    def build_cfg(self) -> dict | None:
        """Validate the form and return a session-config dict (or ``None``)."""
        pilot_data  = self._combo_pilot.currentData()
        drone_data  = self._combo_drone.currentData()
        track_data  = self._combo_track.currentData()
        rate_data   = self._combo_rate.currentData()
        camera_data = self._combo_camera.currentData()

        missing = []
        if not pilot_data: missing.append("Pilot")
        if not drone_data: missing.append("Drone")
        if not track_data: missing.append("Track")

        mode = self._combo_ds_mode.currentData() or _SRC_LIFTOFF
        if mode in (_SRC_RC, _SRC_BOTH) and not self._combo_com.currentText():
            missing.append("RC COM port")

        if missing:
            QMessageBox.warning(
                self, "Session config incomplete",
                "Заполните перед стартом:\n  • " + "\n  • ".join(missing),
            )
            return None

        track_id = track_data.get("track_id") or track_data.get("id") or ""
        track_path = tracks_io.track_json_path(str(track_id))
        if not track_path.exists():
            entry = tracks_io.find_by_id(str(track_id)) or tracks_io.find_by_name(
                track_data.get("name", ""),
            )
            track_path = entry.path if entry is not None else track_path

        loc_profile, loc_path = self.current_localizer_profile()
        if self._chk_localizer.isChecked():
            if loc_path is None:
                QMessageBox.warning(
                    self, "Localizer",
                    "Включён локализатор, но для трассы не выбран профиль эталона.",
                )
                return None
            if not loc_path.is_file():
                QMessageBox.warning(
                    self, "Localizer", f"Файл эталона не найден:\n{loc_path}",
                )
                return None
        loc_enabled = (
            self._chk_localizer.isChecked()
            and loc_path is not None
            and loc_path.is_file()
            and mode in (_SRC_LIFTOFF, _SRC_RC, _SRC_BOTH)
        )
        return {
            "pilot":        pilot_data.get("nickname", pilot_data.get("id", "?")),
            "drone":        drone_data.get("id", drone_data["name"]),
            "track":        track_data.get("track_id") or track_data.get("id") or track_data["name"],
            "track_id":     str(track_id),
            "purpose":      "training",
            "track_path":   str(track_path) if track_path.exists() else None,
            "rate":         rate_data,
            "camera":       camera_data,
            "video_source": self._combo_source.currentData() or {"type": "screen"},
            "data_source":  mode,
            "rc_port":      self._combo_com.currentText() or None,
            "sessions_dir": self.get_sessions_dir(),
            "localizer_enabled":         loc_enabled,
            "localizer_profile":         loc_profile,
            "localizer_reference_path":  str(loc_path) if loc_enabled else None,
        }

    # ── private slots ──────────────────────────────────────────────────────

    def _on_ds_mode_changed(self) -> None:
        mode = self._combo_ds_mode.currentData()
        rc_needed = mode in (_SRC_RC, _SRC_BOTH)
        self._combo_com.setEnabled(rc_needed)
        self._btn_refresh_com.setEnabled(rc_needed)
        self._lbl_rc_status.setVisible(rc_needed)
        self.cfg_changed.emit()

    def _on_video_source_changed(self) -> None:
        data = self._combo_source.currentData()
        if data:
            self.video_source_changed.emit(data)
            d = ui_settings.load()
            if data.get("type") == "device":
                d["video_source_index"] = data.get("index")
            ui_settings.save(d)

    def _refresh_com_ports(self) -> None:
        threading.Thread(target=self._scan_com_bg, daemon=True).start()

    def _scan_com_bg(self) -> None:
        self._com_ports_ready.emit(scan_serial_ports())

    @pyqtSlot(list)
    def _apply_com_ports(self, ports: list) -> None:
        current = self._combo_com.currentText()
        self._combo_com.blockSignals(True)
        self._combo_com.clear()
        for port in ports:
            self._combo_com.addItem(port)
        target = current or self._saved_com_port
        idx = self._combo_com.findText(target)
        if idx >= 0:
            self._combo_com.setCurrentIndex(idx)
        self._combo_com.blockSignals(False)

    def _refresh_video_sources(self) -> None:
        threading.Thread(target=self._scan_video_bg, daemon=True).start()

    def _scan_video_bg(self) -> None:
        self._video_devices_ready.emit(scan_video_devices())

    @pyqtSlot(list)
    def _apply_video_devices(self, devices: list) -> None:
        current = self._combo_source.currentData()
        saved_index = self._saved_video_index
        self._combo_source.blockSignals(True)
        self._combo_source.clear()
        self._combo_source.addItem(
            "Liftoff (screen capture)", userData={"type": "screen"},
        )
        for idx, label in devices:
            self._combo_source.addItem(
                f"HDZero – {label}", userData={"type": "device", "index": idx},
            )
        restored = False
        if current:
            for i in range(self._combo_source.count()):
                if self._combo_source.itemData(i) == current:
                    self._combo_source.setCurrentIndex(i)
                    restored = True
                    break
        if not restored and saved_index is not None:
            for i in range(self._combo_source.count()):
                d = self._combo_source.itemData(i)
                if d and d.get("type") == "device" and d.get("index") == saved_index:
                    self._combo_source.setCurrentIndex(i)
                    break
        self._combo_source.blockSignals(False)
        d = self._combo_source.currentData()
        if d and d.get("type") == "device":
            ui_settings.update("video_source_index", d["index"])

    def _browse_sessions_dir(self) -> None:
        current = self.get_sessions_dir()
        folder = QFileDialog.getExistingDirectory(
            self, "Папка для сохранения сессий", current,
        )
        if folder:
            self._lbl_sessions_dir.setText(folder)
            ui_settings.update("sessions_dir", folder)

    # ── localizer ──────────────────────────────────────────────────────────

    def _migrate_legacy_loc_setting(self, track_id: str) -> None:
        """One-shot copy of the old ``localizer_reference_npz`` setting.

        Imports the old file as ``tracks/<track_id>/references/default.npz`` if
        the current track has no profiles yet.
        """
        if not track_id or refbuild.find_for_track(track_id):
            return
        legacy = str(ui_settings.load().get("localizer_reference_npz") or "").strip()
        if not legacy:
            return
        src = Path(legacy)
        if not src.is_file():
            return
        try:
            from dct.localization.online_localizer import Reference
            ref = Reference.load(src)
            refbuild.save_for_track(
                ref,
                track_id=track_id,
                profile="default",
                source=str(src),
            )
        except Exception:
            return

    def _reload_loc_profiles(self) -> None:
        track_id = self.current_track_id() or ""
        if track_id:
            self._migrate_legacy_loc_setting(track_id)
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
        # Persisted preferred profile
        preferred = ui_settings.load().get("localizer", {}).get(track_id, {}).get(
            "profile",
        ) if track_id else None
        if preferred:
            idx = self._combo_loc_profile.findText(preferred)
            if idx >= 0:
                self._combo_loc_profile.setCurrentIndex(idx)
        self._combo_loc_profile.blockSignals(False)
        self._update_loc_meta_label()
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
            src = Path(str(meta["source"])).name
            bits.append(f"src={src}")
        if meta.get("lap_index") is not None:
            bits.append(f"lap={meta['lap_index']}")
        self._lbl_loc_meta.setText("  ·  ".join(bits))

    def _on_loc_profile_changed(self) -> None:
        if self._loading:
            return
        track_id = self.current_track_id() or ""
        info = self._combo_loc_profile.currentData()
        d = ui_settings.load()
        loc = d.setdefault("localizer", {})
        if track_id and info:
            loc[track_id] = {"profile": info["profile"]}
            ui_settings.save(d)
        self._update_loc_meta_label()
        self.localizer_settings_changed.emit()

    def _on_loc_state_changed(self) -> None:
        if self._loading:
            return
        d = ui_settings.load()
        d["localizer_enabled"] = self._chk_localizer.isChecked()
        ui_settings.save(d)
        self.localizer_settings_changed.emit()

    def _on_show_changed(self) -> None:
        self.localizer_settings_changed.emit()

    def _on_build_clicked(self) -> None:
        track_id = self.current_track_id()
        if not track_id:
            QMessageBox.warning(
                self, "Reference builder", "Сначала выберите трассу.",
            )
            return
        from dct.gui.widgets.reference_build_dialog import ReferenceBuildDialog
        dlg = ReferenceBuildDialog(track_id, self)
        if dlg.exec():
            self._reload_loc_profiles()

    # ── persistence ────────────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        s = ui_settings.load()
        self._loading = True
        try:
            for combo, key in [
                (self._combo_pilot,  "pilot"),
                (self._combo_drone,  "drone"),
                (self._combo_rate,   "rate"),
                (self._combo_camera, "camera"),
                (self._combo_track,  "track"),
            ]:
                text = s.get(key, "")
                if text:
                    idx = combo.findText(text)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            mode = s.get("ds_mode")
            if mode:
                for i in range(self._combo_ds_mode.count()):
                    if self._combo_ds_mode.itemData(i) == mode:
                        self._combo_ds_mode.setCurrentIndex(i)
                        break
            self._chk_localizer.setChecked(bool(s.get("localizer_enabled", False)))
        finally:
            self._loading = False
        self._saved_com_port = s.get("com_port", "")
        self._saved_video_index = s.get("video_source_index")

    def _save_settings(self) -> None:
        if self._loading:
            return
        d = ui_settings.load()
        d.update({
            "pilot":    self._combo_pilot.currentText(),
            "drone":    self._combo_drone.currentText(),
            "rate":     self._combo_rate.currentText(),
            "camera":   self._combo_camera.currentText(),
            "track":    self._combo_track.currentText(),
            "ds_mode":  self._combo_ds_mode.currentData(),
            "com_port": self._combo_com.currentText(),
        })
        ui_settings.save(d)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_combo(placeholder: str, layout: QVBoxLayout) -> QComboBox:
        row = QHBoxLayout()
        row.setSpacing(4)
        lbl = QLabel(placeholder)
        lbl.setProperty("role", "dim")
        lbl.setMinimumWidth(58)
        row.addWidget(lbl)
        cb = QComboBox()
        cb.setPlaceholderText(placeholder)
        cb.setToolTip(placeholder)
        cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(cb, stretch=1)
        layout.addLayout(row)
        return cb

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[dict], label_key: str) -> None:
        combo.clear()
        for item in items:
            combo.addItem(item.get(label_key, "?"), userData=item)

    def _fill_track_combo(self) -> None:
        self._combo_track.clear()
        for entry in tracks_io.iter_tracks():
            data = dict(entry.data)
            data["track_id"] = entry.track_id
            data["name"] = entry.name
            self._combo_track.addItem(entry.name, userData=data)

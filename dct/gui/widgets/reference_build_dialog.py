"""Modal dialog for building / saving a Reference for a track."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from dct.localization import reference_builder as refbuild
from dct.localization.lap_loader import Lap
from dct.localization.online_localizer import Reference
from dct.rate_features import FEATURE_BETAFLIGHT_CLASSIC_V1

_log = logging.getLogger(__name__)


class _UserAbort(BaseException):
    """Signals cooperative cancel inside background workers.

    Inherits :class:`BaseException` so that ``except Exception`` blocks inside
    third-party loops (e.g. ``select_best_reference``) do not swallow it.
    """


class _LoadJob(QObject):
    finished = pyqtSignal(object, object, list)   # laps | None, track | None, summary | error
    failed   = pyqtSignal(str)

    def __init__(self, source: Path) -> None:
        super().__init__()
        self._src = source

    @pyqtSlot()
    def run(self) -> None:
        try:
            laps, track, summary = refbuild.list_laps(self._src)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(laps, track, summary)


class _AutoPickJob(QObject):
    progress  = pyqtSignal(int, int)
    finished  = pyqtSignal(int)
    failed    = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, laps: list[Lap], smooth_w: int) -> None:
        super().__init__()
        self._laps = laps
        self._smooth_w = smooth_w
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    @pyqtSlot()
    def run(self) -> None:
        def _cb(done: int, total: int) -> None:
            self.progress.emit(done, total)
            if self._cancel:
                raise _UserAbort()
        try:
            best = refbuild.auto_pick(
                self._laps,
                smooth_w=self._smooth_w,
                progress_cb=_cb,
            )
        except _UserAbort:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(int(best))


class ReferenceBuildDialog(QDialog):
    def __init__(self, track_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Build reference — {track_id}")
        self.setModal(True)
        self.resize(560, 520)

        self._track_id = track_id
        self._laps: list[Lap] = []
        self._summary: list[dict[str, Any]] = []
        self._best_idx: int | None = None

        self._loader_thread: QThread | None = None
        self._loader_worker: _LoadJob | None = None
        self._auto_thread:   QThread | None = None
        self._auto_worker:   _AutoPickJob | None = None

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        # ── Source ─────────────────────────────────────────────────────────
        src_row = QHBoxLayout()
        self._rb_single = QRadioButton("Single session")
        self._rb_folder = QRadioButton("Folder of sessions")
        self._rb_single.setChecked(True)
        self._rb_group = QButtonGroup(self)
        self._rb_group.addButton(self._rb_single, 0)
        self._rb_group.addButton(self._rb_folder, 1)
        src_row.addWidget(self._rb_single)
        src_row.addWidget(self._rb_folder)
        src_row.addStretch(1)
        outer.addLayout(src_row)

        path_row = QHBoxLayout()
        self._lbl_path = QLabel("(не выбрано)")
        self._lbl_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._lbl_path.setProperty("role", "dim")
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(self._lbl_path, stretch=1)
        path_row.addWidget(self._btn_browse)
        outer.addLayout(path_row)

        # ── Laps list ──────────────────────────────────────────────────────
        outer.addWidget(QLabel("Lap to use as reference:"))
        self._lst = QListWidget()
        self._lst.setMinimumHeight(160)
        outer.addWidget(self._lst, stretch=1)

        # ── Auto pick + smooth ─────────────────────────────────────────────
        opts_row = QHBoxLayout()
        self._btn_auto = QPushButton("Auto (LOO)")
        self._btn_auto.setToolTip("Подобрать лучший круг по leave-one-out NN-greedy")
        self._btn_auto.clicked.connect(self._on_auto)
        opts_row.addWidget(self._btn_auto)
        opts_row.addStretch(1)
        opts_row.addWidget(QLabel("smooth_w:"))
        self._spin_smooth = QSpinBox()
        self._spin_smooth.setRange(1, 25)
        self._spin_smooth.setSingleStep(2)
        self._spin_smooth.setValue(5)
        opts_row.addWidget(self._spin_smooth)

        opts_row.addWidget(QLabel("Profile:"))
        self._le_profile = QLineEdit("default")
        self._le_profile.setMaximumWidth(140)
        opts_row.addWidget(self._le_profile)
        outer.addLayout(opts_row)

        # ── Progress ───────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        self._btn_save = QPushButton("Build && Save")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)
        outer.addLayout(btn_row)

        # disable controls until source is selected
        self._set_busy(False)

    # ── source selection ───────────────────────────────────────────────────

    def _on_browse(self) -> None:
        if self._rb_single.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Папка сессии (содержит telemetry.parquet)")
        else:
            folder = QFileDialog.getExistingDirectory(self, "Папка сессий (несколько подпапок)")
        if not folder:
            return
        self._lbl_path.setText(folder)
        self._load_source(Path(folder))

    def _load_source(self, src: Path) -> None:
        self._stop_thread_blocking("loader", timeout_ms=5000)
        self._lst.clear()
        self._best_idx = None
        self._set_busy(True, message="Загрузка…", indeterminate=True)
        thread = QThread(self)
        worker = _LoadJob(src)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_load_done)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_loader_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._loader_thread = thread
        self._loader_worker = worker
        thread.start()

    @pyqtSlot()
    def _on_loader_thread_finished(self) -> None:
        self._loader_thread = None
        self._loader_worker = None

    @pyqtSlot(object, object, list)
    def _on_load_done(self, laps, _track, summary) -> None:
        self._laps = laps or []
        self._summary = summary or []
        self._refresh_list()
        self._set_busy(False)
        self._btn_save.setEnabled(bool(self._laps))

    @pyqtSlot(str)
    def _on_load_failed(self, msg: str) -> None:
        self._set_busy(False)
        QMessageBox.warning(self, "Reference builder", f"Не удалось загрузить:\n{msg}")

    def _refresh_list(self) -> None:
        self._lst.clear()
        for s in self._summary:
            txt = (
                f"Lap {s['index']:>3}  ·  {s['duration']:5.2f} s  "
                f"·  {s['length_m']:6.1f} m  ·  {s['frames']:5} frames"
            )
            if s.get("source"):
                txt += f"  ·  {s['source']}"
            item = QListWidgetItem(txt)
            self._lst.addItem(item)
        if self._lst.count() > 0:
            self._lst.setCurrentRow(0)

    # ── auto pick ──────────────────────────────────────────────────────────

    def _on_auto(self) -> None:
        if not self._laps:
            QMessageBox.information(self, "Auto", "Сначала загрузите кружочки.")
            return
        self._stop_thread_blocking("auto", timeout_ms=10000)
        self._set_busy(True, message="Auto LOO…", indeterminate=False)
        thread = QThread(self)
        worker = _AutoPickJob(self._laps, int(self._spin_smooth.value()))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_auto_progress)
        worker.finished.connect(self._on_auto_done)
        worker.failed.connect(self._on_auto_failed)
        worker.cancelled.connect(self._on_auto_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_auto_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._auto_thread = thread
        self._auto_worker = worker
        thread.start()

    @pyqtSlot(int, int)
    def _on_auto_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    @pyqtSlot(int)
    def _on_auto_done(self, best: int) -> None:
        self._set_busy(False)
        self._best_idx = best
        if 0 <= best < self._lst.count():
            self._lst.setCurrentRow(best)
            item = self._lst.item(best)
            if item:
                item.setText("★ " + item.text())

    @pyqtSlot(str)
    def _on_auto_failed(self, msg: str) -> None:
        self._set_busy(False)
        QMessageBox.warning(self, "Auto", msg)

    @pyqtSlot()
    def _on_auto_cancelled(self) -> None:
        self._set_busy(False)

    @pyqtSlot()
    def _on_auto_thread_finished(self) -> None:
        self._auto_thread = None
        self._auto_worker = None

    # ── save ───────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if not self._laps:
            return
        idx = self._lst.currentRow()
        if idx < 0:
            return
        lap = self._laps[idx]
        smooth_w = int(self._spin_smooth.value())
        profile = self._le_profile.text().strip() or "default"
        try:
            ref = refbuild.build(lap, smooth_w=smooth_w)
            path = refbuild.save_for_track(
                ref,
                track_id=self._track_id,
                profile=profile,
                source=str(Path(self._lbl_path.text()).resolve()),
                lap_index=int(lap.index),
                smooth_w=smooth_w,
            )
            path_legacy: Path | None = None
            if getattr(ref, "feature_kind", None) == FEATURE_BETAFLIGHT_CLASSIC_V1:
                ref_l = Reference.build(
                    t=lap.t.copy(),
                    sticks=lap.sticks.copy(),
                    pos=lap.pos.copy(),
                    smooth_w=smooth_w,
                )
                leg_prof = refbuild.legacy_sticks_profile_name(profile)
                path_legacy = refbuild.save_for_track(
                    ref_l,
                    track_id=self._track_id,
                    profile=leg_prof,
                    source=str(Path(self._lbl_path.text()).resolve()),
                    lap_index=int(lap.index),
                    smooth_w=smooth_w,
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Reference builder", f"Не удалось сохранить:\n{exc}")
            return
        msg = f"Готово:\n{path}"
        if path_legacy is not None:
            msg += f"\n\nLegacy sticks (сырой PF, тот же круг):\n{path_legacy}"
        QMessageBox.information(self, "Reference builder", msg)
        self.accept()

    # ── shutdown / thread cleanup ──────────────────────────────────────────

    def _stop_thread_blocking(self, which: str, *, timeout_ms: int = 5000) -> None:
        """Cooperatively stop a worker thread and wait for it.

        ``which`` is ``"loader"`` or ``"auto"``. Falls back to ``terminate()``
        if the thread refuses to finish in time — better than letting Qt
        destroy a still-running QThread on dialog close.
        """
        thread_attr = f"_{which}_thread"
        worker_attr = f"_{which}_worker"
        thread = getattr(self, thread_attr, None)
        worker = getattr(self, worker_attr, None)
        if thread is None:
            return
        try:
            if worker is not None and hasattr(worker, "request_cancel"):
                try:
                    worker.request_cancel()
                except RuntimeError:
                    pass
            try:
                if thread.isRunning():
                    thread.quit()
                    if not thread.wait(timeout_ms):
                        _log.warning(
                            "ReferenceBuildDialog: %s did not finish in %d ms, "
                            "terminating",
                            which, timeout_ms,
                        )
                        thread.terminate()
                        thread.wait(2000)
            except RuntimeError:
                pass
        finally:
            setattr(self, thread_attr, None)
            setattr(self, worker_attr, None)

    def _stop_all_threads(self) -> None:
        self._stop_thread_blocking("auto", timeout_ms=10000)
        self._stop_thread_blocking("loader", timeout_ms=5000)

    def done(self, result: int) -> None:  # noqa: D401 — Qt override
        self._stop_all_threads()
        super().done(result)

    def closeEvent(self, ev) -> None:  # noqa: D401 — Qt override
        self._stop_all_threads()
        super().closeEvent(ev)

    # ── helpers ────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool, *, message: str = "", indeterminate: bool = False) -> None:
        for w in (self._btn_browse, self._btn_save, self._btn_auto, self._lst,
                  self._spin_smooth, self._le_profile,
                  self._rb_single, self._rb_folder):
            w.setEnabled(not busy)
        if busy:
            self._progress.setVisible(True)
            if indeterminate:
                self._progress.setRange(0, 0)
            else:
                self._progress.setRange(0, 100)
                self._progress.setValue(0)
            self.setWindowTitle(f"Build reference — {self._track_id}  · {message}")
        else:
            self._progress.setVisible(False)
            self.setWindowTitle(f"Build reference — {self._track_id}")

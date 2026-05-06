"""QApplication entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QIcon
from PyQt6.QtWidgets import QApplication

from dct.gui.main_window import MainWindow
from dct.gui.theme import QSS
from dct.log import setup_logging
from dct.tracks_io import migrate_legacy_tracks

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def run() -> None:
    setup_logging()

    try:
        migrate_legacy_tracks()
    except Exception:
        pass

    # Windows: регистрируем AppUserModelID чтобы иконка отображалась в панели задач
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RedSheep.DCT.1")
        except Exception:
            pass

    # HiDPI: дробные коэффициенты Windows 125/150% передаём как есть
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
    )

    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication(sys.argv)
    app.setApplicationName("DCT")
    if _LOGO_PATH.exists():
        app.setWindowIcon(QIcon(str(_LOGO_PATH)))
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

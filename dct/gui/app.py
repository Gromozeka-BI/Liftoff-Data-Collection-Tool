"""QApplication entry point."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from dct.gui.main_window import MainWindow
from dct.gui.theme import QSS
from dct.log import setup_logging, get_logger


def run() -> None:
    setup_logging()
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication(sys.argv)
    app.setApplicationName("DCT")
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

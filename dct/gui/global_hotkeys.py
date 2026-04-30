"""Глобальные горячие клавиши через low-level keyboard hook (Windows).

Работает даже когда LiftOff (или любое другое окно) в фокусе.
keyboard.add_hotkey использует SetWindowsHookEx(WH_KEYBOARD_LL) — перехватывает
нажатия до того, как они попадают в целевое приложение.

Callbacks всегда диспетчеризуются в Qt-поток через QTimer.singleShot(0, ...),
поэтому безопасно вызывать Qt-слоты напрямую.
"""
from __future__ import annotations

import sys
from typing import Callable

from PyQt6.QtCore import QTimer

from dct.log import get_logger

_log = get_logger("hotkeys")


class GlobalHotkeyManager:
    """Менеджер глобальных хоткеев. Поддерживается только на Windows."""

    def __init__(self) -> None:
        self._hooks: list[tuple[str, Callable]] = []  # (key, callback) для удаления
        self._available = sys.platform == "win32"
        if self._available:
            try:
                import keyboard as _kb  # noqa: F401
            except ImportError:
                _log.warning("keyboard library not installed — global hotkeys disabled")
                self._available = False

    def register(self, key: str, qt_slot: Callable) -> None:
        """Регистрирует глобальный хук: нажатие key → вызов qt_slot в Qt-потоке."""
        if not self._available:
            return
        import keyboard

        def _dispatch():
            # keyboard callback выполняется в отдельном потоке — передаём в Qt
            QTimer.singleShot(0, qt_slot)

        keyboard.add_hotkey(key, _dispatch, suppress=False)
        self._hooks.append((key, _dispatch))
        _log.debug("Global hotkey registered: %s", key)

    def unregister_all(self) -> None:
        """Снимает все зарегистрированные хуки."""
        if not self._available:
            return
        import keyboard
        for key, _ in self._hooks:
            try:
                keyboard.remove_hotkey(key)
            except Exception:
                pass
        self._hooks.clear()
        _log.debug("All global hotkeys unregistered")

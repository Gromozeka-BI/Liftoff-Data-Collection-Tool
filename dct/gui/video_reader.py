"""Фоновый поток для чтения видеокадров без блокировки Qt-потока.

cv2.VideoCapture.set(CAP_PROP_POS_FRAMES) на H.264 mp4 декодирует от ближайшего
ключевого кадра — это может занять 100–500 ms. Выносим всё видео-IO сюда.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoReader:
    """Асинхронный ридер: принимает целевой ts_wall, возвращает последний готовый кадр."""

    def __init__(self, vid_path: Path, vid_ts: np.ndarray) -> None:
        self._cap = cv2.VideoCapture(str(vid_path))
        self._ts = vid_ts
        self._cur_idx: int = -1
        self._target_idx: int = 0
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="VideoReader")
        self._thread.start()

    # ── публичный API (из Qt-потока) ──────────────────────────────────────────

    def set_target_ts(self, ts_wall: float) -> None:
        """Устанавливает целевую позицию воспроизведения."""
        if len(self._ts) == 0:
            return
        idx = int(np.searchsorted(self._ts, ts_wall, side="left"))
        idx = max(0, min(idx, len(self._ts) - 1))
        with self._lock:
            changed = idx != self._target_idx
            self._target_idx = idx
        if changed:
            self._wakeup.set()

    def get_latest(self) -> Optional[np.ndarray]:
        """Возвращает последний готовый кадр (не блокирует)."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        self._cap.release()

    # ── внутренний поток ──────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._running:
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()
            if not self._running:
                break

            with self._lock:
                target = self._target_idx

            if target == self._cur_idx:
                continue

            # Если нужно прыгнуть назад или далеко вперёд — используем seek.
            # Если вперёд на ≤ 8 кадров — быстро пропускаем через grab() без decode.
            if target < self._cur_idx or target > self._cur_idx + 8:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self._cur_idx = target - 1  # read() сдвинет на +1
            else:
                # Последовательный fast-skip
                while self._cur_idx < target - 1:
                    self._cap.grab()
                    self._cur_idx += 1

            ret, frame = self._cap.read()
            if ret:
                # Конвертируем BGR→RGB здесь, в фоновом потоке.
                # Qt-поток получит уже готовый RGB и только создаст QImage.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cur_idx = target
                with self._lock:
                    self._latest = rgb

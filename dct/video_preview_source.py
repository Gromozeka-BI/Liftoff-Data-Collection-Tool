"""Lightweight video preview source — screen/device capture without file recording.

Used to show live preview in Record mode before (and after) a recording session.
Runs at ~20 fps to minimise CPU load (UI doesn't need 60 fps for preview).

Capture backend selection mirrors the recorder's:
``settings.screen_capture_backend`` ∈ {``"auto"``, ``"dxgi"``, ``"mss"``}.
``auto`` prefers DXGI (``dxcam``) on Windows so the live mouse cursor doesn't
jitter while the preview is running; falls back to mss when ``dxcam`` is
not installed or DXGI duplication is unavailable.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np

from dct.config import settings
from dct.log import get_logger
from dct.screen_recorder import (
    get_shared_dxgi_camera,
    is_virtual_device,
    list_all_video_device_names,
)

_log = get_logger("video_preview_source")

_PREVIEW_FPS = 20


def _resolve_screen_backend() -> str:
    """Return the actual backend to use: ``"dxgi"`` or ``"mss"``."""
    requested = (settings.screen_capture_backend or "auto").lower()
    if requested in ("mss", "gdi"):
        return "mss"
    if sys.platform != "win32":
        return "mss"
    try:
        import dxcam  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        if requested in ("dxgi", "dxcam"):
            _log.warning(
                "screen_capture_backend='%s' but dxcam is not installed — "
                "preview falls back to mss",
                requested,
            )
        return "mss"
    return "dxgi"


class VideoPreviewSource:
    """Continuously captures frames and calls ``on_frame(bgr_array)`` on each."""

    def __init__(self, source_cfg: dict, on_frame: Callable[[np.ndarray], None]):
        self._cfg = source_cfg
        self._on_frame = on_frame
        self._thread: threading.Thread | None = None
        self._running = False
        self._pyav_container = None   # held so stop() can interrupt av.open / decode

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Close pyav container from the calling thread to unblock any pending
        # av.open() or decode() call in the worker thread.
        c = self._pyav_container
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ── internal ───────────────────────────────────────────────────────────

    def _loop(self) -> None:
        if self._cfg.get("type") == "device":
            self._loop_device()
        else:
            self._loop_screen()

    def _loop_screen(self) -> None:
        backend = _resolve_screen_backend()
        if backend == "dxgi":
            try:
                self._loop_screen_dxgi()
                return
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "DXGI preview failed (%s) — falling back to mss",
                    exc,
                )
        self._loop_screen_mss()

    def _loop_screen_mss(self) -> None:
        import mss
        interval = 1.0 / _PREVIEW_FPS
        try:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                while self._running:
                    t0 = time.monotonic()
                    img = np.array(sct.grab(mon))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    self._on_frame(frame)
                    elapsed = time.monotonic() - t0
                    rem = interval - elapsed
                    if rem > 0:
                        time.sleep(rem)
        except Exception as exc:
            if self._running:
                _log.error("Screen preview error (mss): %s", exc)

    def _loop_screen_dxgi(self) -> None:
        """DXGI Desktop Duplication preview — does not perturb the hardware
        mouse cursor overlay, so the live cursor doesn't jitter while the
        preview is running."""
        interval = 1.0 / _PREVIEW_FPS
        camera = get_shared_dxgi_camera()
        last: np.ndarray | None = None
        while self._running:
            t0 = time.monotonic()
            try:
                frame = camera.grab()
            except Exception as exc:  # noqa: BLE001
                _log.debug("dxcam.grab() raised: %s", exc)
                frame = None

            if frame is None:
                # No new frame since last grab — reuse last frame so the
                # preview remains responsive at the requested FPS.
                frame = last
            else:
                last = frame

            if frame is not None:
                frame = cv2.resize(
                    frame, (640, 360), interpolation=cv2.INTER_LINEAR,
                )
                self._on_frame(frame)

            elapsed = time.monotonic() - t0
            rem = interval - elapsed
            if rem > 0:
                time.sleep(rem)

    def _loop_device(self) -> None:
        idx = self._cfg.get("index", 0)
        all_names = list_all_video_device_names()
        dev_name = all_names[idx] if idx < len(all_names) else ""
        virtual = is_virtual_device(dev_name) if dev_name else True

        if virtual or not dev_name:
            self._loop_device_opencv(idx)
        else:
            self._loop_device_pyav(dev_name)

    def _loop_device_opencv(self, idx: int) -> None:
        """Preview via OpenCV — used for virtual cameras (OBS etc.)."""
        interval = 1.0 / _PREVIEW_FPS
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                _log.error("VideoPreviewSource: cannot open device %d", idx)
                return
            while self._running:
                t0 = time.monotonic()
                ret, frame = cap.read()
                if ret:
                    self._on_frame(frame)
                elapsed = time.monotonic() - t0
                rem = interval - elapsed
                if rem > 0:
                    time.sleep(rem)
        except Exception as exc:
            if self._running:
                _log.error("Device preview error (opencv): %s", exc)
        finally:
            if cap:
                cap.release()

    def _loop_device_pyav(self, dev_name: str) -> None:
        """Preview via pyav/dshow — used for hardware UVC capture cards."""
        import av
        interval = 1.0 / _PREVIEW_FPS
        device_uri = f"video={dev_name}"
        _log.info("VideoPreviewSource: opening %s via pyav", device_uri)
        container = None
        try:
            for opts in (
                {"vcodec": "mjpeg", "video_size": "1280x720", "framerate": "30"},
                {"video_size": "1280x720", "framerate": "30"},
                {},
            ):
                if not self._running:
                    return
                try:
                    container = av.open(device_uri, format="dshow", options=opts)
                    self._pyav_container = container
                    break
                except Exception:
                    pass

            if container is None:
                _log.error("VideoPreviewSource: cannot open %s via pyav", device_uri)
                return

            for av_frame in container.decode(video=0):
                if not self._running:
                    break
                bgr = av_frame.to_ndarray(format="bgr24")
                self._on_frame(bgr)
                time.sleep(interval)

        except Exception as exc:
            if self._running:
                _log.error("Device preview error (pyav): %s", exc)
        finally:
            self._pyav_container = None
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

"""Cross-platform screen recorder that captures the LiftOff window.

Windows: uses pygetwindow to find window bounds.
Linux:   uses xdotool subprocess to find window bounds.
Fallback: captures primary monitor.

Each captured frame is timestamped with ts_wall at the moment of sct.grab()
and saved to video_timestamps.parquet for precise telemetry alignment.

video.mp4 is encoded at the target fps declared in VideoWriter. Actual playback
speed may differ from real time. Use video_timestamps.parquet (not fps header)
as the authoritative timeline — it gives the exact wall-clock time of each frame.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dct.log import get_logger

_log = get_logger("screen_recorder")


_TIMESTAMPS_SCHEMA = pa.schema([
    pa.field("frame_idx", pa.int64()),
    pa.field("ts_wall",   pa.float64()),
])


def _get_window_region(title: str) -> dict[str, int] | None:
    if sys.platform == "win32":
        try:
            import pygetwindow as gw
            wins = gw.getWindowsWithTitle(title)
            if not wins:
                return None
            w = wins[0]
            if w.width <= 0 or w.height <= 0:
                return None
            return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
        except Exception:
            return None
    else:
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title, "getwindowgeometry", "--shell"],
                capture_output=True, text=True, timeout=3,
            )
            geo: dict[str, int] = {}
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    geo[k.strip()] = int(v.strip())
            if "X" in geo and "Y" in geo and "WIDTH" in geo and "HEIGHT" in geo:
                return {"left": geo["X"], "top": geo["Y"],
                        "width": geo["WIDTH"], "height": geo["HEIGHT"]}
        except Exception:
            pass
        return None


class ScreenRecorder:
    def __init__(
        self,
        output_path: Path,
        window_title: str,
        fps: int = 60,
        target_w: int = 1280,
        target_h: int = 720,
    ):
        self._output = output_path
        self._ts_path = output_path.parent / "video_timestamps.parquet"
        self._title = window_title
        self._fps = fps
        self._target_w = target_w
        self._target_h = target_h
        self._thread: threading.Thread | None = None
        self._running = False
        self.frames_written = 0
        self.actual_fps: float = float(fps)
        self._error: Exception | None = None
        self.latest_frame_bgr: np.ndarray | None = None  # последний захваченный кадр для GUI

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        _log.info("Screen recorder started: %dx%d@%dfps → %s",
                  self._target_w, self._target_h, self._fps, self._output.name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        if self._error:
            _log.error("Screen recorder error: %s", self._error)
        else:
            _log.info("Screen recorder stopped: %d frames @ %.1f fps",
                      self.frames_written, self.actual_fps)

    def _record_loop(self) -> None:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self._output),
                fourcc,
                self._fps,
                (self._target_w, self._target_h),
            )
            interval = 1.0 / self._fps
            frame_timestamps: list[float] = []

            with mss.mss() as sct:
                region_cache: dict[str, int] | None = None
                region_cache_at = 0.0

                while self._running:
                    t0 = time.monotonic()

                    if t0 - region_cache_at > 5.0:
                        region_cache = _get_window_region(self._title)
                        region_cache_at = t0
                        if region_cache is None:
                            _log.debug("Window '%s' not found, capturing full screen", self._title)
                        else:
                            _log.debug("Window region: %s", region_cache)
                    monitor = region_cache if region_cache is not None else sct.monitors[1]

                    # Timestamp before grab() — closest to the game state in this frame.
                    ts_wall = time.time()
                    img = np.array(sct.grab(monitor))

                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                        frame = cv2.resize(frame, (self._target_w, self._target_h))

                    writer.write(frame)
                    self.latest_frame_bgr = frame
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1

                    elapsed = time.monotonic() - t0
                    sleep = interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            writer.release()
            self._save_timestamps(frame_timestamps)

            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

        except Exception as exc:
            self._error = exc
            _log.exception("Screen recorder crashed: %s", exc)

    def _save_timestamps(self, timestamps: list[float]) -> None:
        table = pa.table(
            {
                "frame_idx": pa.array(range(len(timestamps)), type=pa.int64()),
                "ts_wall":   pa.array(timestamps, type=pa.float64()),
            },
            schema=_TIMESTAMPS_SCHEMA,
        )
        pq.write_table(table, self._ts_path, compression="snappy")

    @property
    def has_error(self) -> bool:
        return self._error is not None


def _record_backend() -> int:
    """Backend for recording: MSMF on Windows correctly negotiates MJPEG with USB
    capture cards. DSHOW silently ignores CAP_PROP_FOURCC=MJPG and stays on YUY2."""
    if sys.platform != "win32":
        return cv2.CAP_ANY
    return cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_DSHOW


def scan_video_devices(max_index: int = 6) -> list[tuple[int, str]]:
    """Return list of (index, label) for available cv2 capture devices (excluding virtual).

    Uses DSHOW for enumeration — DSHOW supports index-based device discovery on Windows.
    MSMF uses different indices and cannot reliably enumerate by integer index.
    """
    found = []
    scan_backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, scan_backend)
        if cap.isOpened():
            name = cap.getBackendName()
            found.append((i, f"Device {i} ({name})"))
            cap.release()
    return found


class CaptureDeviceRecorder:
    """Records video from a USB capture card / webcam via cv2.VideoCapture."""

    def __init__(
        self,
        output_path: Path,
        device_index: int,
        fps: int = 60,
        target_w: int = 1280,
        target_h: int = 720,
    ):
        self._output = output_path
        self._ts_path = output_path.parent / "video_timestamps.parquet"
        self._device_index = device_index
        self._fps = fps
        self._target_w = target_w
        self._target_h = target_h
        self._thread: threading.Thread | None = None
        self._running = False
        self.frames_written = 0
        self.actual_fps: float = float(fps)
        self._error: Exception | None = None
        self.latest_frame_bgr: np.ndarray | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        _log.info("Capture device recorder started: device=%d %dx%d@%dfps → %s",
                  self._device_index, self._target_w, self._target_h, self._fps,
                  self._output.name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        if self._error:
            _log.error("Capture device recorder error: %s", self._error)
        else:
            _log.info("Capture device recorder stopped: %d frames @ %.1f fps",
                      self.frames_written, self.actual_fps)

    def _record_loop(self) -> None:
        try:
            cap = cv2.VideoCapture(self._device_index, _record_backend())
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open capture device {self._device_index}")

            # Request MJPEG from the device before setting resolution/FPS.
            # Without this, DirectShow negotiates uncompressed YUV which is
            # bandwidth-limited to ~10 FPS at 720p over USB 2.0.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_h)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            # Keep the internal DirectShow buffer minimal so cap.read() always
            # returns the most recent frame without stale-queue delay.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Log what the driver actually agreed to (may differ from what we asked).
            actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(chr((actual_fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
            actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            _log.info(
                "Capture device %d negotiated: %dx%d @ %.1f fps  fourcc=%s  (requested %dx%d @ %d fps MJPG)",
                self._device_index, actual_w, actual_h, actual_fps, actual_fourcc,
                self._target_w, self._target_h, self._fps,
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self._output), fourcc, self._fps, (self._target_w, self._target_h)
            )
            interval = 1.0 / self._fps
            frame_timestamps: list[float] = []

            # Track real-time progress so we can pad with duplicate frames when the
            # capture device delivers fewer FPS than the target (e.g. 10 vs 30).
            # This keeps the output file at correct real-time speed regardless of
            # actual device frame rate.
            t_start = time.monotonic()
            frames_written_idx = 0  # total slots written to VideoWriter (includes duplicates)
            captured_count = 0      # unique frames actually read from the device
            _last_fps_log = t_start
            _captured_at_last_log = 0

            while self._running:
                ts_wall = time.time()
                ret, frame = cap.read()
                if not ret:
                    _log.warning("Capture device %d: failed to read frame", self._device_index)
                    continue

                captured_count += 1

                # Log actual incoming FPS every 5 seconds so the user can see
                # whether the device is truly delivering the requested rate.
                now_log = time.monotonic()
                if now_log - _last_fps_log >= 5.0:
                    elapsed_log = now_log - _last_fps_log
                    incoming_fps = (captured_count - _captured_at_last_log) / elapsed_log
                    _log.info(
                        "Capture device %d: incoming %.1f fps (unique)  written %d frames total (with dupes)",
                        self._device_index, incoming_fps, self.frames_written,
                    )
                    _last_fps_log = now_log
                    _captured_at_last_log = captured_count

                if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                    frame = cv2.resize(frame, (self._target_w, self._target_h))

                self.latest_frame_bgr = frame

                # How many frames should exist in the file by now (real-time)?
                now = time.monotonic()
                frames_due = int((now - t_start) / interval) + 1

                # Write this frame once per "slot" that has passed since the last read.
                # When the device delivers 10 FPS each captured frame fills ~3 slots.
                while frames_written_idx < frames_due:
                    writer.write(frame)
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1
                    frames_written_idx += 1

            cap.release()
            writer.release()
            self._save_timestamps(frame_timestamps)

            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

        except Exception as exc:
            self._error = exc
            _log.exception("Capture device recorder crashed: %s", exc)

    def _save_timestamps(self, timestamps: list[float]) -> None:
        table = pa.table(
            {
                "frame_idx": pa.array(range(len(timestamps)), type=pa.int64()),
                "ts_wall":   pa.array(timestamps, type=pa.float64()),
            },
            schema=_TIMESTAMPS_SCHEMA,
        )
        pq.write_table(table, self._ts_path, compression="snappy")

    @property
    def has_error(self) -> bool:
        return self._error is not None

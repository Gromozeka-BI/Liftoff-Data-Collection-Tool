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


def _write_timestamps(path: Path, timestamps: list[float]) -> None:
    table = pa.table(
        {
            "frame_idx": pa.array(range(len(timestamps)), type=pa.int64()),
            "ts_wall":   pa.array(timestamps, type=pa.float64()),
        },
        schema=_TIMESTAMPS_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


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
            _write_timestamps(self._ts_path, frame_timestamps)

            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

        except Exception as exc:
            self._error = exc
            _log.exception("Screen recorder crashed: %s", exc)

    @property
    def has_error(self) -> bool:
        return self._error is not None


def _open_capture(device_index: int) -> cv2.VideoCapture:
    """Open a capture device, preferring MSMF on Windows (proper MJPEG negotiation).
    Falls back to DSHOW when MSMF cannot open the device — MSMF and DSHOW may use
    different internal device indices for the same physical hardware."""
    if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
        cap = cv2.VideoCapture(device_index, cv2.CAP_MSMF)
        if cap.isOpened():
            _log.debug("Capture device %d opened via MSMF", device_index)
            return cap
        cap.release()
        _log.debug("MSMF could not open device %d, falling back to DSHOW", device_index)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(device_index, backend)


def _list_dshow_video_devices() -> list[str]:
    """Return DirectShow video device friendly names in enumeration order.

    Reads from the Windows registry: DirectShow video input devices register
    their friendly names under HKCR\\CLSID\\{860BB310...}\\Instance.
    The registry enumeration order matches ICreateDevEnum (alphabetical by
    filter name), which is the same order OpenCV DSHOW uses for indices.
    """
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    # CLSID for VideoInputDeviceCategory
    VIDEO_INPUT_CAT = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
    key_path = f"CLSID\\{VIDEO_INPUT_CAT}\\Instance"

    names: list[str] = []
    try:
        cat_key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path)
        i = 0
        while True:
            try:
                clsid = winreg.EnumKey(cat_key, i)
                try:
                    dev_key = winreg.OpenKey(cat_key, clsid)
                    friendly, _ = winreg.QueryValueEx(dev_key, "FriendlyName")
                    winreg.CloseKey(dev_key)
                    if friendly:
                        names.append(friendly)
                except OSError:
                    pass
                i += 1
            except OSError:
                break
        winreg.CloseKey(cat_key)
    except Exception as exc:
        _log.debug("Registry dshow enumeration failed: %s", exc)

    # DirectShow enumerates in the same order as the registry key insertion
    # order (which matches alphabetical sort for most drivers).
    _log.debug("Registry dshow devices: %s", names)
    return names


def scan_video_devices(max_index: int = 6) -> list[tuple[int, str]]:
    """Return list of (index, label) for available capture devices.

    On Windows: uses pyav to get DirectShow friendly names (same enumeration
    order as OpenCV DSHOW). Falls back to OpenCV DSHOW index scanning.
    """
    if sys.platform == "win32":
        try:
            dshow_names = _list_dshow_video_devices()
            if dshow_names:
                return [(i, name) for i, name in enumerate(dshow_names)]
        except Exception:
            pass

    # Fallback: OpenCV index scan
    found = []
    scan_backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, scan_backend)
        if cap.isOpened():
            found.append((i, f"Device {i}"))
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
            cap = _open_capture(self._device_index)
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
            _write_timestamps(self._ts_path, frame_timestamps)

            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

        except Exception as exc:
            self._error = exc
            _log.exception("Capture device recorder crashed: %s", exc)

    @property
    def has_error(self) -> bool:
        return self._error is not None


class PyAvCaptureRecorder:
    """Records from a USB capture card using pyav (libav/ffmpeg bindings).

    Opens the DirectShow device via libav's dshow demuxer which correctly
    negotiates MJPEG format — achieving 30 FPS where OpenCV's backends fail.
    Frames are timestamped per-frame for precise telemetry alignment.
    """

    def __init__(
        self,
        output_path: Path,
        device_index: int,
        fps: int = 30,
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
        self._in_container = None   # held so stop() can interrupt decode()
        self.frames_written = 0
        self.actual_fps: float = float(fps)
        self._error: Exception | None = None
        self.latest_frame_bgr: np.ndarray | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        _log.info("PyAv capture recorder started: device=%d %dx%d@%dfps → %s",
                  self._device_index, self._target_w, self._target_h, self._fps,
                  self._output.name)

    def stop(self) -> None:
        self._running = False
        # Close the container from outside to unblock any pending decode() call.
        c = self._in_container
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=30)
        if self._error:
            _log.error("PyAv capture recorder error: %s", self._error)
        else:
            _log.info("PyAv capture recorder stopped: %d frames @ %.1f fps",
                      self.frames_written, self.actual_fps)

    def _record_loop(self) -> None:
        import av

        # Resolve device index → DirectShow friendly name.
        # ffmpeg/libav dshow requires "video=<Device Name>", not index-based addressing.
        dshow_names = _list_dshow_video_devices()
        if self._device_index < len(dshow_names):
            device_uri = f"video={dshow_names[self._device_index]}"
        else:
            # pyav device enumeration failed — fall back to OpenCV capture.
            # Video will record at the device's native FPS (may be ~10fps on some cards).
            _log.warning(
                "pyav dshow enumeration returned %d devices — "
                "falling back to OpenCV CaptureDeviceRecorder",
                len(dshow_names),
            )
            self._record_loop_opencv()
            return

        _log.info("pyav opening device: %s", device_uri)

        base_opts = {
            "video_size": f"{self._target_w}x{self._target_h}",
            "framerate":  str(self._fps),
        }
        # Try MJPEG first (higher throughput over USB), then let the driver decide.
        in_container = None
        for extra in ({"vcodec": "mjpeg"}, {}):
            try:
                in_container = av.open(
                    device_uri,
                    format="dshow",
                    options={**base_opts, **extra},
                )
                break
            except Exception as e:
                _log.debug("pyav dshow open attempt failed (extra=%s): %s", extra, e)

        if in_container is None:
            self._error = RuntimeError(
                f"Cannot open capture device '{device_uri}' via pyav/dshow"
            )
            _log.error(str(self._error))
            return

        self._in_container = in_container

        try:
            in_stream = in_container.streams.video[0]
            codec_name = (
                in_stream.codec_context.codec.name
                if in_stream.codec_context and in_stream.codec_context.codec
                else "unknown"
            )
            rate = float(in_stream.average_rate) if in_stream.average_rate else self._fps
            _log.info(
                "pyav device %d negotiated: %dx%d @ %.1f fps  codec=%s",
                self._device_index,
                in_stream.codec_context.width, in_stream.codec_context.height,
                rate, codec_name,
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self._output), fourcc, self._fps, (self._target_w, self._target_h)
            )
            interval = 1.0 / self._fps
            frame_timestamps: list[float] = []
            t_start = time.monotonic()
            frames_written_idx = 0
            captured_count = 0
            _last_fps_log = t_start
            _captured_at_last_log = 0

            for av_frame in in_container.decode(video=0):
                if not self._running:
                    break

                ts_wall = time.time()
                captured_count += 1

                now_log = time.monotonic()
                if now_log - _last_fps_log >= 5.0:
                    incoming_fps = (captured_count - _captured_at_last_log) / (now_log - _last_fps_log)
                    _log.info(
                        "pyav device %d: incoming %.1f fps (unique)  written %d frames total",
                        self._device_index, incoming_fps, self.frames_written,
                    )
                    _last_fps_log = now_log
                    _captured_at_last_log = captured_count

                bgr = av_frame.to_ndarray(format="bgr24")
                if bgr.shape[1] != self._target_w or bgr.shape[0] != self._target_h:
                    bgr = cv2.resize(bgr, (self._target_w, self._target_h))

                self.latest_frame_bgr = bgr

                now = time.monotonic()
                frames_due = int((now - t_start) / interval) + 1
                while frames_written_idx < frames_due:
                    writer.write(bgr)
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1
                    frames_written_idx += 1

            writer.release()
            _write_timestamps(self._ts_path, frame_timestamps)

            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

        except Exception as exc:
            if self._running:
                self._error = exc
                _log.exception("pyav capture recorder crashed: %s", exc)
        finally:
            try:
                in_container.close()
            except Exception:
                pass
            self._in_container = None

    def _record_loop_opencv(self) -> None:
        """Fallback path: record via OpenCV DSHOW when pyav device enumeration fails."""
        _log.info("PyAv fallback: recording via OpenCV device=%d", self._device_index)
        cap = _open_capture(self._device_index)
        if not cap.isOpened():
            self._error = RuntimeError(
                f"OpenCV fallback: cannot open capture device {self._device_index}"
            )
            _log.error(str(self._error))
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_h)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self._output), fourcc, self._fps, (self._target_w, self._target_h)
        )
        interval = 1.0 / self._fps
        frame_timestamps: list[float] = []
        t_start = time.monotonic()
        frames_written_idx = 0

        try:
            while self._running:
                ts_wall = time.time()
                ret, frame = cap.read()
                if not ret:
                    continue
                if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                    frame = cv2.resize(frame, (self._target_w, self._target_h))
                self.latest_frame_bgr = frame
                now = time.monotonic()
                frames_due = int((now - t_start) / interval) + 1
                while frames_written_idx < frames_due:
                    writer.write(frame)
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1
                    frames_written_idx += 1
        finally:
            cap.release()
            writer.release()
            _write_timestamps(self._ts_path, frame_timestamps)
            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    self.actual_fps = round(self.frames_written / duration, 1)

    @property
    def has_error(self) -> bool:
        return self._error is not None

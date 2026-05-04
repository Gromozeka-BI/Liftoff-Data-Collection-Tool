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

import multiprocessing as mp
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


def _make_av_writer(output_path: Path, fps: int, width: int, height: int):
    """Open a pyav H.264 output container.  Returns (container, stream)."""
    import av
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width   = width
    stream.height  = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "ultrafast", "movflags": "faststart"}
    return container, stream


def _av_write_frame(stream, container, bgr_frame: np.ndarray, pts: int) -> None:
    """Encode one BGR frame and mux into *container*."""
    import av
    av_frame = av.VideoFrame.from_ndarray(
        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), format="rgb24"
    )
    av_frame.pts = pts
    for packet in stream.encode(av_frame):
        container.mux(packet)


def _av_close(stream, container) -> None:
    """Flush encoder and close container."""
    for packet in stream.encode():
        container.mux(packet)
    container.close()


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
            out_container, out_stream = _make_av_writer(
                self._output, self._fps, self._target_w, self._target_h
            )
            interval = 1.0 / self._fps
            frame_timestamps: list[float] = []
            pts = 0

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

                    _av_write_frame(out_stream, out_container, frame, pts)
                    pts += 1
                    self.latest_frame_bgr = frame
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1

                    elapsed = time.monotonic() - t0
                    sleep = interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            _av_close(out_stream, out_container)
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


def _list_camera_class_devices() -> list[str]:
    """Return friendly names of Windows Camera-class UVC devices.

    USB capture cards using the standard Microsoft UVC driver (usbvideo.inf)
    register under the Windows Camera setup class
    (ClassGuid = {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}) instead of the
    classic DirectShow VideoInputDeviceCategory. This function scans
    HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USB for Camera-class entries so
    that pyav/dshow can open them as 'video=<FriendlyName>'.

    Only active (Status OK) instances are returned.
    """
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    CAMERA_CLASS_GUID = "{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}"
    usb_root = r"SYSTEM\CurrentControlSet\Enum\USB"
    names: list[str] = []
    seen: set[str] = set()

    try:
        usb_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, usb_root)
        vid_idx = 0
        while True:
            try:
                vid_name = winreg.EnumKey(usb_key, vid_idx)
                vid_key = winreg.OpenKey(usb_key, vid_name)
                inst_idx = 0
                while True:
                    try:
                        inst_name = winreg.EnumKey(vid_key, inst_idx)
                        inst_key = winreg.OpenKey(vid_key, inst_name)
                        try:
                            class_guid, _ = winreg.QueryValueEx(inst_key, "ClassGUID")
                            if class_guid.lower() == CAMERA_CLASS_GUID:
                                try:
                                    friendly, _ = winreg.QueryValueEx(inst_key, "FriendlyName")
                                except OSError:
                                    friendly = ""
                                if not friendly:
                                    try:
                                        friendly, _ = winreg.QueryValueEx(inst_key, "DeviceDesc")
                                    except OSError:
                                        friendly = ""
                                # Strip driver-store decoration (e.g. "@oem12.inf,..." prefix)
                                if friendly and friendly.startswith("@"):
                                    friendly = ""
                                if friendly and friendly not in seen:
                                    seen.add(friendly)
                                    names.append(friendly)
                        except OSError:
                            pass
                        finally:
                            winreg.CloseKey(inst_key)
                        inst_idx += 1
                    except OSError:
                        break
                winreg.CloseKey(vid_key)
                vid_idx += 1
            except OSError:
                break
        winreg.CloseKey(usb_key)
    except Exception as exc:
        _log.debug("Camera class registry enumeration failed: %s", exc)

    _log.debug("Camera class devices: %s", names)
    return names


def list_all_video_device_names() -> list[str]:
    """Return all capturable video device names: DirectShow + Camera-class UVC.

    Used by PyAvCaptureRecorder to resolve a device index to a
    'video=<FriendlyName>' URI for libav dshow.  Virtual devices (OBS etc.)
    appear first (from DirectShow registry), followed by UVC hardware devices
    (from Camera class registry).
    """
    dshow = _list_dshow_video_devices()
    camera = _list_camera_class_devices()
    # Merge: avoid duplicates (some devices register in both places).
    seen = set(dshow)
    for name in camera:
        if name not in seen:
            dshow.append(name)
            seen.add(name)
    return dshow


# Known virtual / software DirectShow devices that libav dshow cannot open.
_VIRTUAL_DEVICE_KEYWORDS = (
    "obs virtual",
    "virtual camera",
    "vcam",
    "ndi",
    "snap camera",
    "droidcam",
    "iriun",
    "epoccam",
    "camo",
    "reincubate",
)


def is_virtual_device(name: str) -> bool:
    """Return True if the device friendly name indicates a software/virtual camera.

    libav/dshow can only open real hardware devices; virtual cameras must be
    captured via OpenCV (CaptureDeviceRecorder) instead of PyAvCaptureRecorder.
    """
    lname = name.lower()
    return any(kw in lname for kw in _VIRTUAL_DEVICE_KEYWORDS)


def scan_video_devices(max_index: int = 6) -> list[tuple[int, str]]:
    """Return list of (index, label) for available capture devices.

    On Windows: merges DirectShow VideoInputDeviceCategory devices (virtual
    cameras, webcams) with Windows Camera-class UVC devices (capture cards
    using usbvideo.inf).  Falls back to OpenCV DSHOW index scanning when
    neither registry source yields any results.
    """
    if sys.platform == "win32":
        try:
            all_names = list_all_video_device_names()
            if all_names:
                return [(i, name) for i, name in enumerate(all_names)]
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
            self._thread.join(timeout=5)
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

            out_container, out_stream = _make_av_writer(
                self._output, self._fps, self._target_w, self._target_h
            )
            interval = 1.0 / self._fps
            frame_timestamps: list[float] = []

            # Track real-time progress so we can pad with duplicate frames when the
            # capture device delivers fewer FPS than the target (e.g. 10 vs 30).
            # This keeps the output file at correct real-time speed regardless of
            # actual device frame rate.
            t_start = time.monotonic()
            frames_written_idx = 0  # total pts written to encoder (includes duplicates)
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
                    _av_write_frame(out_stream, out_container, frame, frames_written_idx)
                    frame_timestamps.append(ts_wall)
                    self.frames_written += 1
                    frames_written_idx += 1

            cap.release()
            _av_close(out_stream, out_container)
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
    """Records hardware UVC capture cards via pyav/dshow + H.264 output.

    Architecture:
      • A separate *process* runs the capture + encode loop so that blocking
        av.open() and demux() calls never freeze the Qt GUI thread.
      • Two queues keep the IPC clean:
          _cmd_queue   – parent → child: {"stop": True}
          _status_queue – child → parent: {"status"}, {"frames", "fps"}, {"error"}
      • A lightweight daemon *thread* drains _status_queue and _preview_queue
        so the child process never blocks on a full queue.
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
        self._device_name: str | None = None
        self._fps = fps
        self._target_w = target_w
        self._target_h = target_h
        self._process: "mp.Process | None" = None  # type: ignore[name-defined]
        self._running = False
        self.frames_written = 0
        self.actual_fps: float = float(fps)
        self._error: Exception | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self._cmd_queue = None
        self._status_queue = None
        # maxsize=2: drop stale preview frames; child skips put() when full
        self._preview_queue = None

    def start(self) -> None:
        import multiprocessing as mp

        all_names = list_all_video_device_names()
        if self._device_index >= len(all_names):
            self._error = RuntimeError(f"Device index {self._device_index} not found")
            _log.error("Device index %d not found in device list", self._device_index)
            return

        self._device_name = all_names[self._device_index]
        _log.info("PyAvCaptureRecorder: device=%s (index=%d)", self._device_name, self._device_index)

        self._cmd_queue    = mp.Queue()
        self._status_queue = mp.Queue()
        self._preview_queue = mp.Queue(maxsize=2)

        self._process = mp.Process(
            target=PyAvCaptureRecorder._record_process,
            args=(
                self._output, self._ts_path, self._device_name,
                self._fps, self._target_w, self._target_h,
                self._cmd_queue, self._status_queue, self._preview_queue,
            ),
            daemon=True,
        )
        self._running = True
        self._process.start()
        _log.info(
            "PyAv capture recorder started: device=%s %dx%d@%dfps → %s",
            self._device_name, self._target_w, self._target_h, self._fps, self._output.name,
        )

        # Drain queues in a background thread — prevents child blocking on full queue
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _drain_loop(self) -> None:
        """Background thread: forward status messages and preview frames."""
        while self._running or (self._process and self._process.is_alive()):
            # Status messages
            try:
                while True:
                    msg = self._status_queue.get_nowait()
                    if "error" in msg:
                        _log.error("PyAv recorder process error: %s", msg["error"])
                        self._error = RuntimeError(msg["error"])
                    elif "frames" in msg:
                        self.frames_written = msg["frames"]
                        if "fps" in msg:
                            self.actual_fps = msg["fps"]
                    elif "status" in msg:
                        _log.debug("PyAv recorder: %s", msg["status"])
            except Exception:
                pass

            # Preview frames — keep only the latest
            try:
                while True:
                    frame = self._preview_queue.get_nowait()
                    if frame is not None:
                        self.latest_frame_bgr = frame
            except Exception:
                pass

            time.sleep(0.05)

    def stop(self) -> None:
        self._running = False

        if self._process:
            _log.debug("Stopping PyAv recorder process...")
            try:
                self._cmd_queue.put({"stop": True}, timeout=1)
            except Exception:
                pass

            # Wait for graceful finish (encoder flush can take a moment)
            self._process.join(timeout=15)

            if self._process.is_alive():
                _log.warning("PyAv recorder process still alive after 15 s — terminating")
                self._process.terminate()
                self._process.join(timeout=3)
            if self._process.is_alive():
                _log.warning("PyAv recorder process still alive — killing")
                self._process.kill()
                self._process.join()

            self._process = None

        # Drain any remaining status messages after process exits
        if self._status_queue:
            try:
                while True:
                    msg = self._status_queue.get_nowait()
                    if "frames" in msg:
                        self.frames_written = msg["frames"]
                    if "fps" in msg:
                        self.actual_fps = msg["fps"]
            except Exception:
                pass

        if self.frames_written > 0:
            _log.info(
                "PyAv capture recorder stopped: %d frames @ %.1f fps",
                self.frames_written, self.actual_fps,
            )
        else:
            _log.warning("PyAv capture recorder stopped with 0 frames — check device")

    @staticmethod
    def _record_process(
        output_path: Path,
        ts_path: Path,
        device_name: str,
        fps: int,
        target_w: int,
        target_h: int,
        cmd_queue,     # parent → child: {"stop": True}
        status_queue,  # child → parent: status / frame count / errors
        preview_queue, # child → parent: bgr frames for GUI preview (maxsize=2)
    ) -> None:
        """Entry point for the recording subprocess."""
        import av
        import cv2 as _cv2
        import time as _time

        def _put_status(msg: dict) -> None:
            try:
                status_queue.put_nowait(msg)
            except Exception:
                pass  # queue full — non-critical

        def _put_preview(bgr: np.ndarray) -> None:
            try:
                preview_queue.put_nowait(bgr)
            except Exception:
                pass  # full — GUI will show a slightly older frame

        try:
            device_uri = f"video={device_name}"
            _put_status({"status": f"opening {device_uri}"})

            in_container = av.open(
                device_uri,
                format="dshow",
                options={
                    "video_size": f"{target_w}x{target_h}",
                    "framerate": str(fps),
                    "vcodec": "mjpeg",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                },
            )
            _put_status({"status": "device_opened"})

            out_container = av.open(str(output_path), mode="w")
            out_stream = out_container.add_stream("libx264", rate=fps)
            out_stream.width   = target_w
            out_stream.height  = target_h
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {
                "crf": "23",
                "preset": "ultrafast",
                "movflags": "faststart",
            }
            _put_status({"status": "capturing"})

            frame_timestamps: list[float] = []
            frames_written = 0
            t_start = _time.monotonic()

            for packet in in_container.demux(video=0):
                # Check for stop command (non-blocking)
                try:
                    msg = cmd_queue.get_nowait()
                    if msg.get("stop"):
                        break
                except Exception:
                    pass

                for av_frame in packet.decode():
                    frames_written += 1
                    ts_wall = _time.time()

                    bgr = av_frame.to_ndarray(format="bgr24")
                    if bgr.shape[1] != target_w or bgr.shape[0] != target_h:
                        bgr = _cv2.resize(bgr, (target_w, target_h))

                    # Preview every 5th frame
                    if frames_written % 5 == 0:
                        _put_preview(bgr.copy())

                    # Encode to H.264
                    rgb_frame = av.VideoFrame.from_ndarray(
                        _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB), format="rgb24"
                    )
                    rgb_frame.pts = frames_written
                    for pkt in out_stream.encode(rgb_frame):
                        out_container.mux(pkt)

                    frame_timestamps.append(ts_wall)

                    # Report progress every 30 frames
                    if frames_written % 30 == 0:
                        elapsed = _time.monotonic() - t_start
                        cur_fps = round(frames_written / elapsed, 1) if elapsed > 0 else 0.0
                        _put_status({"frames": frames_written, "fps": cur_fps})

            # Flush encoder
            for pkt in out_stream.encode():
                out_container.mux(pkt)
            out_container.close()
            in_container.close()

            if frame_timestamps:
                _write_timestamps(ts_path, frame_timestamps)
                elapsed = _time.monotonic() - t_start
                final_fps = round(frames_written / elapsed, 1) if elapsed > 0 else 0.0
                _put_status({"frames": frames_written, "fps": final_fps, "status": "finished"})
            else:
                _put_status({"error": "No frames captured"})

        except Exception as exc:
            _put_status({"error": str(exc)})
            raise

    @property
    def has_error(self) -> bool:
        return self._error is not None
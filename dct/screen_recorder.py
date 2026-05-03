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
    def __init__(self, output_path: Path, device_index: int, fps: int = 30, target_w: int = 1280, target_h: int = 720):
        self._output = output_path
        self._ts_path = output_path.parent / "video_timestamps.parquet"
        self._device_index = device_index
        self._device_name = None
        self._fps = fps
        self._target_w = target_w
        self._target_h = target_h
        self._process = None
        self._running = False
        self.frames_written = 0
        self.actual_fps: float = float(fps)
        self._error: Exception | None = None
        self.latest_frame_bgr: np.ndarray | None = None  # Для GUI
        self._frames_queue = None
        self._preview_queue = None  # Новая очередь для кадров предпросмотра

    def start(self) -> None:
        """Start the recording process."""
        self._running = True
        import multiprocessing as mp
        
        all_names = list_all_video_device_names()
        if self._device_index < len(all_names):
            self._device_name = all_names[self._device_index]
        else:
            _log.error("Device index %d not found in device list", self._device_index)
            self._error = RuntimeError(f"Device index {self._device_index} not found")
            return
        
        _log.info("Using device: %s (index %d)", self._device_name, self._device_index)
        
        self._frames_queue = mp.Queue()
        self._preview_queue = mp.Queue()  # Для кадров предпросмотра
        
        self._process = mp.Process(target=self._record_process, args=(
            self._output, self._ts_path, self._device_name,
            self._fps, self._target_w, self._target_h,
            self._frames_queue, self._preview_queue
        ))
        self._process.start()
        _log.info("PyAv capture recorder started in separate process: device=%s %dx%d@%dfps → %s",
                  self._device_name, self._target_w, self._target_h, self._fps, self._output.name)
        
        time.sleep(1.5)
        
        # Запускаем поток для получения кадров предпросмотра
        self._start_preview_receiver()
        
        if not self._process.is_alive():
            _log.error("Process died immediately after start")
            try:
                while not self._frames_queue.empty():
                    err = self._frames_queue.get_nowait()
                    _log.error("Process error: %s", err)
            except Exception:
                pass
            self._error = RuntimeError("Recording process failed to start")

    def _start_preview_receiver(self) -> None:
        """Запускает поток для получения кадров предпросмотра из процесса."""
        import threading
        
        def receive_preview_frames():
            while self._running:
                try:
                    # Ждём кадр с таймаутом
                    frame_data = self._preview_queue.get(timeout=0.1)
                    if frame_data is None:
                        break
                    # Обновляем последний кадр для GUI
                    self.latest_frame_bgr = frame_data
                except Exception:
                    pass
        
        self._preview_thread = threading.Thread(target=receive_preview_frames, daemon=True)
        self._preview_thread.start()

    def stop(self) -> None:
        """Stop the recording process gracefully."""
        self._running = False
        
        if self._process:
            _log.debug("Stopping PyAv recorder process gracefully...")
            
            # Проверяем, сколько кадров было записано
            try:
                frames_from_queue = 0
                while not self._frames_queue.empty():
                    msg = self._frames_queue.get_nowait()
                    if "frames" in msg:
                        frames_from_queue = msg["frames"]
                if frames_from_queue > 0:
                    self.frames_written = frames_from_queue
                _log.debug("Frames written: %d", self.frames_written)
            except Exception as e:
                _log.debug("Error reading from queue: %s", e)
            
            # Отправляем сигнал завершения
            try:
                self._frames_queue.put({"stop": True}, timeout=1)
            except Exception:
                pass
            
            # Ждём 3 секунды для корректного завершения
            self._process.join(timeout=3)
            
            if self._process.is_alive():
                _log.warning("Process still alive after 3s, terminating...")
                self._process.terminate()
                self._process.join(timeout=2)
                
                if self._process.is_alive():
                    _log.warning("Process still alive, killing...")
                    self._process.kill()
                    self._process.join()
            
            self._process = None
        
        if self.frames_written > 0:
            _log.info("PyAv capture recorder stopped: %d frames", self.frames_written)
        else:
            _log.warning("PyAv capture recorder stopped with 0 frames - check device availability")

    @staticmethod
    def _record_process(
        output_path: Path,
        ts_path: Path,
        device_name: str,
        fps: int,
        target_w: int,
        target_h: int,
        frames_queue,
        preview_queue,  # Новая очередь для кадров предпросмотра
    ) -> None:
        """Run in a separate process - handles the actual recording."""
        import av
        import time
        import sys
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        try:
            device_uri = f"video={device_name}"
            frames_queue.put({"status": f"opening {device_uri}"})
            
            in_container = av.open(
                device_uri,
                format="dshow",
                options={
                    "video_size": f"{target_w}x{target_h}",
                    "framerate": str(fps),
                    "vcodec": "mjpeg",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                }
            )
            
            frames_queue.put({"status": "device_opened"})
            
            out_container = av.open(str(output_path), mode="w")
            out_stream = out_container.add_stream("libx264", rate=fps)
            out_stream.width = target_w
            out_stream.height = target_h
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {
                "crf": "23",
                "preset": "ultrafast",
                "movflags": "faststart",
            }
            
            frames_queue.put({"status": "output_opened"})
            
            frame_timestamps: list[float] = []
            frames_written = 0
            stop_requested = False
            
            frames_queue.put({"status": "capturing"})
            
            for packet in in_container.demux(video=0):
                try:
                    if not frames_queue.empty():
                        msg = frames_queue.get_nowait()
                        if msg.get("stop"):
                            stop_requested = True
                            break
                except Exception:
                    pass
                
                if stop_requested:
                    break
                
                for av_frame in packet.decode():
                    frames_written += 1
                    ts_wall = time.time()
                    
                    if frames_written % 30 == 0:
                        frames_queue.put({"frames": frames_written})
                    
                    bgr = av_frame.to_ndarray(format="bgr24")
                    if bgr.shape[1] != target_w or bgr.shape[0] != target_h:
                        import cv2
                        bgr = cv2.resize(bgr, (target_w, target_h))
                    
                    # Отправляем кадр для предпросмотра (уменьшенный для экономии)
                    # Отправляем каждый 5-й кадр, чтобы не перегружать очередь
                    if frames_written % 5 == 0:
                        # Отправляем копию кадра
                        preview_queue.put(bgr.copy())
                    
                    import cv2
                    yuv_frame = av.VideoFrame.from_ndarray(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24"
                    )
                    yuv_frame.pts = frames_written
                    
                    for out_packet in out_stream.encode(yuv_frame):
                        out_container.mux(out_packet)
                    
                    frame_timestamps.append(ts_wall)
            
            # Flush encoder
            for packet in out_stream.encode():
                out_container.mux(packet)
            
            out_container.close()
            in_container.close()
            
            if frame_timestamps:
                _write_timestamps(ts_path, frame_timestamps)
                frames_queue.put({"frames": len(frame_timestamps), "finished": True})
            else:
                frames_queue.put({"error": "No timestamps collected"})
                
        except Exception as e:
            frames_queue.put({"error": str(e)})
            raise

    @property
    def has_error(self) -> bool:
        return self._error is not None
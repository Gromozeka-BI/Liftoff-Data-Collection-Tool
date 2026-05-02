"""RC receiver — reads ESP32 CRSF data from COM port at 100 Hz.

Line format: timestamp_us,ch1,ch2,...,ch8

Time synchronisation
--------------------
After flushing the OS serial buffer on connect we capture one ``time.time()``
anchor at the first packet and then derive every subsequent wall-clock timestamp
from the ESP32's own ``micros()`` counter:

    ts_wall = T_anchor + (ts_us - D_anchor) / 1_000_000

This gives us:
  * Correct absolute position — anchored to PC wall clock at connect time.
  * Drift-free relative spacing — driven by the ESP32 crystal (≈50 ppm,
    i.e. 0.5 ms error per 10 s), far better than Windows thread-scheduling
    jitter (can be 15–50 ms per burst) if we used time.time() per packet.

``micros()`` wraps at 2³² µs ≈ 71.6 min; rollover is handled below.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from dct.log import get_logger

_log = get_logger("rc_receiver")

_BAUD        = 115_200
_US_ROLLOVER = 2**32          # micros() wraps every ~71.6 min


def scan_serial_ports() -> list[str]:
    """Return list of available serial port names (blocking, call from thread)."""
    try:
        import serial.tools.list_ports
        return sorted(p.device for p in serial.tools.list_ports.comports())
    except Exception:
        return []


class RCReceiver:
    """Reads ESP32 CSV lines from COM port and queues parsed frames.

    Thread-safe: ``_read_loop`` runs in a daemon thread and puts frames into
    ``self.queue``. The main thread drains the queue in its tick.
    """

    def __init__(
        self,
        port: str,
        on_status_change: Callable[[bool], None] | None = None,
    ):
        self._port              = port
        self._on_status_change  = on_status_change
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=1000)
        self._thread: threading.Thread | None = None
        self._running           = False
        self._connected         = False
        self._disconnect_logged = False
        self._seq               = 0
        # Anchor for device-clock → wall-clock conversion (set on first packet)
        self._anchor_wall:  float | None = None  # time.time() at first packet
        self._prev_dev:     int   | None = None  # previous ts_device_us (for delta)
        self._elapsed_us:   int          = 0     # cumulative µs since anchor (handles rollover)

    # ── public ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── internal ───────────────────────────────────────────────────────────

    def _set_connected(self, state: bool) -> None:
        if self._connected != state:
            self._connected = state
            if self._on_status_change:
                self._on_status_change(state)

    def _read_loop(self) -> None:
        """High-throughput read loop.

        Instead of pyserial's ``readline()`` (which calls ``read(1)`` in a loop
        — up to 5 000 syscalls/s at 100 Hz × 50 B/packet on Windows), we read
        all bytes currently in the OS buffer in a single ``read()`` call and
        split by ``\\n`` ourselves.  This reduces syscall overhead by ~50× and
        eliminates the growing lag that caused RC data to drift off-screen.
        """
        import serial
        while self._running:
            try:
                _log.info("RC: connecting to %s @%d baud", self._port, _BAUD)
                # timeout=0.05 s: blocks up to 50 ms waiting for the first byte,
                # then returns immediately with whatever is in the buffer.
                with serial.Serial(self._port, _BAUD, timeout=0.05) as ser:
                    ser.reset_input_buffer()
                    self._anchor_wall = None
                    self._prev_dev    = None
                    self._elapsed_us  = 0
                    self._set_connected(True)
                    self._disconnect_logged = False
                    _log.info("RC: connected to %s", self._port)

                    buf = bytearray()
                    while self._running:
                        # Read everything currently in the OS buffer (or wait up
                        # to 50 ms for the first byte when the buffer is empty).
                        waiting = ser.in_waiting
                        chunk   = ser.read(waiting if waiting > 0 else 1)
                        if not chunk:
                            continue
                        ts = time.time()   # one timestamp per read burst
                        buf.extend(chunk)

                        # Process all complete lines in the accumulation buffer.
                        while b"\n" in buf:
                            idx  = buf.index(b"\n")
                            line = bytes(buf[: idx + 1])
                            buf  = buf[idx + 1 :]
                            self._parse(line, ts)

            except Exception as exc:
                self._set_connected(False)
                if self._running and not self._disconnect_logged:
                    _log.warning("RC %s error: %s — retrying in 2 s", self._port, exc)
                    self._disconnect_logged = True
                if self._running:
                    time.sleep(2.0)
        self._set_connected(False)

    def _parse(self, raw: bytes, ts_now: float) -> None:
        """Parse one CSV line.

        ts_now  — time.time() captured right after readline(); used ONLY for the
                  first-packet anchor.  All subsequent timestamps are derived from
                  the ESP32 device clock so that Windows thread-scheduling jitter
                  does not accumulate into drift.
        """
        try:
            parts = raw.decode("ascii", errors="ignore").strip().split(",")
            if len(parts) < 9:
                return
            ts_us = int(parts[0])
            chs   = [int(x) for x in parts[1:9]]
        except (ValueError, IndexError):
            return

        # ── Anchor / rollover-safe elapsed time ───────────────────────────
        if self._anchor_wall is None:
            # First fresh packet after buffer flush — set absolute anchor.
            self._anchor_wall = ts_now
            self._prev_dev    = ts_us
            self._elapsed_us  = 0
        else:
            # Inter-packet delta in device µs; handle micros() rollover at 2^32.
            delta = ts_us - self._prev_dev               # type: ignore[operator]
            if delta < 0:
                delta += _US_ROLLOVER                    # rollover (~71 min)
            # Sanity: ignore glitches > 1 s (corrupt packet)
            if delta < 1_000_000:
                self._elapsed_us += delta
            self._prev_dev = ts_us

        ts_wall = self._anchor_wall + self._elapsed_us / 1_000_000.0

        self._seq += 1
        frame = {
            "seq":          self._seq,
            "ts_wall":      ts_wall,
            "ts_device_us": ts_us,
            "ch1": chs[0], "ch2": chs[1], "ch3": chs[2], "ch4": chs[3],
            "ch5": chs[4], "ch6": chs[5], "ch7": chs[6], "ch8": chs[7],
        }
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

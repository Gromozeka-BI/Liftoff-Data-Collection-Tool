"""System event logger — parseable JSONL file with timestamped events.

Each line is a JSON object: {"ts_wall": 1234.5, "event": "...", ...extra_fields}
File: <session_dir>/system.jsonl
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from dct.log import get_logger

_log = get_logger("system_logger")


class SystemLogger:
    def __init__(self, session_dir: Path):
        self._path = session_dir / "system.jsonl"
        self._lock = threading.Lock()
        try:
            self._f = open(self._path, "a", encoding="utf-8")
        except Exception as e:
            _log.error("SystemLogger cannot open %s: %s", self._path, e)
            self._f = None
        self.log("session_start")

    def log(self, event: str, **details) -> None:
        entry = {"ts_wall": time.time(), "event": event}
        entry.update(details)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            if self._f:
                try:
                    self._f.write(line)
                    self._f.flush()
                except Exception as e:
                    _log.error("SystemLogger write error: %s", e)

    def close(self) -> None:
        self.log("session_end")
        with self._lock:
            if self._f:
                try:
                    self._f.close()
                except Exception:
                    pass
                self._f = None

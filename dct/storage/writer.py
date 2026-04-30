"""Streaming Parquet writers for telemetry and events."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dct.storage.schema import TELEMETRY_SCHEMA, EVENTS_SCHEMA
from dct.log import get_logger

_log = get_logger("writer")


class StreamingParquetWriter:
    """Buffers rows in memory and flushes row groups to disk periodically.

    A background thread triggers time-based flushes even when data stops flowing.
    """

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        flush_rows: int = 500,
        flush_interval: float = 2.0,
    ):
        self._path = path
        self._schema = schema
        self._flush_rows = flush_rows
        self._flush_interval = flush_interval
        self._buf: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._writer: pq.ParquetWriter | None = None
        self._last_flush = time.monotonic()
        self.total_rows = 0
        self._bg = threading.Thread(target=self._bg_flush, daemon=True)
        self._bg.start()

    def _bg_flush(self) -> None:
        while True:
            time.sleep(self._flush_interval / 2)
            with self._lock:
                if time.monotonic() - self._last_flush >= self._flush_interval:
                    self._flush_locked()

    def _open(self) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._path,
                self._schema,
                compression="snappy",
            )
            _log.debug("Opened parquet writer: %s", self._path)

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(row)
            if (
                len(self._buf) >= self._flush_rows
                or time.monotonic() - self._last_flush >= self._flush_interval
            ):
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        self._open()
        arrays = {f.name: [] for f in self._schema}
        for row in self._buf:
            for f in self._schema:
                arrays[f.name].append(row.get(f.name))
        table = pa.table(
            {name: pa.array(vals, type=self._schema.field(name).type)
             for name, vals in arrays.items()},
            schema=self._schema,
        )
        n = len(self._buf)
        self._writer.write_table(table)
        self.total_rows += n
        self._buf.clear()
        self._last_flush = time.monotonic()
        _log.debug("Flushed %d rows to %s (total=%d)", n, self._path.name, self.total_rows)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._writer:
                self._writer.close()
                self._writer = None
                _log.info("Closed parquet writer: %s (total_rows=%d)", self._path.name, self.total_rows)


class TelemetryWriter(StreamingParquetWriter):
    def __init__(self, session_dir: Path, flush_rows: int = 500, flush_interval: float = 2.0):
        super().__init__(
            session_dir / "telemetry.parquet",
            TELEMETRY_SCHEMA,
            flush_rows,
            flush_interval,
        )


class EventsWriter(StreamingParquetWriter):
    _seq = 0

    def __init__(self, session_dir: Path):
        super().__init__(session_dir / "events.parquet", EVENTS_SCHEMA, flush_rows=1, flush_interval=0)

    def write_event(
        self,
        event_type: str,
        ts_wall: float,
        gate_id: int | None = None,
        lap_num: int | None = None,
        source: str = "",
    ) -> None:
        EventsWriter._seq += 1
        self.write({
            "seq": EventsWriter._seq,
            "ts_wall": ts_wall,
            "event_type": event_type,
            "gate_id": gate_id if gate_id is not None else -1,
            "lap_num": lap_num if lap_num is not None else -1,
            "source": source,
        })

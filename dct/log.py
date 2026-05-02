"""Centralized logging configuration for DCT."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.DEBUG) -> None:
    """Configure root DCT logger with file + console handlers.

    Call once at application entry point (CLI or GUI). Subsequent calls are
    no-ops if handlers are already attached, so it's safe to call from tests.
    """
    root = logging.getLogger("dct")
    if root.handlers:
        return

    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above to keep terminal readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — DEBUG and above, one new file per run (timestamp in name)
    import datetime as _dt
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"dct_{ts}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'dct' namespace."""
    return logging.getLogger(f"dct.{name}")

"""Persistent UI settings — saved to ui_settings.json between restarts."""
from __future__ import annotations

import json
from pathlib import Path

_PATH = Path("ui_settings.json")


def load() -> dict:
    """Return settings dict; empty dict on any error."""
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data: dict) -> None:
    """Write settings dict to disk; silently ignore errors."""
    try:
        _PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def update(key: str, value) -> None:
    """Load → set one key → save."""
    d = load()
    d[key] = value
    save(d)

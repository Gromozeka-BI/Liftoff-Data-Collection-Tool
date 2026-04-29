"""Session directory creation and metadata management."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dct.config import settings


def _next_session_index(base: Path, prefix: str) -> int:
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    indices = []
    for d in existing:
        parts = d.name.split("_session-")
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices, default=0) + 1


def create_session(pilot: str, drone: str, track: str, purpose: str) -> Path:
    base = settings.sessions_dir
    base.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    prefix = f"{date_str}_pilot-{pilot}_drone-{drone}_track-{track}"
    idx = _next_session_index(base, prefix)
    session_dir = base / f"{prefix}_session-{idx:03d}"
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": "0.1",
        "session_id": session_dir.name,
        "pilot": pilot,
        "drone": drone,
        "track": track,
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_s": None,
        "total_packets": 0,
        "total_laps": 0,
        "validated": False,
    }
    _write_meta(session_dir, meta)
    return session_dir


def load_meta(session_dir: Path) -> dict[str, Any]:
    with open(session_dir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(session_dir: Path, meta: dict[str, Any]) -> None:
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def finalize_meta(session_dir: Path, total_packets: int, total_laps: int, start_time: float) -> None:
    meta = load_meta(session_dir)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["duration_s"] = round(time.time() - start_time, 2)
    meta["total_packets"] = total_packets
    meta["total_laps"] = total_laps
    _write_meta(session_dir, meta)


def copy_track(session_dir: Path, track_path: Path) -> None:
    if track_path.exists():
        shutil.copy2(track_path, session_dir / "track.json")


def load_track(session_dir: Path) -> dict[str, Any] | None:
    p = session_dir / "track.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

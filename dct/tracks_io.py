"""Track folder layout helpers and one-shot legacy migration.

New layout (preferred):
    tracks/
      <track_id>/
        track.json                 — описание трассы
        references/
          default.npz              — эталон для OnlineLocalizer
          default.meta.json        — метаданные эталона

Legacy layout (still readable):
    tracks/<track_id>.json
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_log = logging.getLogger(__name__)

TRACKS_ROOT = Path("tracks")


@dataclass
class TrackEntry:
    track_id: str
    name: str
    path: Path
    data: dict[str, Any]

    @property
    def references_dir(self) -> Path:
        return references_dir(self.track_id)


def tracks_root() -> Path:
    return TRACKS_ROOT


def track_dir(track_id: str) -> Path:
    return TRACKS_ROOT / track_id


def track_json_path(track_id: str) -> Path:
    """Path to track.json — works for new layout only.

    For legacy single-file tracks pass the full path obtained via :func:`iter_tracks`.
    """
    return TRACKS_ROOT / track_id / "track.json"


def references_dir(track_id: str) -> Path:
    return TRACKS_ROOT / track_id / "references"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to read track JSON %s: %s", path, exc)
        return None


def iter_tracks() -> list[TrackEntry]:
    """Return all tracks regardless of layout, sorted by track_id.

    Both ``tracks/<id>/track.json`` and legacy ``tracks/<id>.json`` are recognised.
    Duplicates (same track_id) are de-duped, new layout wins.
    """
    if not TRACKS_ROOT.exists():
        return []

    found: dict[str, TrackEntry] = {}

    for child in TRACKS_ROOT.iterdir():
        if child.is_dir():
            tj = child / "track.json"
            if tj.exists():
                data = _read_json(tj) or {}
                tid = str(data.get("id") or child.name)
                name = str(data.get("name") or tid)
                found[tid] = TrackEntry(track_id=tid, name=name, path=tj, data=data)
        elif child.is_file() and child.suffix.lower() == ".json":
            data = _read_json(child) or {}
            tid = str(data.get("id") or child.stem)
            if tid in found:
                continue
            name = str(data.get("name") or tid)
            found[tid] = TrackEntry(track_id=tid, name=name, path=child, data=data)

    return sorted(found.values(), key=lambda t: t.track_id)


def find_by_name(name: str) -> TrackEntry | None:
    for t in iter_tracks():
        if t.name == name or t.track_id == name:
            return t
    return None


def find_by_id(track_id: str) -> TrackEntry | None:
    for t in iter_tracks():
        if t.track_id == track_id:
            return t
    return None


def migrate_legacy_tracks() -> int:
    """Move ``tracks/<id>.json`` -> ``tracks/<id>/track.json``.

    Idempotent: if a folder ``tracks/<id>`` already exists with ``track.json``
    we simply skip. Returns the number of files migrated.
    """
    if not TRACKS_ROOT.exists():
        return 0

    migrated = 0
    for child in list(TRACKS_ROOT.iterdir()):
        if not (child.is_file() and child.suffix.lower() == ".json"):
            continue

        track_id = child.stem
        new_dir = TRACKS_ROOT / track_id
        new_path = new_dir / "track.json"

        if new_path.exists():
            # already migrated; remove the legacy duplicate carefully
            try:
                child.unlink()
            except Exception:
                pass
            continue

        try:
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(new_path))
            migrated += 1
            _log.info("Migrated track %s -> %s", child.name, new_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to migrate %s: %s", child, exc)
    return migrated


def list_reference_profiles(track_id: str) -> list[dict[str, Any]]:
    """List reference profiles for a track.

    Each entry: {"name", "npz", "meta", "label", "built_at", ...}
    """
    rdir = references_dir(track_id)
    if not rdir.exists():
        return []
    out = []
    for npz in sorted(rdir.glob("*.npz")):
        meta_path = npz.with_suffix(".meta.json")
        meta = _read_json(meta_path) or {}
        meta.update(
            name=npz.stem,
            npz=str(npz.resolve()),
            meta=str(meta_path.resolve()) if meta_path.exists() else "",
        )
        out.append(meta)
    return out


def default_reference(track_id: str) -> Path | None:
    """Return path to the preferred reference for a track or None.

    Order: explicit ``default.npz`` -> first alphabetic profile.
    """
    rdir = references_dir(track_id)
    if not rdir.exists():
        return None
    candidate = rdir / "default.npz"
    if candidate.exists():
        return candidate
    profiles = list(rdir.glob("*.npz"))
    return sorted(profiles)[0] if profiles else None


def all_track_ids() -> Iterable[str]:
    return [t.track_id for t in iter_tracks()]

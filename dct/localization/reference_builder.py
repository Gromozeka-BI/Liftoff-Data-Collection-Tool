"""High-level facade for building and saving track references inside DCT.

Used by the GUI's "Build reference" dialog and by code that wants to load
the default profile of a track without going through the legacy
``localizer_reference_npz`` ``ui_settings`` key.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dct.localization.lap_loader import (
    Lap,
    laps_summary,
    load_dct_session,
    load_dct_sessions_dir,
)
from dct.localization.online_localizer import Reference
from dct.localization.reference_select import (
    evaluate_reference_quality,
    select_best_reference,
)
from dct import tracks_io

_log = logging.getLogger(__name__)

_VALID_PROFILE_NAME = re.compile(r"[A-Za-z0-9_\-]{1,40}")


@dataclass
class ReferenceInfo:
    """Lightweight description of a saved reference file."""

    track_id: str
    profile: str
    npz_path: Path
    meta_path: Path
    meta: dict = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.profile

    @property
    def built_at(self) -> str:
        return str(self.meta.get("built_at", ""))

    @property
    def source(self) -> str:
        return str(self.meta.get("source", ""))

    @property
    def lap_index(self) -> int | None:
        v = self.meta.get("lap_index")
        return int(v) if v is not None else None


def list_laps(source: str | Path) -> tuple[list[Lap], dict, list[dict]]:
    """Load laps from either a single session folder or a folder of sessions.

    Returns ``(laps, track_dict, summary)``.
    """
    src = Path(source)
    if (src / "telemetry.parquet").exists():
        laps, track = load_dct_session(src)
    else:
        laps, track = load_dct_sessions_dir(src)
    return laps, track, laps_summary(laps)


def auto_pick(laps: list[Lap], *, smooth_w: int = 5, progress_cb=None) -> int:
    """Pick the best lap index (0-based) by LOO NN-greedy."""
    best, _scores = select_best_reference(
        laps, smooth_w=smooth_w, progress_cb=progress_cb,
    )
    return best


def build(lap: Lap, *, smooth_w: int = 5) -> Reference:
    """Build a :class:`Reference` from a single lap."""
    return Reference.build(
        t=lap.t.copy(),
        sticks=lap.sticks.copy(),
        pos=lap.pos.copy(),
        smooth_w=smooth_w,
    )


def _validate_profile_name(name: str) -> str:
    name = (name or "default").strip()
    if not _VALID_PROFILE_NAME.fullmatch(name):
        raise ValueError(
            f"Недопустимое имя профиля '{name}'. "
            "Разрешены латиница, цифры, '-', '_'.",
        )
    return name


def save_for_track(
    ref: Reference,
    *,
    track_id: str,
    profile: str = "default",
    source: str | Path = "",
    lap_index: int | None = None,
    metrics: dict | None = None,
    smooth_w: int | None = None,
) -> Path:
    """Persist ``ref`` to ``tracks/<track_id>/references/<profile>.npz`` + meta.

    Returns the path of the saved ``.npz``.
    """
    profile = _validate_profile_name(profile)
    rdir = tracks_io.references_dir(track_id)
    rdir.mkdir(parents=True, exist_ok=True)

    npz_path = rdir / f"{profile}.npz"
    meta_path = npz_path.with_suffix(".meta.json")

    ref.save(npz_path)

    meta = {
        "track_id": track_id,
        "profile": profile,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source) if source else "",
        "lap_index": lap_index,
        "smooth_w": smooth_w if smooth_w is not None else int(ref.smooth_w),
        "track_length_m": float(ref.L),
        "frames": int(ref.sticks_norm.shape[0]),
        "metrics": metrics or {},
    }
    try:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to write meta for %s: %s", npz_path, exc)
    return npz_path


def find_for_track(track_id: str) -> list[ReferenceInfo]:
    rdir = tracks_io.references_dir(track_id)
    if not rdir.exists():
        return []
    out: list[ReferenceInfo] = []
    for npz in sorted(rdir.glob("*.npz")):
        meta_path = npz.with_suffix(".meta.json")
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append(
            ReferenceInfo(
                track_id=track_id,
                profile=npz.stem,
                npz_path=npz.resolve(),
                meta_path=meta_path.resolve(),
                meta=meta,
            ),
        )
    return out


def default_for_track(track_id: str) -> Path | None:
    """Return preferred reference for the track or ``None``."""
    return tracks_io.default_reference(track_id)


def evaluate_quality(
    ref_lap: Lap,
    other_laps: list[Lap],
    *,
    smooth_w: int = 5,
) -> float:
    """Re-export so callers don't depend on :mod:`reference_select` directly."""
    return evaluate_reference_quality(
        ref_lap, other_laps, smooth_w=smooth_w,
    )

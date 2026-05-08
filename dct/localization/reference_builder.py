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
    _is_rc_only_session,
    laps_summary,
    load_dct_session,
    load_dct_sessions_dir,
    load_rc_only_session,
)
from dct.localization.online_localizer import Reference
from dct.rate_features import FEATURE_BETAFLIGHT_CLASSIC_V1, physical_observation_matrix
from dct.localization.reference_select import (
    evaluate_reference_quality,
    select_best_reference,
)
from dct import tracks_io

_log = logging.getLogger(__name__)

_VALID_PROFILE_NAME = re.compile(r"[A-Za-z0-9_\-]{1,40}")

_LEGACY_STICKS_SUFFIX = "_legacy_sticks"

# Maps GUI invert-state keys (e.g. "in_roll") → column index in STICK_COLS
_INVERT_KEY_TO_COL: dict[str, int] = {
    "in_throttle": 0,
    "in_yaw":      1,
    "in_pitch":    2,
    "in_roll":     3,
}


def _apply_invert_lf(sticks: np.ndarray, invert_lf: dict | None) -> np.ndarray:
    """Return a copy of *sticks* with the GUI LF invert settings applied.

    ``invert_lf`` uses the same key convention as
    ``StickGraphsWidget.get_invert_state()["lf"]``:
    ``{"in_throttle": bool, "in_yaw": bool, "in_pitch": bool, "in_roll": bool}``.
    Channels with ``True`` are sign-flipped.  Returns ``sticks`` unchanged when
    ``invert_lf`` is *None* or empty.
    """
    if not invert_lf:
        return sticks
    s = sticks.copy()
    for key, col in _INVERT_KEY_TO_COL.items():
        if invert_lf.get(key, False):
            s[:, col] = -s[:, col]
    return s


def legacy_sticks_profile_name(profile: str) -> str:
    """Имя профиля для sidecar-эталона (сырые стики)."""
    p = (profile or "default").strip() or "default"
    if p.endswith(_LEGACY_STICKS_SUFFIX):
        return p
    return f"{p}{_LEGACY_STICKS_SUFFIX}"


def legacy_sticks_npz_path(primary_npz: Path) -> Path:
    """Sidecar эталона: те же t/pos, наблюдения — сырые стики (legacy PF).

    Имя: ``{stem(primary)}_legacy_sticks.npz`` для ``default.npz`` →
    ``default_legacy_sticks.npz``. Создаётся при сборке эталона с Betaflight
    фичами (см. Reference build dialog).
    """
    stem = primary_npz.stem
    if stem.endswith(_LEGACY_STICKS_SUFFIX):
        return primary_npz
    return primary_npz.with_name(f"{stem}{_LEGACY_STICKS_SUFFIX}.npz")


def npz_feature_kind(path: Path) -> str | None:
    """Значение ``feature_kind`` из .npz (строка) или ``None`` для старых эталонов."""
    with np.load(path, allow_pickle=False) as d:
        if "feature_kind" not in d.files:
            return None
        fk = d["feature_kind"]
        s = fk.item() if fk.ndim == 0 else str(fk.flat[0])
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        s = str(s).strip()
        return s or None


def resolve_bf_and_legacy_npz(selected: Path) -> tuple[Path | None, Path | None]:
    """Пара путей ``(betaflight_npz, legacy_npz)`` для одного логического профиля.

    В комбо можно выбрать либо ``default.npz``, либо ``default_legacy_sticks.npz`` —
    оба варианта подключают те же два файла, если они есть на диске.
    """
    selected = selected.resolve()
    stem = selected.stem
    parent = selected.parent
    if stem.endswith(_LEGACY_STICKS_SUFFIX):
        base_stem = stem[: -len(_LEGACY_STICKS_SUFFIX)]
        p_leg = selected if selected.is_file() else None
        p_bf = parent / f"{base_stem}.npz"
        if not p_bf.is_file():
            p_bf = None
        return (p_bf, p_leg)
    p_bf = selected if selected.is_file() else None
    p_leg = legacy_sticks_npz_path(selected)
    if not p_leg.is_file():
        p_leg = None
    return (p_bf, p_leg)


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

    Supports three layouts:
    * Single Liftoff session (``telemetry.parquet`` present)
    * RC-only session (``rc_channels.parquet`` present, no telemetry)
    * Folder of sessions (auto-detects the above per sub-folder)

    Returns ``(laps, track_dict, summary)``.
    """
    src = Path(source)
    if (src / "telemetry.parquet").exists():
        laps, track = load_dct_session(src)
    elif _is_rc_only_session(src):
        laps, track = load_rc_only_session(src)
    else:
        laps, track = load_dct_sessions_dir(src)
    return laps, track, laps_summary(laps)


def auto_pick(laps: list[Lap], *, smooth_w: int = 5, progress_cb=None) -> int:
    """Pick the best lap index (0-based) by LOO NN-greedy."""
    best, _scores = select_best_reference(
        laps, smooth_w=smooth_w, progress_cb=progress_cb,
    )
    return best


def build(
    lap: Lap,
    *,
    smooth_w: int = 5,
    rate_profile: dict | None = None,
    invert_lf: dict | None = None,
) -> Reference:
    """Build a :class:`Reference` from a single lap.

    ``invert_lf`` — GUI LF reverse-checkbox state (keys ``in_throttle`` …
    ``in_roll``).  When provided, the same sign-flips that the live localizer
    applies to incoming Liftoff telemetry are applied here so that the
    reference feature vectors share the same convention.  Pass the value of
    ``StickGraphsWidget.get_invert_state()["lf"]`` from the GUI.

    If ``rate_profile`` is omitted, loads ``<session>/rate_profile.json`` when
    ``lap.session_dir`` is set and the file exists. Otherwise falls back to
    legacy stick-based observations.
    """
    # invert_lf corrects Liftoff's Roll sign-flip; RC sticks need no correction.
    effective_invert = invert_lf if getattr(lap, "data_source", "liftoff") == "liftoff" else None
    sticks = _apply_invert_lf(lap.sticks, effective_invert)

    prof = rate_profile
    if prof is None and lap.session_dir is not None:
        rp = lap.session_dir / "rate_profile.json"
        if rp.is_file():
            try:
                prof = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                _log.warning("Could not read rate profile: %s", rp)
                prof = None

    if isinstance(prof, dict) and prof.get("model") == "betaflight":
        obs = physical_observation_matrix(sticks, prof)
        return Reference.build_from_features(
            lap.t.copy(),
            obs,
            lap.pos.copy(),
            smooth_w=smooth_w,
            feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
            rate_profile=prof,
        )

    return Reference.build(
        t=lap.t.copy(),
        sticks=sticks,
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
        "feature_kind": getattr(ref, "feature_kind", None) or "legacy_sticks",
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

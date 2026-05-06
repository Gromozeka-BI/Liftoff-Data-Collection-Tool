"""Load DCT sessions and split telemetry into laps for reference building.

Ported from ``stick_localizer_bench/data_io.py`` and adapted to the DCT
on-disk layout (``telemetry.parquet`` + ``events.parquet`` + ``track.json``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

STICK_COLS = ["input_throttle", "input_yaw", "input_pitch", "input_roll"]
POS_COLS = ["position_x", "position_y", "position_z"]

_DCT_RENAME = {
    "ts_wall": "timestamp",
    "pos_x": "position_x",
    "pos_y": "position_y",
    "pos_z": "position_z",
    "in_throttle": "input_throttle",
    "in_yaw": "input_yaw",
    "in_pitch": "input_pitch",
    "in_roll": "input_roll",
}


@dataclass
class Lap:
    """A single lap pulled out of a recorded session."""

    index: int
    t: np.ndarray
    sticks: np.ndarray
    pos: np.ndarray
    duration: float
    source_session: str = ""

    def __len__(self) -> int:
        return len(self.t)


def _read_track(path: Path) -> dict:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_start_finish_xyz(track: dict) -> np.ndarray:
    for g in track.get("gates", []):
        if g.get("is_start_finish"):
            return np.array(g["position"], dtype=float)
    return np.array(track["gates"][0]["position"], dtype=float)


def _split_laps_by_events(df, events, *, min_lap_duration: float = 3.0) -> list[Lap]:
    """Split telemetry rows into laps using DCT lap/finish events."""
    lap_events = events[
        events["event_type"].isin({"lap", "rh_lap", "button_lap", "session_stop"})
    ].sort_values("ts_wall")
    boundaries = lap_events["ts_wall"].to_numpy()
    if len(boundaries) < 2:
        raise RuntimeError(
            f"Недостаточно событий круга в events.parquet (найдено {len(boundaries)}). "
            "Убедитесь, что в сессии отмечались круги (LAP-события).",
        )

    ts = df["timestamp"].to_numpy()
    laps: list[Lap] = []
    for i in range(len(boundaries) - 1):
        t_start, t_end = boundaries[i], boundaries[i + 1]
        mask = (ts >= t_start) & (ts < t_end)
        sub = df[mask]
        if len(sub) < 2:
            continue
        t = sub["timestamp"].to_numpy(dtype=float)
        if t[-1] - t[0] < min_lap_duration:
            continue
        laps.append(
            Lap(
                index=len(laps) + 1,
                t=t,
                sticks=sub[STICK_COLS].to_numpy(dtype=float),
                pos=sub[POS_COLS].to_numpy(dtype=float),
                duration=float(t[-1] - t[0]),
            ),
        )
    return laps


def split_into_laps_by_radius(
    df,
    track: dict,
    *,
    cross_radius: float | None = None,
    min_lap_duration: float = 3.0,
) -> list[Lap]:
    """Fallback splitter when no events are recorded — by start/finish radius."""
    if cross_radius is None:
        cross_radius = float(track.get("check_radius", 2.0))
    sf = _get_start_finish_xyz(track)

    pos = df[POS_COLS].to_numpy()
    dist = np.linalg.norm(pos - sf, axis=1)
    inside = dist < cross_radius
    enters = np.where(inside[1:] & ~inside[:-1])[0] + 1
    if len(enters) < 2:
        raise RuntimeError(
            f"Не нашли минимум 2 пересечения старт-финиша (радиус {cross_radius} м). "
            f"Всего точек в зоне: {int(inside.sum())}",
        )

    laps: list[Lap] = []
    for i in range(len(enters) - 1):
        a, b = enters[i], enters[i + 1]
        sub = df.iloc[a:b]
        t = sub["timestamp"].to_numpy()
        if t[-1] - t[0] < min_lap_duration:
            continue
        laps.append(
            Lap(
                index=len(laps) + 1,
                t=t.astype(float),
                sticks=sub[STICK_COLS].to_numpy(dtype=float),
                pos=sub[POS_COLS].to_numpy(dtype=float),
                duration=float(t[-1] - t[0]),
            ),
        )
    return laps


def _lap_track_length(lap: Lap) -> float:
    return float(np.sum(np.linalg.norm(np.diff(lap.pos, axis=0), axis=1)))


def filter_anomalous_laps(
    laps: list[Lap],
    *,
    min_fraction: float = 0.7,
) -> list[Lap]:
    """Iteratively drop incomplete/aborted laps (length < ``min_fraction × median``)."""
    if len(laps) < 3:
        return laps

    remaining = list(laps)
    while True:
        lengths = np.array([_lap_track_length(lap) for lap in remaining])
        threshold = float(np.median(lengths)) * min_fraction
        good = [lap for lap, length in zip(remaining, lengths) if length >= threshold]
        if len(good) == len(remaining):
            break
        remaining = good
        if len(remaining) < 3:
            break
    return remaining


def _is_dct_session(path: Path) -> bool:
    return (path / "telemetry.parquet").exists()


def load_dct_session(
    session_path: str | Path,
    *,
    use_ts_sim: bool = False,
    min_lap_duration: float = 3.0,
) -> tuple[list[Lap], dict]:
    """Load a single DCT session and return ``(laps, track)``."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Для загрузки DCT-сессий требуются pandas и pyarrow.",
        ) from exc

    session_path = Path(session_path)

    telem_path = session_path / "telemetry.parquet"
    events_path = session_path / "events.parquet"
    track_path = session_path / "track.json"

    if not telem_path.exists():
        raise FileNotFoundError(f"Не найден файл телеметрии: {telem_path}")
    if not track_path.exists():
        raise FileNotFoundError(f"Не найден файл трассы: {track_path}")

    df = pd.read_parquet(telem_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    ts_col = "ts_sim" if use_ts_sim else "ts_wall"
    if ts_col not in df.columns:
        raise ValueError(
            f"Колонка '{ts_col}' не найдена в telemetry.parquet. Есть: {list(df.columns)}",
        )

    rename = {ts_col: "timestamp", **{k: v for k, v in _DCT_RENAME.items() if k != "ts_wall"}}
    if not use_ts_sim:
        rename["ts_wall"] = "timestamp"
    df = df.rename(columns=rename)

    needed = ["timestamp", *POS_COLS, *STICK_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"В telemetry.parquet нет колонок: {missing}\nЕсть: {list(df.columns)}",
        )

    df = df[needed].dropna().sort_values("timestamp").reset_index(drop=True)
    track = _read_track(track_path)

    if events_path.exists():
        events = pd.read_parquet(events_path)
        events.columns = [str(c).strip().lower() for c in events.columns]
        try:
            laps = _split_laps_by_events(df, events, min_lap_duration=min_lap_duration)
        except RuntimeError:
            laps = split_into_laps_by_radius(
                df, track, min_lap_duration=min_lap_duration,
            )
    else:
        laps = split_into_laps_by_radius(
            df, track, min_lap_duration=min_lap_duration,
        )

    if not laps:
        raise RuntimeError("Не удалось выделить ни одного круга из сессии DCT.")

    for lap in laps:
        lap.source_session = session_path.name

    return laps, track


def load_dct_sessions_dir(
    dir_path: str | Path,
    *,
    use_ts_sim: bool = False,
    min_lap_duration: float = 3.0,
) -> tuple[list[Lap], dict]:
    """Load every DCT session inside a folder and return combined ``(laps, track)``.

    The track is taken from the first successful session.
    """
    dir_path = Path(dir_path)
    if _is_dct_session(dir_path):
        return load_dct_session(
            dir_path,
            use_ts_sim=use_ts_sim,
            min_lap_duration=min_lap_duration,
        )

    session_dirs = sorted(
        p for p in dir_path.iterdir() if p.is_dir() and _is_dct_session(p)
    )
    if not session_dirs:
        raise RuntimeError(
            f"В папке {dir_path} не найдено ни одной DCT-сессии "
            f"(подпапок с telemetry.parquet).",
        )

    all_laps: list[Lap] = []
    common_track: dict | None = None
    for session_dir in session_dirs:
        try:
            laps, track = load_dct_session(
                session_dir,
                use_ts_sim=use_ts_sim,
                min_lap_duration=min_lap_duration,
            )
        except Exception as exc:  # noqa: BLE001
            _log.info("[skip] %s: %s", session_dir.name, exc)
            continue
        if common_track is None:
            common_track = track
        all_laps.extend(laps)

    if not all_laps:
        raise RuntimeError("Не удалось загрузить ни одного круга из всех сессий.")

    all_laps = filter_anomalous_laps(all_laps)
    for i, lap in enumerate(all_laps, start=1):
        lap.index = i
    return all_laps, common_track or {}


def laps_summary(laps: list[Lap]) -> list[dict]:
    """Compact list of ``{index, duration, length, frames, source}`` for UI."""
    out = []
    for lap in laps:
        out.append(
            {
                "index": int(lap.index),
                "duration": float(lap.duration),
                "length_m": _lap_track_length(lap),
                "frames": int(len(lap)),
                "source": lap.source_session,
            },
        )
    return out

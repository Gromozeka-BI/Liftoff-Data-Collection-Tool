"""Build RedSeep_S2 reference from session-002 (matches test session speed).

Run once from project root:
    python tools/_build_pilot_refs2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.lap_loader import filter_anomalous_laps, load_dct_session
from dct.localization.reference_builder import auto_pick, build, save_for_track

TRACK_ID = "track-002"
SMOOTH_W  = 5


def _load_invert_lf(session_dir: Path) -> dict | None:
    import json
    inv_path = session_dir / "invert.json"
    if inv_path.exists():
        data = json.loads(inv_path.read_text(encoding="utf-8"))
        return data.get("lf")
    return None


def _build_ref(session_dir: Path, profile_name: str) -> None:
    from dct import tracks_io
    ref_path = tracks_io.references_dir(TRACK_ID) / f"{profile_name}.npz"

    invert_lf = _load_invert_lf(session_dir)
    print(f"\nBuilding '{profile_name}' from: {session_dir.name}")
    print(f"  invert_lf: {invert_lf}")
    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    print(f"  Good laps: {len(laps)}")
    for lap in laps:
        print(f"    Lap {lap.index}: {lap.duration:.1f} s")

    best_idx = auto_pick(laps, smooth_w=SMOOTH_W)
    best_lap = laps[best_idx]
    print(f"  => Best lap: index={best_lap.index}, duration={best_lap.duration:.1f} s")

    ref = build(best_lap, smooth_w=SMOOTH_W, invert_lf=invert_lf)
    path = save_for_track(
        ref,
        track_id=TRACK_ID,
        profile=profile_name,
        source=str(session_dir),
        lap_index=best_lap.index,
        smooth_w=SMOOTH_W,
    )
    print(f"  Saved -> {path}")
    print(f"  Track length: {ref.L:.1f} m, frames: {ref.sticks_norm.shape[0]}")
    return best_lap.index  # return to know which lap to skip in test


def main() -> None:
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")

    # RedSeep_S2 — from Part_9 / session-002 (same session as test!)
    part9 = liftoff / "Part_9"
    sess = sorted([d for d in part9.iterdir()
                   if "RedSeep" in d.name and "track-002" in d.name and "session-002" in d.name])[0]
    _build_ref(sess, "RedSeep_S2")

    print("\nDone. RedSeep_S2 reference built from session-002.")
    print("Use this as the same-pilot reference for RedSeep test (session-002).")


if __name__ == "__main__":
    main()

"""Build references for Experiment 5: RedSeep_1 and PlatOnAir_1.

Run once from project root:
    python tools/_build_pilot_refs.py
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


def _build_ref(
    session_dir: Path,
    profile_name: str,
    *,
    force: bool = False,
) -> None:
    from dct import tracks_io
    ref_path = tracks_io.references_dir(TRACK_ID) / f"{profile_name}.npz"
    if ref_path.exists() and not force:
        print(f"[SKIP] {profile_name}.npz already exists — pass force=True to rebuild")
        return

    invert_lf = _load_invert_lf(session_dir)
    print(f"\nBuilding '{profile_name}' from: {session_dir.name}")
    print(f"  invert_lf: {invert_lf}")
    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    print(f"  Good laps: {len(laps)}")
    for lap in laps:
        print(f"    Lap {lap.index}: {lap.duration:.1f} s")

    print("  Auto-picking best lap (LOO greedy) ...")
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


def main() -> None:
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")

    # RedSeep_1 — from Part_9 / session-001 (13 clean laps, ~21 s each)
    part9 = liftoff / "Part_9"
    redseep_sessions = sorted(
        [d for d in part9.iterdir()
         if "RedSeep" in d.name and "track-002" in d.name and "session-001" in d.name]
    )
    if not redseep_sessions:
        print("ERROR: RedSeep track-002 session-001 not found in Part_9")
        sys.exit(1)
    _build_ref(redseep_sessions[0], "RedSeep_1", force=True)

    # PlatOnAir_1 — from Part_10 / session-027 (7 consistent laps, ~17 s each)
    part10 = liftoff / "Part_10"
    platair_sessions = sorted(
        [d for d in part10.iterdir()
         if "PlatOnAir" in d.name and "track-002" in d.name and "session-027" in d.name]
    )
    if not platair_sessions:
        print("ERROR: PlatOnAir track-002 session-027 not found in Part_10")
        sys.exit(1)
    _build_ref(platair_sessions[0], "PlatOnAir_1", force=True)

    print("\nAll references ready.")


if __name__ == "__main__":
    main()

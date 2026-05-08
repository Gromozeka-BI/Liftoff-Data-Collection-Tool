"""Experiment 5: Cross-Pilot Generalization.

STATUS: WAITING FOR DATA
========================
This script requires flight sessions from a second pilot on track-002.
See data_collection_spec.md for the pilot brief.

Research question
-----------------
Can the localizer built from Pilot 1's reference (GromFF_1) accurately track
a different pilot's (Pilot 2) flights? Does the pilot's flying style affect
localization accuracy more or less than drone type?

Context from Experiments 1–4
-----------------------------
- Drone type: dominant factor (eta2=0.41, Exp 1)
- RC+Rate normalization handles rate profile differences well
- Reference quality > drone type (Exp 4): GromFF_1 cross-drone p90=9.7m,
  better than same-drone LiftOff200_Grom_1 p90=15.9m
- Best same-drone p90: 5.8m (Exp 4)
- Optimal hyperparams: obs_sigma=2.0, pnv=8.0 (Exp 3)

Hypotheses
----------
H1: Cross-pilot p90 < cross-drone p90 (9.7m) — pilot style less impactful than drone type
H2: RC+Rate normalization handles pilot differences too
H3: Same-pilot ceiling < 5.8m (Exp 4 best result)
H4: Cross-pilot gap is symmetric between pilots

Experimental design
-------------------
References:
  GromFF_1       — Pilot 1 (Gromozeka),  MadTrainer, Gromozeka_rate [EXISTS]
  <Callsign>_1   — Pilot 2 (<Callsign>), MadTrainer, Gromozeka_rate [NEEDED]
  <Callsign>_2   — Pilot 2 (<Callsign>), MadTrainer, Gromozeka_rate [NEEDED, LOO check]

Test flights:
  G-F1: Pilot 1, MadTrainer, Gromozeka_rate  [EXISTS]
  G-F2: Pilot 1, MadTrainer, RedSheep_rate   [EXISTS]
  P2-S1: Pilot 2, MadTrainer, Gromozeka_rate [NEEDED]
  P2-S3: Pilot 2, MadTrainer, Own rate       [NEEDED]

Condition types:
  same_pilot_same_drone_same_rate  — baseline ceiling
  cross_pilot_same_drone_same_rate — KEY: pilot generalization
  cross_pilot_same_drone_cross_rate — pilot + rate mismatch
  cross_pilot_reversed             — symmetry check

Fixed hyperparameters (optimal from Exp 3):
  obs_sigma = 2.0
  process_noise_v = 8.0
  process_noise_s = 1.5
  Mode = RC+Rate

Usage (from project root):
    python tools/experiment_pilot.py

TODO (fill in when Pilot 2 data arrives)
-----------------------------------------
1. Replace PILOT2_CALLSIGN with the actual callsign
2. Replace PILOT2_DATA_BASE with the actual data path
3. Add Pilot 2 sessions to PILOT2_FLIGHTS list
4. Add Pilot 2's own rate profile name to PILOT2_OWN_RATE
5. Ensure reference files exist in tracks/track-002/references/
6. Run and verify outputs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── TODO: Fill in when data arrives ───────────────────────────────────────────

# Pilot 2 identity
PILOT2_CALLSIGN = "UNKNOWN"       # e.g. "RedSeep"
PILOT2_OWN_RATE = "UNKNOWN_rate"  # e.g. "RedSheep_rate"

# Base directory for Pilot 2 session data
# e.g. Path(r"D:\DroneTrackerDB\Liftoff\Part_6")
PILOT2_DATA_BASE: Path | None = None

# Pilot 2 session folder names (relative to PILOT2_DATA_BASE)
PILOT2_FLIGHTS: list[dict] = [
    # dict(
    #     flight_id="P2-S1",
    #     pilot=PILOT2_CALLSIGN,
    #     drone="MadTrainer",
    #     rate="Gromozeka_rate",
    #     session="2026-XX-XX_pilot-<Callsign>_drone-MadTrainer_track-track-002_session-001",
    # ),
    # dict(
    #     flight_id="P2-S2",
    #     pilot=PILOT2_CALLSIGN,
    #     drone="MadTrainer",
    #     rate="Gromozeka_rate",
    #     session="2026-XX-XX_pilot-<Callsign>_drone-MadTrainer_track-track-002_session-002",
    # ),
    # dict(
    #     flight_id="P2-S3",
    #     pilot=PILOT2_CALLSIGN,
    #     drone="MadTrainer",
    #     rate=PILOT2_OWN_RATE,
    #     session="2026-XX-XX_pilot-<Callsign>_drone-MadTrainer_track-track-002_session-003",
    # ),
    # dict(
    #     flight_id="P2-S4",
    #     pilot=PILOT2_CALLSIGN,
    #     drone="MadTrainer",
    #     rate=PILOT2_OWN_RATE,
    #     session="2026-XX-XX_pilot-<Callsign>_drone-MadTrainer_track-track-002_session-004",
    # ),
]

# Reference files for Pilot 2 (in tracks/track-002/references/)
PILOT2_REF_NAME_1 = f"{PILOT2_CALLSIGN}_1"   # e.g. "RedSeep_1"
PILOT2_REF_NAME_2 = f"{PILOT2_CALLSIGN}_2"   # e.g. "RedSeep_2"

# ── Existing data (Pilot 1 — Gromozeka) ──────────────────────────────────────

def _find_part5() -> Path:
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")
    candidates = sorted([d for d in liftoff.iterdir() if d.name.startswith("Part_5")])
    if not candidates:
        raise FileNotFoundError(f"No Part_5* directory found under {liftoff}")
    return candidates[0]


PART5 = _find_part5()
REF_DIR = ROOT / "tracks" / "track-002" / "references"

PILOT1_FLIGHTS: list[dict] = [
    dict(
        flight_id="G-F1", pilot="Gromozeka", drone="MadTrainer", rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
        base=PART5,
    ),
    dict(
        flight_id="G-F2", pilot="Gromozeka", drone="MadTrainer", rate="RedSheep_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-002",
        base=PART5,
    ),
]

REFERENCES: list[dict] = [
    dict(name="GromFF_1", path=REF_DIR / "GromFF_1.npz",
         pilot="Gromozeka", drone="MadTrainer", rate="Gromozeka_rate"),
    # Pilot 2 references — add when available:
    # dict(name=PILOT2_REF_NAME_1, path=REF_DIR / f"{PILOT2_REF_NAME_1}.npz",
    #      pilot=PILOT2_CALLSIGN, drone="MadTrainer", rate="Gromozeka_rate"),
    # dict(name=PILOT2_REF_NAME_2, path=REF_DIR / f"{PILOT2_REF_NAME_2}.npz",
    #      pilot=PILOT2_CALLSIGN, drone="MadTrainer", rate="Gromozeka_rate"),
]

# ── Hyperparameters (optimal from Experiment 3) ───────────────────────────────

OBS_SIGMA       = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5
JUMP_THRESHOLD_M = 15.0
N_LAST_LAPS     = 3

WEIGHT_PRESETS: list[tuple[str, list[float]]] = [
    ("baseline",       [1.0, 1.0, 1.0, 1.0]),
    ("angular_scaled", [0.0, 0.7, 0.5, 1.0]),
    ("no_thr",         [0.0, 1.0, 1.0, 1.0]),
]

_OUT_DIR = Path(__file__).parent / "results"
_OUT_DIR.mkdir(exist_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if PILOT2_DATA_BASE is None or not PILOT2_FLIGHTS:
        print("=" * 60)
        print("EXPERIMENT 5: WAITING FOR PILOT 2 DATA")
        print("=" * 60)
        print()
        print("To proceed, fill in the following in this script:")
        print(f"  1. PILOT2_CALLSIGN = '<callsign>'  (currently: {PILOT2_CALLSIGN!r})")
        print(f"  2. PILOT2_OWN_RATE = '<rate_name>' (currently: {PILOT2_OWN_RATE!r})")
        print(f"  3. PILOT2_DATA_BASE = Path(...)     (currently: None)")
        print(f"  4. Uncomment sessions in PILOT2_FLIGHTS list")
        print(f"  5. Add reference .npz files to:  {REF_DIR}")
        print()
        print("See tools/exp5_pilot/data_collection_spec.md for the pilot brief.")
        print()
        print("Existing Pilot 1 data check:")
        all_ok = True
        for f in PILOT1_FLIGHTS:
            session_dir = f["base"] / f["session"]
            status = "[OK]" if session_dir.exists() else "[MISSING]"
            print(f"  {f['flight_id']}: {status}  {session_dir.name}")
            if not session_dir.exists():
                all_ok = False
        for r in REFERENCES:
            status = "[OK]" if r["path"].exists() else "[MISSING]"
            print(f"  Ref {r['name']}: {status}")
            if not r["path"].exists():
                all_ok = False
        print()
        if all_ok:
            print("All Pilot 1 data present. Ready to run once Pilot 2 data arrives.")
        else:
            print("WARNING: Some Pilot 1 data is missing!")
        return

    # ── Full experiment logic (to be implemented when data arrives) ───────────
    # This section mirrors experiment_reference.py structure.
    # Steps:
    #   1. Load all flights (_load_flight_raw for each)
    #   2. Load all references
    #   3. Pre-compute lap caches per (ref, flight) pair
    #   4. Run localizer for each (ref, flight, preset) combination
    #   5. Compute condition labels (pilot_match, drone_match, rate_match)
    #   6. Aggregate metrics, generate plots, write report
    #
    # See experiment_reference.py for the complete implementation template.
    raise NotImplementedError(
        "Full implementation pending Pilot 2 data. "
        "See experiment_reference.py for the template."
    )


if __name__ == "__main__":
    main()

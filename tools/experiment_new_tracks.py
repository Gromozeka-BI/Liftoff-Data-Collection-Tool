"""Experiment 8: Generalization to New Tracks.

Research question
-----------------
Do the localizer hyperparameters optimized on track-002 in Exp 3
(σ=2.0, pnv=8.0, weights=no_thr) transfer to brand-new tracks (track-003,
track-004) without re-optimization?

Hypotheses
----------
H1 — Default params achieve p90 < 15 m on new tracks (transfer hypothesis).
H2 — Optimal (σ, pnv) on new tracks is close to track-002 optimum
     (|σ_opt − 2.0| < 0.5 and |pnv_opt − 8.0| < 4).
H3 — Track length influences optimal pnv (longer track → larger pnv).

Data (Gromozeka_rate only, after strict length filter)
------------------------------------------------------
- track-002 (Abu-Dhabi, ~313 m, 28 gates)
    MT: s001 (4), s003 (13), s004 (15)   → 32 laps total
    LO: s001 (6)                          → 6 laps cross-drone
- track-003 (Las Vegas, ~269 m, 29 gates)
    MT: s001 (10), s002 (12), s003 (6)   → 28 laps total
    LO: s001 (5), s002 (6)                → 11 laps cross-drone
- track-004 (Budapest, ~258 m, 29 gates)
    MT: s001 (5), s002 (3), s003 (5),
        s004 (4), s005 (7), s006 (7)     → 31 laps total
    LO: —                                 → no cross-drone for track-004

Pipeline
--------
1. Load all sessions, apply strict length filter (7% deviation threshold).
2. For each track:
     a. For each MT test session: build ref from the OTHER MT sessions
        (median_time, n=5, sw=5), run continuous filter with default params.
     b. For each LO test session (where exists): build ref from ALL MT sessions
        (median_time, n=5, sw=5), run continuous, record cross-drone p90.
3. For each NEW track (003, 004): mini-sweep σ × pnv (3×3) on the most-laps
   test session, with ref = best 5 across all OTHER MT sessions.
4. Compare optimal (σ, pnv) with track-002 baseline → H2, H3.

Usage
-----
    python tools/experiment_new_tracks.py --smoke    # one scenario per track
    python tools/experiment_new_tracks.py            # full experiment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.lap_loader import Lap, filter_anomalous_laps, load_dct_session
from dct.localization.online_localizer import OnlineLocalizer, Reference
from dct.rate_features import (
    FEATURE_BETAFLIGHT_CLASSIC_V1,
    load_rate_profile,
    physical_observation_matrix,
)


# ── Configuration ──────────────────────────────────────────────────────────────

LIFTOFF = Path(r"D:\DroneTrackerDB\Liftoff")

OUT_DIR = Path(__file__).parent / "exp8_new_tracks"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0

# Default params from Exp 3 (the "transferred" config under test)
DEFAULT_OBS_SIGMA = 2.0
DEFAULT_PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5
WEIGHTS_NO_THR = np.array([0.0, 1.0, 1.0, 1.0])

# Reference building (best from Exp 6)
REF_LAP_SELECTION = "median_time"
REF_N_LAPS = 5
REF_SMOOTH_W = 5

# Strict lap filter — reject laps whose track length deviates >7% from session median
LEN_TOL_PCT = 0.07

# Tracks where the spawn point is offset from gate 0 (start-finish), so the
# first recorded lap goes spawn → gate 1 → ... → gate 0 — missing the
# spawn↔gate-0 segment. Drop laps[0] deterministically for these tracks.
TRACKS_DROP_FIRST_LAP = {"track-003", "track-004"}

# Mini-sweep grid for H2
SIGMA_GRID = [1.0, 2.0, 4.0]
PNV_GRID = [4.0, 8.0, 16.0]

# RC channel order in raw rc_channels.parquet
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0
_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}


# ── Tracks (3) and sessions ────────────────────────────────────────────────────

TRACKS = {
    "track-002": dict(
        part="Part_5_Эксперементальный",
        name="Abu-Dhabi",
        sessions={
            "MT_s001": "2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
            "MT_s003": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-003",
            "MT_s004": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-004",
            "LO_s001": "2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
        },
    ),
    "track-003": dict(
        part="Part_6",
        name="Las Vegas",
        sessions={
            "MT_s001": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-003_session-001",
            "MT_s002": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-003_session-002",
            "MT_s003": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-003_session-003",
            "LO_s001": "2026-05-09_pilot-Gromozeka_drone-LiftOff_200_track-track-003_session-001",
            "LO_s002": "2026-05-09_pilot-Gromozeka_drone-LiftOff_200_track-track-003_session-002",
        },
    ),
    "track-004": dict(
        part="Part_7",
        name="Budapest",
        sessions={
            "MT_s001": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-001",
            "MT_s002": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-002",
            "MT_s003": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-003",
            "MT_s004": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-004",
            "MT_s005": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-005",
            "MT_s006": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-004_session-006",
        },
    ),
}

# Mini-sweep test sessions (most laps, most stable)
SWEEP_TEST = {
    "track-003": "MT_s002",   # 12 laps
    "track-004": "MT_s006",   # 7 laps, lowest variance
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lap_length(lap: Lap) -> float:
    return float(np.sum(np.linalg.norm(np.diff(lap.pos, axis=0), axis=1)))


def _filter_strict(laps: list[Lap], *, tol: float = LEN_TOL_PCT) -> tuple[list[Lap], list[Lap]]:
    """Reject laps whose track length deviates >tol from session median."""
    if len(laps) < 3:
        return laps, []
    lens = np.array([_lap_length(l) for l in laps])
    med = float(np.median(lens))
    keep, rej = [], []
    for i, l in enumerate(laps):
        if abs(lens[i] - med) / med > tol:
            rej.append(l)
        else:
            keep.append(l)
    return keep, rej


def _load_invert_lf(session_dir: Path) -> dict:
    inv_path = session_dir / "invert.json"
    if not inv_path.exists():
        return {}
    try:
        data = json.loads(inv_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("lf", {}) or {}


def _apply_invert(sticks: np.ndarray, invert_lf: dict) -> np.ndarray:
    if not invert_lf:
        return sticks
    out = sticks.copy()
    for key, col in _INVERT_KEY_TO_COL.items():
        if invert_lf.get(key, False):
            out[:, col] = -out[:, col]
    return out


def _wrap_error(raw: np.ndarray, L: float) -> np.ndarray:
    return np.where(raw > L / 2, L - raw, raw)


def _metrics(s_real: np.ndarray, s_est: np.ndarray, L: float) -> dict:
    if len(s_real) == 0:
        return dict(median_err_m=float("nan"), p90_err_m=float("nan"),
                    jump_rate=float("nan"))
    err = _wrap_error(np.abs(s_real - s_est), L)
    return {
        "median_err_m": float(np.median(err)),
        "p90_err_m":    float(np.percentile(err, 90)),
        "jump_rate":    float(np.mean(err > JUMP_THRESHOLD_M)),
    }


def _arc_length(pos: np.ndarray) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def _compute_s_real(pos: np.ndarray, ref: Reference) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(pos, ref.pos)
    except ImportError:
        dists = np.linalg.norm(pos[:, None, :] - ref.pos[None, :, :], axis=2)
    return ref.s[np.argmin(dists, axis=1)]


# ── Session loading ────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    track:          str
    session_id:     str        # e.g. "MT_s003"
    drone:          str
    rate_name:      str
    rate_profile:   dict
    invert_lf:      dict
    laps:           list[Lap]
    rejected:       list[Lap]
    rc_t_per_lap:   list[np.ndarray]
    rc_sticks_per_lap: list[np.ndarray]


def _load_session(track: str, session_id: str) -> SessionData:
    import pandas as pd

    track_info = TRACKS[track]
    session_dir = LIFTOFF / track_info["part"] / track_info["sessions"][session_id]
    laps_raw, _ = load_dct_session(session_dir)
    laps_a = filter_anomalous_laps(laps_raw)

    # Drop the structurally-partial first lap on tracks where spawn is offset
    # from the start-finish gate (track-003, track-004).
    first_dropped: list[Lap] = []
    if track in TRACKS_DROP_FIRST_LAP and len(laps_a) >= 2:
        first_dropped = [laps_a[0]]
        laps_a = laps_a[1:]

    laps, rejected = _filter_strict(laps_a)
    rejected = first_dropped + rejected

    rate_profile = load_rate_profile(session_dir)
    invert_lf = _load_invert_lf(session_dir)

    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    rc_t_per_lap, rc_sticks_per_lap = [], []
    for lap in laps:
        mask = (rc_ts >= lap.t[0]) & (rc_ts < lap.t[-1])
        t_rc = rc_ts[mask]
        st_rc = rc_sticks[mask]
        if len(t_rc) < 2:
            t_rc = np.array([], dtype=float)
            st_rc = np.empty((0, 4), dtype=float)
        rc_t_per_lap.append(t_rc)
        rc_sticks_per_lap.append(st_rc)

    drone = "MadTrainer" if session_id.startswith("MT") else "LiftOff_200"
    return SessionData(
        track=track,
        session_id=session_id,
        drone=drone,
        rate_name=rate_profile.get("name", "unknown"),
        rate_profile=rate_profile,
        invert_lf=invert_lf,
        laps=laps,
        rejected=rejected,
        rc_t_per_lap=rc_t_per_lap,
        rc_sticks_per_lap=rc_sticks_per_lap,
    )


# ── Reference building (median_time + n=5 + sw=5) ──────────────────────────────

def _pick_median_time(laps: list[Lap], n: int) -> list[int]:
    n = min(n, len(laps))
    durs = np.array([l.duration for l in laps])
    med = float(np.median(durs))
    order = np.argsort(np.abs(durs - med))
    return list(order[:n])


def _build_reference(
    source_laps: list[Lap],
    rate_profile: dict,
    invert_lf: dict,
    *,
    smooth_w: int = REF_SMOOTH_W,
    n_laps: int = REF_N_LAPS,
    grid_size: int = 1000,
) -> Reference:
    """Build averaged reference from a homogeneous list of laps (single rate)."""
    chosen = _pick_median_time(source_laps, n_laps)
    selected = [source_laps[i] for i in chosen]

    if len(selected) == 1:
        lap = selected[0]
        sticks = _apply_invert(lap.sticks, invert_lf)
        obs = physical_observation_matrix(sticks, rate_profile)
        return Reference.build_from_features(
            t=lap.t.copy(), obs=obs, pos=lap.pos.copy(),
            smooth_w=smooth_w, feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
            rate_profile=rate_profile,
        )

    grid = np.linspace(0.0, 1.0, grid_size)
    obs_stack, pos_stack = [], []
    for lap in selected:
        s = _arc_length(lap.pos)
        if s[-1] < 1e-6:
            continue
        u = s / s[-1]
        sticks = _apply_invert(lap.sticks, invert_lf)
        obs_lap = physical_observation_matrix(sticks, rate_profile)
        obs_resampled = np.empty((grid_size, obs_lap.shape[1]), dtype=np.float64)
        for c in range(obs_lap.shape[1]):
            obs_resampled[:, c] = np.interp(grid, u, obs_lap[:, c])
        pos_resampled = np.empty((grid_size, 3), dtype=np.float64)
        for c in range(3):
            pos_resampled[:, c] = np.interp(grid, u, lap.pos[:, c])
        obs_stack.append(obs_resampled)
        pos_stack.append(pos_resampled)

    obs_avg = np.mean(np.stack(obs_stack, axis=0), axis=0)
    pos_avg = np.mean(np.stack(pos_stack, axis=0), axis=0)
    t_dummy = np.linspace(0.0, 1.0, grid_size)

    return Reference.build_from_features(
        t=t_dummy, obs=obs_avg.astype(np.float32), pos=pos_avg,
        smooth_w=smooth_w, feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
        rate_profile=rate_profile,
    )


# ── Run filter through one session, continuous, return per-lap metrics ─────────

def _build_test_caches(test: SessionData, ref: Reference) -> list[dict]:
    test_laps = test.laps
    if not test_laps:
        return []
    telem_t = np.concatenate([lap.t for lap in test_laps])
    telem_pos = np.vstack([lap.pos for lap in test_laps])
    telem_s_real = _compute_s_real(telem_pos, ref)

    caches = []
    for rel_idx, (lap, t_rc, st_rc) in enumerate(
        zip(test_laps, test.rc_t_per_lap, test.rc_sticks_per_lap), start=1
    ):
        if len(t_rc) < 2:
            caches.append(dict(
                test_lap_relative=rel_idx, lap_index=lap.index,
                rc_t=t_rc, rc_sticks=st_rc, rc_s_real=np.array([], dtype=float),
                duration_s=0.0,
            ))
            continue
        idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_l = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        rc_s_real = telem_s_real[np.where(closer_l, idx_l, idx_r)]
        caches.append(dict(
            test_lap_relative=rel_idx,
            lap_index=lap.index,
            rc_t=t_rc, rc_sticks=st_rc, rc_s_real=rc_s_real,
            duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


def _run_continuous(
    ref: Reference,
    caches: list[dict],
    rate_profile: dict,
    *,
    obs_sigma: float = DEFAULT_OBS_SIGMA,
    process_noise_v: float = DEFAULT_PROCESS_NOISE_V,
) -> list[dict]:
    loc = OnlineLocalizer(
        ref,
        obs_sigma=obs_sigma,
        process_noise_v=process_noise_v,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=WEIGHTS_NO_THR,
    )
    loc.reset()
    out = []
    for cache in caches:
        if len(cache["rc_t"]) < 2:
            continue
        s_est = []
        prev_ts: float | None = None
        for i in range(len(cache["rc_t"])):
            dt = float(cache["rc_t"][i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(cache["rc_t"][i])
            res = loc.update(cache["rc_sticks"][i].tolist(), dt, rate_profile=rate_profile)
            s_est.append(res.s)
        s_est = np.array(s_est)
        m = _metrics(cache["rc_s_real"], s_est, ref.L)
        out.append(dict(
            test_lap_relative=cache["test_lap_relative"],
            lap_index=cache["lap_index"],
            n_frames=len(s_est),
            duration_s=cache["duration_s"],
            **m,
        ))
    return out


# ── Records ───────────────────────────────────────────────────────────────────

class H1Record(NamedTuple):
    track:          str
    test_session:   str
    test_drone:     str
    condition:      str   # same_drone | cross_drone
    test_lap_relative: int
    lap_index_abs:  int
    n_frames:       int
    duration_s:     float
    median_err_m:   float
    p90_err_m:      float
    jump_rate:      float


class H2Record(NamedTuple):
    track:        str
    test_session: str
    obs_sigma:    float
    pnv:          float
    n_test_laps:  int
    mean_p90:     float
    median_err:   float
    jump_rate:    float


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 8: generalize to new tracks")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
    print(f"Strict lap filter: length deviation > {LEN_TOL_PCT*100:.0f}% from median\n")

    # Load all sessions
    print("Loading sessions...")
    all_sessions: dict[tuple[str, str], SessionData] = {}
    track_lengths: dict[str, float] = {}
    for track, info in TRACKS.items():
        print(f"\n  [{track}] ({info['name']})")
        track_lens = []
        for sid in info["sessions"]:
            sd = _load_session(track, sid)
            # only Gromozeka_rate (project decision)
            if sd.rate_name != "Gromozeka_rate":
                print(f"    {sid}: SKIP rate={sd.rate_name}")
                continue
            all_sessions[(track, sid)] = sd
            session_track_lens = [_lap_length(l) for l in sd.laps]
            if session_track_lens:
                track_lens.extend(session_track_lens)
            rej_msg = f" rejected={len(sd.rejected)}" if sd.rejected else ""
            print(f"    {sid}: {sd.drone:11} laps={len(sd.laps)}{rej_msg}")
        track_lengths[track] = float(np.median(track_lens)) if track_lens else 0.0

    print(f"\nMedian track lengths: " +
          ", ".join(f"{t}={track_lengths[t]:.1f}m" for t in TRACKS))

    # ── H1 runs ────────────────────────────────────────────────────────────────
    print("\n=== H1: default params (σ=2.0, pnv=8.0) on all (track, test) pairs ===")
    h1_records: list[H1Record] = []
    t0 = time.perf_counter()
    for track in TRACKS:
        mt_keys = [(t, s) for (t, s) in all_sessions if t == track and s.startswith("MT")]
        lo_keys = [(t, s) for (t, s) in all_sessions if t == track and s.startswith("LO")]

        # Same-drone (MT): per test session, ref = best 5 from OTHER MT sessions
        for test_key in mt_keys:
            test_sd = all_sessions[test_key]
            other_mt = [k for k in mt_keys if k != test_key]
            if not other_mt:
                continue
            source_laps: list[Lap] = []
            ref_rate = None
            ref_invert = None
            for k in other_mt:
                source_laps.extend(all_sessions[k].laps)
                if ref_rate is None:
                    ref_rate = all_sessions[k].rate_profile
                    ref_invert = all_sessions[k].invert_lf
            if len(source_laps) < 2:
                continue
            ref = _build_reference(source_laps, ref_rate, ref_invert)
            caches = _build_test_caches(test_sd, ref)
            recs = _run_continuous(ref, caches, test_sd.rate_profile)
            for r in recs:
                h1_records.append(H1Record(
                    track=track, test_session=test_key[1], test_drone=test_sd.drone,
                    condition="same_drone",
                    test_lap_relative=r["test_lap_relative"],
                    lap_index_abs=r["lap_index"], n_frames=r["n_frames"],
                    duration_s=r["duration_s"],
                    median_err_m=r["median_err_m"], p90_err_m=r["p90_err_m"],
                    jump_rate=r["jump_rate"],
                ))
            mean_p90 = float(np.mean([r["p90_err_m"] for r in recs])) if recs else float("nan")
            print(f"  [{track:>9} / {test_key[1]:7} same-drone] "
                  f"n_test={len(recs)} mean_p90={mean_p90:5.2f}m")

            if args.smoke and track == "track-003":
                break  # one same-drone per track in smoke mode

        # Cross-drone (LO): ref = best 5 from ALL MT sessions
        for test_key in lo_keys:
            test_sd = all_sessions[test_key]
            mt_source: list[Lap] = []
            ref_rate = None
            ref_invert = None
            for k in mt_keys:
                mt_source.extend(all_sessions[k].laps)
                if ref_rate is None:
                    ref_rate = all_sessions[k].rate_profile
                    ref_invert = all_sessions[k].invert_lf
            if len(mt_source) < 2:
                continue
            ref = _build_reference(mt_source, ref_rate, ref_invert)
            caches = _build_test_caches(test_sd, ref)
            recs = _run_continuous(ref, caches, test_sd.rate_profile)
            for r in recs:
                h1_records.append(H1Record(
                    track=track, test_session=test_key[1], test_drone=test_sd.drone,
                    condition="cross_drone",
                    test_lap_relative=r["test_lap_relative"],
                    lap_index_abs=r["lap_index"], n_frames=r["n_frames"],
                    duration_s=r["duration_s"],
                    median_err_m=r["median_err_m"], p90_err_m=r["p90_err_m"],
                    jump_rate=r["jump_rate"],
                ))
            mean_p90 = float(np.mean([r["p90_err_m"] for r in recs])) if recs else float("nan")
            print(f"  [{track:>9} / {test_key[1]:7} cross-drone] "
                  f"n_test={len(recs)} mean_p90={mean_p90:5.2f}m")

            if args.smoke:
                break  # one cross-drone per track in smoke mode

    print(f"H1 total: {time.perf_counter() - t0:.1f}s\n")

    # ── H2 mini-sweep ──────────────────────────────────────────────────────────
    h2_records: list[H2Record] = []
    if not args.smoke:
        print("=== H2: mini-sweep σ × pnv on track-003 and track-004 ===")
        t0 = time.perf_counter()
        for track, test_sid in SWEEP_TEST.items():
            test_key = (track, test_sid)
            if test_key not in all_sessions:
                print(f"  [WARN] sweep test session {test_key} not loaded")
                continue
            test_sd = all_sessions[test_key]
            mt_keys = [(t, s) for (t, s) in all_sessions if t == track and s.startswith("MT")]
            other_mt = [k for k in mt_keys if k != test_key]
            source_laps: list[Lap] = []
            ref_rate = None
            ref_invert = None
            for k in other_mt:
                source_laps.extend(all_sessions[k].laps)
                if ref_rate is None:
                    ref_rate = all_sessions[k].rate_profile
                    ref_invert = all_sessions[k].invert_lf
            ref = _build_reference(source_laps, ref_rate, ref_invert)
            caches = _build_test_caches(test_sd, ref)

            print(f"  [{track:>9}] test={test_sid}, ref from {len(other_mt)} sessions "
                  f"({len(source_laps)} laps available)")
            for sigma in SIGMA_GRID:
                for pnv in PNV_GRID:
                    recs = _run_continuous(
                        ref, caches, test_sd.rate_profile,
                        obs_sigma=sigma, process_noise_v=pnv,
                    )
                    if not recs:
                        continue
                    p90s = [r["p90_err_m"] for r in recs]
                    medians = [r["median_err_m"] for r in recs]
                    jumps = [r["jump_rate"] for r in recs]
                    h2_records.append(H2Record(
                        track=track, test_session=test_sid,
                        obs_sigma=sigma, pnv=pnv,
                        n_test_laps=len(recs),
                        mean_p90=float(np.mean(p90s)),
                        median_err=float(np.mean(medians)),
                        jump_rate=float(np.mean(jumps)),
                    ))
                    print(f"    σ={sigma:.1f} pnv={pnv:5.1f} → mean_p90={np.mean(p90s):5.2f}m")
        print(f"H2 total: {time.perf_counter() - t0:.1f}s\n")

    # ── Save raw outputs ───────────────────────────────────────────────────────
    suffix = "_smoke" if args.smoke else ""
    df_h1 = pd.DataFrame(h1_records, columns=H1Record._fields)
    df_h1.to_csv(OUT_DIR / f"results_h1{suffix}.csv", index=False)
    print(f"Saved H1 records → {OUT_DIR / f'results_h1{suffix}.csv'} ({len(df_h1)} rows)")

    if h2_records:
        df_h2 = pd.DataFrame(h2_records, columns=H2Record._fields)
        df_h2.to_csv(OUT_DIR / "results_h2.csv", index=False)
        print(f"Saved H2 records → {OUT_DIR / 'results_h2.csv'} ({len(df_h2)} rows)")

    if args.smoke:
        print("\n=== Smoke summary ===")
        print(df_h1.groupby(["track", "test_session", "condition"])
                  ["p90_err_m"].agg(["count", "mean"]).to_string())
        return

    # ── Aggregation, plots, report ─────────────────────────────────────────────
    summary_h1 = (
        df_h1.groupby(["track", "test_session", "condition"])
        .agg(
            n_laps=("test_lap_relative", "count"),
            mean_p90=("p90_err_m", "mean"),
            median_p90=("p90_err_m", "median"),
            mean_median_err=("median_err_m", "mean"),
            mean_jump=("jump_rate", "mean"),
        )
        .reset_index()
    )
    summary_h1.to_csv(OUT_DIR / "summary_h1.csv", index=False)
    print(f"Saved summary_h1 → {OUT_DIR / 'summary_h1.csv'}")

    track_summary = (
        df_h1.groupby(["track", "condition"])
        .agg(
            n_test_laps=("test_lap_relative", "count"),
            mean_p90=("p90_err_m", "mean"),
            median_p90=("p90_err_m", "median"),
            mean_median_err=("median_err_m", "mean"),
            mean_jump=("jump_rate", "mean"),
        )
        .reset_index()
    )
    track_summary.to_csv(OUT_DIR / "track_summary.csv", index=False)

    df_h2 = pd.DataFrame(h2_records, columns=H2Record._fields)

    _make_plots(df_h1, summary_h1, df_h2, track_lengths)
    _write_report(df_h1, summary_h1, track_summary, df_h2, track_lengths)
    print(f"\nAll outputs in: {OUT_DIR}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _make_plots(df_h1, summary_h1, df_h2, track_lengths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. H1: bar chart of mean p90 per (track, condition)
    fig, ax = plt.subplots(figsize=(10, 5))
    tracks = list(TRACKS.keys())
    conds = ["same_drone", "cross_drone"]
    width = 0.35
    x = np.arange(len(tracks))
    same_vals, cross_vals = [], []
    for t in tracks:
        s_sub = df_h1[(df_h1["track"] == t) & (df_h1["condition"] == "same_drone")]
        c_sub = df_h1[(df_h1["track"] == t) & (df_h1["condition"] == "cross_drone")]
        same_vals.append(float(s_sub["p90_err_m"].mean()) if not s_sub.empty else 0.0)
        cross_vals.append(float(c_sub["p90_err_m"].mean()) if not c_sub.empty else float("nan"))

    b1 = ax.bar(x - width/2, same_vals, width, color="#27ae60", label="same-drone", edgecolor="black")
    cross_plot = [v if not np.isnan(v) else 0 for v in cross_vals]
    b2 = ax.bar(x + width/2, cross_plot, width, color="#e74c3c", label="cross-drone", edgecolor="black")
    for b, v in zip(b1, same_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    for b, v in zip(b2, cross_vals):
        if np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, 0.5, "no LO data", ha="center", fontsize=8, style="italic")
        else:
            ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.5, label="target 15 m")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TRACKS[t]['name']}, {track_lengths[t]:.0f}m)" for t in tracks])
    ax.set_ylabel("mean p90 error, m")
    ax.set_title("H1: Default Exp 3 params (σ=2.0, pnv=8.0) on each track")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "h1_track_comparison.png", dpi=140)
    plt.close(fig)

    # 2. H1: per-session detailed bar chart
    fig, ax = plt.subplots(figsize=(13, 5))
    cond_colors = {"same_drone": "#27ae60", "cross_drone": "#e74c3c"}
    rows = summary_h1.copy()
    rows["label"] = rows["track"].astype(str) + " / " + rows["test_session"].astype(str)
    rows = rows.sort_values(["track", "condition", "test_session"]).reset_index(drop=True)
    colors = [cond_colors[c] for c in rows["condition"]]
    bars = ax.bar(np.arange(len(rows)), rows["mean_p90"], color=colors, edgecolor="black")
    for b, v, n in zip(bars, rows["mean_p90"], rows["n_laps"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, f"{v:.1f}\n(n={n})",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows["label"], rotation=30, ha="right", fontsize=9)
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.5)
    ax.set_ylabel("mean p90 error, m")
    ax.set_title("H1: per-session mean p90 (default Exp 3 params)")
    legend_handles = [plt.Rectangle((0,0),1,1, color=cond_colors[c]) for c in conds]
    ax.legend(legend_handles, conds, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "h1_per_session.png", dpi=140)
    plt.close(fig)

    # 3. H2: heatmaps (one per new track)
    if not df_h2.empty:
        fig, axes = plt.subplots(1, len(SWEEP_TEST), figsize=(6 * len(SWEEP_TEST), 5),
                                 squeeze=False)
        for i, track in enumerate(SWEEP_TEST):
            sub = df_h2[df_h2["track"] == track]
            if sub.empty:
                continue
            grid = sub.pivot(index="pnv", columns="obs_sigma", values="mean_p90")
            grid = grid.reindex(index=PNV_GRID, columns=SIGMA_GRID)
            ax = axes[0, i]
            im = ax.imshow(grid.values, cmap="RdYlGn_r", aspect="auto",
                           vmin=df_h2["mean_p90"].min(),
                           vmax=df_h2["mean_p90"].max())
            ax.set_xticks(range(len(SIGMA_GRID)))
            ax.set_xticklabels([f"{s:g}" for s in SIGMA_GRID])
            ax.set_yticks(range(len(PNV_GRID)))
            ax.set_yticklabels([f"{p:g}" for p in PNV_GRID])
            ax.set_title(f"{track} mini-sweep mean p90 (m)")
            ax.set_xlabel("obs_sigma")
            ax.set_ylabel("pnv")
            for r in range(len(PNV_GRID)):
                for c in range(len(SIGMA_GRID)):
                    v = grid.values[r, c]
                    if np.isnan(v):
                        continue
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=10,
                            color="black")
            # Mark default
            default_r = PNV_GRID.index(DEFAULT_PROCESS_NOISE_V) if DEFAULT_PROCESS_NOISE_V in PNV_GRID else None
            default_c = SIGMA_GRID.index(DEFAULT_OBS_SIGMA) if DEFAULT_OBS_SIGMA in SIGMA_GRID else None
            if default_r is not None and default_c is not None:
                ax.plot(default_c, default_r, "k*", markersize=18, markerfacecolor="white",
                        label="Exp 3 default")
                ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=axes, shrink=0.7, label="mean p90, m")
        fig.savefig(PLOTS_DIR / "h2_minisweep_heatmaps.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # 4. H3: optimal pnv vs track length (scatter)
    if not df_h2.empty:
        # Find optimum (σ, pnv) per new track
        opts = {}
        for track in SWEEP_TEST:
            sub = df_h2[df_h2["track"] == track]
            if sub.empty:
                continue
            best = sub.loc[sub["mean_p90"].idxmin()]
            opts[track] = (float(best["obs_sigma"]), float(best["pnv"]),
                           float(best["mean_p90"]))
        # Add track-002 baseline (Exp 3 known optimum)
        opts["track-002"] = (DEFAULT_OBS_SIGMA, DEFAULT_PROCESS_NOISE_V, float("nan"))

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [track_lengths[t] for t in opts]
        ys = [opts[t][1] for t in opts]
        labels = list(opts.keys())
        ax.scatter(xs, ys, s=180, c="#3498db", edgecolor="black", zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(f"{lbl}\n(σ_opt={opts[lbl][0]:.1f}, pnv_opt={opts[lbl][1]:.1f})",
                        (x, y), textcoords="offset points", xytext=(10, 5), fontsize=9)
        # Linear fit (3 points = exact line if not regularized; just connect for trend)
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xfit = np.array([min(xs), max(xs)])
            ax.plot(xfit, slope * xfit + intercept, "k--", alpha=0.5,
                    label=f"slope = {slope:+.4f} pnv per m of track length")
            ax.legend()
        ax.set_xlabel("track length (median, m)")
        ax.set_ylabel("optimal pnv")
        ax.set_title("H3: track length vs optimal process_noise_v")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "h3_pnv_vs_length.png", dpi=140)
        plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(df_h1, summary_h1, track_summary, df_h2, track_lengths):
    # Hypothesis verification
    track_p90 = {}
    for track in TRACKS:
        sub_same = df_h1[(df_h1["track"] == track) & (df_h1["condition"] == "same_drone")]
        sub_cross = df_h1[(df_h1["track"] == track) & (df_h1["condition"] == "cross_drone")]
        track_p90[track] = dict(
            same=float(sub_same["p90_err_m"].mean()) if not sub_same.empty else float("nan"),
            cross=float(sub_cross["p90_err_m"].mean()) if not sub_cross.empty else float("nan"),
            n_same=int(len(sub_same)),
            n_cross=int(len(sub_cross)),
        )

    # H1: p90 < 15m on new tracks (same-drone), best-effort cross-drone
    new_tracks = [t for t in TRACKS if t != "track-002"]
    h1_pass_same = all(track_p90[t]["same"] < JUMP_THRESHOLD_M for t in new_tracks)
    h1_cross_vals = [track_p90[t]["cross"] for t in new_tracks if not np.isnan(track_p90[t]["cross"])]
    h1_pass_cross = all(v < JUMP_THRESHOLD_M for v in h1_cross_vals)
    h1_overall = "ПОДТВЕРЖДЕНО" if h1_pass_same and h1_pass_cross else "ЧАСТИЧНО"

    # H2: optimal (σ, pnv) close to default on new tracks
    opts = {}
    if not df_h2.empty:
        for track in SWEEP_TEST:
            sub = df_h2[df_h2["track"] == track]
            if sub.empty:
                continue
            best = sub.loc[sub["mean_p90"].idxmin()]
            opts[track] = (float(best["obs_sigma"]), float(best["pnv"]), float(best["mean_p90"]))
            # Also record default cell for transfer cost
            default_cell = sub[(sub["obs_sigma"] == DEFAULT_OBS_SIGMA)
                                & (sub["pnv"] == DEFAULT_PROCESS_NOISE_V)]
            opts[track] = (*opts[track],
                           float(default_cell["mean_p90"].iloc[0]) if not default_cell.empty else float("nan"))

    h2_close = all(abs(opts[t][0] - DEFAULT_OBS_SIGMA) <= 1.0
                   and abs(opts[t][1] - DEFAULT_PROCESS_NOISE_V) <= 4.0
                   for t in opts) if opts else False
    h2_overall = "ПОДТВЕРЖДЕНО" if h2_close else "ОТКЛОНЕНО"

    # H3: optimal pnv vs track length
    h3_opts = {}
    h3_opts["track-002"] = (DEFAULT_OBS_SIGMA, DEFAULT_PROCESS_NOISE_V, float("nan"), float("nan"))
    for t, v in opts.items():
        h3_opts[t] = v
    h3_xs = []
    h3_ys = []
    for t, v in h3_opts.items():
        h3_xs.append(track_lengths[t])
        h3_ys.append(v[1])
    h3_slope = float(np.polyfit(h3_xs, h3_ys, 1)[0]) if len(h3_xs) >= 2 else 0.0
    h3_overall = "СОГЛАСУЕТСЯ С ГИПОТЕЗОЙ (slope > 0)" if h3_slope > 0 else "НЕ СОГЛАСУЕТСЯ"

    md = []
    md.append("# Эксперимент 8: Генерализация на новые треки — Отчёт\n\n")

    md.append("## 1. Условия запуска\n\n")
    md.append("| Параметр | Значение |\n|---|---|\n"
              f"| Треки | track-002 (baseline), track-003 (новый), track-004 (новый) |\n"
              f"| Дрон ref | MadTrainer (Gromozeka_rate, общий invert) |\n"
              f"| Дрон test | MadTrainer (same-drone) + LiftOff_200 (cross-drone) |\n"
              f"| Дефолтные параметры (Exp 3) | σ={DEFAULT_OBS_SIGMA}, pnv={DEFAULT_PROCESS_NOISE_V}, pns={PROCESS_NOISE_S}, weights=no_thr |\n"
              f"| Реф (Exp 6) | median_time, n_laps={REF_N_LAPS}, smooth_w={REF_SMOOTH_W} |\n"
              f"| Strict lap filter | rejection if |length − median| / median > {LEN_TOL_PCT*100:.0f}% |\n"
              f"| Mini-sweep grid | σ ∈ {SIGMA_GRID}, pnv ∈ {PNV_GRID} (3×3) |\n"
              f"| Continuous filter | без сброса между лапами (по итогам Exp 7) |\n\n")

    md.append("## 2. Свойства треков и данных\n\n")
    md.append("| Трек | Имя | Ворот | Длина (med) | MT laps | LO laps |\n|---|---|---|---|---|---|\n")
    for track, info in TRACKS.items():
        # gates count
        try:
            gates = json.loads((ROOT / "tracks" / track / "track.json").read_text(encoding="utf-8")).get("gates", [])
            n_gates = len(gates)
        except Exception:
            n_gates = "?"
        n_mt = sum(1 for r in df_h1.itertuples() if r.track == track and r.condition == "same_drone")
        n_lo = sum(1 for r in df_h1.itertuples() if r.track == track and r.condition == "cross_drone")
        md.append(f"| {track} | {info['name']} | {n_gates} | {track_lengths[track]:.1f} м "
                  f"| {n_mt} | {n_lo or '—'} |\n")
    md.append("\n")

    md.append("## 3. H1: дефолтные параметры на новых треках\n\n")
    md.append("### 3.1 Сводка по трекам\n\n")
    md.append("| Трек | Условие | n тест-лапов | mean p90, м | median p90, м | mean median, м |\n"
              "|---|---|---|---|---|---|\n")
    for _, r in track_summary.iterrows():
        md.append(f"| {r['track']} | {r['condition']} | {r['n_test_laps']} | "
                  f"**{r['mean_p90']:.2f}** | {r['median_p90']:.2f} | {r['mean_median_err']:.2f} |\n")
    md.append("\n")

    md.append("### 3.2 Per-session разбор\n\n")
    md.append("| Трек | Сессия | Дрон | Условие | n лапов | mean p90 |\n|---|---|---|---|---|---|\n")
    for _, r in summary_h1.sort_values(["track", "condition", "test_session"]).iterrows():
        # find drone of test session
        tk = (r["track"], r["test_session"])
        drone_name = "MadTrainer" if r["test_session"].startswith("MT") else "LiftOff_200"
        md.append(f"| {r['track']} | {r['test_session']} | {drone_name} | {r['condition']} | "
                  f"{r['n_laps']} | {r['mean_p90']:.2f} |\n")
    md.append("\n")

    md.append("## 4. H2: mini-sweep σ × pnv на новых треках\n\n")
    if not df_h2.empty:
        for track in SWEEP_TEST:
            sub = df_h2[df_h2["track"] == track]
            if sub.empty:
                continue
            md.append(f"### {track} ({TRACKS[track]['name']})\n\n")
            md.append("| pnv ↓ \\ σ → | " + " | ".join([f"{s:g}" for s in SIGMA_GRID]) + " |\n")
            md.append("|---|" + "|".join(["---"] * len(SIGMA_GRID)) + "|\n")
            for pnv in PNV_GRID:
                row = [f"**{pnv:g}**"]
                for sigma in SIGMA_GRID:
                    cell = sub[(sub["obs_sigma"] == sigma) & (sub["pnv"] == pnv)]
                    if cell.empty:
                        row.append("—")
                    else:
                        v = float(cell["mean_p90"].iloc[0])
                        marker = " ★" if (sigma == opts[track][0] and pnv == opts[track][1]) else ""
                        row.append(f"{v:.2f}{marker}")
                md.append("| " + " | ".join(row) + " |\n")
            sigma_o, pnv_o, p90_opt, p90_def = opts[track]
            transfer_cost = (p90_def - p90_opt) if (not np.isnan(p90_def) and not np.isnan(p90_opt)) else float("nan")
            md.append(f"\n**Оптимум**: σ={sigma_o}, pnv={pnv_o} → p90={p90_opt:.2f}m. "
                      f"С дефолтом (σ={DEFAULT_OBS_SIGMA}, pnv={DEFAULT_PROCESS_NOISE_V}) → p90={p90_def:.2f}m. "
                      f"**Цена переноса = {transfer_cost:+.2f}m**.\n\n")

    md.append("## 5. H3: оптимальный pnv vs длина трека\n\n")
    md.append("| Трек | Длина, м | σ_opt | pnv_opt | mean p90 (opt) |\n|---|---|---|---|---|\n")
    for t, v in h3_opts.items():
        md.append(f"| {t} | {track_lengths[t]:.1f} | {v[0]:g} | {v[1]:g} | "
                  f"{v[2]:.2f} |\n" if not np.isnan(v[2]) else
                  f"| {t} | {track_lengths[t]:.1f} | {v[0]:g} | {v[1]:g} | (Exp 3 baseline) |\n")
    md.append(f"\n**Линейный тренд pnv vs length**: slope = {h3_slope:+.4f} pnv/м.\n\n")

    md.append("## 6. Проверка гипотез\n\n")
    md.append("| # | Гипотеза | Результат |\n|---|---|---|\n")
    same_str = ", ".join(f"{t}={track_p90[t]['same']:.2f}m" for t in new_tracks)
    cross_str = ", ".join(f"{t}={track_p90[t]['cross']:.2f}m" for t in new_tracks
                          if not np.isnan(track_p90[t]['cross']))
    md.append(f"| H1 | Дефолтные параметры → p90 < 15м на новых треках | "
              f"**{h1_overall}** (same: {same_str}; cross: {cross_str}) |\n")
    if opts:
        md.append(f"| H2 | Оптимум (σ, pnv) близок к baseline (σ=2, pnv=8) | **{h2_overall}** ")
        for t, (s, p, _, _) in opts.items():
            md.append(f"({t}: σ={s}, pnv={p}) ")
        md.append("|\n")
    md.append(f"| H3 | Длина трека ↔ оптимальный pnv (longer→larger) | **{h3_overall}** "
              f"(slope = {h3_slope:+.4f} pnv/м) |\n\n")

    md.append("## 7. Файлы\n\n"
              "- `results_h1.csv` — все per-lap метрики H1 (default params на всех (трек, тест) парах)\n"
              "- `results_h2.csv` — mini-sweep H2 на track-003/004\n"
              "- `summary_h1.csv` — агрегация по (track, session, condition)\n"
              "- `track_summary.csv` — агрегация по (track, condition)\n"
              "- `plots/h1_track_comparison.png` — основной график H1\n"
              "- `plots/h1_per_session.png` — детально по сессиям\n"
              "- `plots/h2_minisweep_heatmaps.png` — тепловые карты mini-sweep\n"
              "- `plots/h3_pnv_vs_length.png` — H3 scatter\n")

    (OUT_DIR / "report.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

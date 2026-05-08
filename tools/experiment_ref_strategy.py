"""Experiment 6: Reference Building Strategy.

Research question
-----------------
How do (lap_selection, n_laps, smooth_w) influence the localizer accuracy
when used to build the reference?

Design
------
- Track:                track-002 (Part_5, same 4 sessions as Exp 1–4).
- Reference sources:    MadTrainer session-001 (F1), LiftOff_200 session-001 (F3).
- Test flights:         the OTHER three flights for each ref (no leakage).
                        MT ref  → F2 (same_drone), F3 (cross), F4 (cross)
                        LO ref  → F1 (cross),     F2 (cross), F4 (same_drone)
- Factors:
    lap_selection ∈ {fastest, median_time, manual_best}
    n_laps        ∈ {1, 3, 5}        — averaged over arc-length parameter
    smooth_w      ∈ {1, 3, 5, 9, 15}
- Fixed (from Exp 3):  RC+Rate, obs_sigma=2.0, pnv=8.0, pns=1.5, preset=no_thr.

Total runs (full): 6 (ref,test) × 3 lap_sel × 3 n_laps × 5 smooth_w × 3 last_laps = 810 records.
Smoke run (--smoke): 2 (ref,test) × 1 lap_sel × 1 n_laps × 2 smooth_w × 3 last_laps ≈ 12 records.

Usage (from project root)
-------------------------
    python tools/experiment_ref_strategy.py --smoke    # smoke test
    python tools/experiment_ref_strategy.py            # full experiment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.lap_loader import Lap, filter_anomalous_laps, load_dct_session
from dct.localization.online_localizer import OnlineLocalizer, Reference
from dct.localization.reference_select import select_best_reference
from dct.rate_features import (
    FEATURE_BETAFLIGHT_CLASSIC_V1,
    load_rate_profile,
    physical_observation_matrix,
)


# ── Configuration ──────────────────────────────────────────────────────────────

def _find_part5() -> Path:
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")
    cands = sorted(d for d in liftoff.iterdir() if d.name.startswith("Part_5"))
    if not cands:
        raise FileNotFoundError(f"No Part_5* under {liftoff}")
    return cands[0]


PART_5 = _find_part5()
OUT_DIR = Path(__file__).parent / "exp6_ref_strategy"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0
N_LAST_LAPS = 3

# Optimal hyperparameters from Exp 3
OBS_SIGMA = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5
WEIGHTS_NO_THR = np.array([0.0, 1.0, 1.0, 1.0])  # preset "no_thr" — best from Exp 3

# Ref sources: one canonical per drone
REF_SOURCES = [
    dict(
        name="MT_ref",
        drone="MadTrainer",
        rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
    ),
    dict(
        name="LO_ref",
        drone="LiftOff_200",
        rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
    ),
]

FLIGHTS = [
    dict(flight_id=1, drone="MadTrainer",  rate="Gromozeka_rate",
         session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001"),
    dict(flight_id=2, drone="MadTrainer",  rate="RedSheep_rate",
         session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-002"),
    dict(flight_id=3, drone="LiftOff_200", rate="Gromozeka_rate",
         session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001"),
    dict(flight_id=4, drone="LiftOff_200", rate="RedSheep_rate",
         session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-002"),
]

LAP_SELECTIONS = ["fastest", "median_time", "manual_best"]
N_LAPS_LEVELS = [1, 3, 5]
SMOOTH_W_LEVELS = [1, 3, 5, 9, 15]

# RC channel order in raw rc_channels.parquet
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0

# Maps invert_lf keys → column index in [thr, yaw, pitch, roll]
_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}


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


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FlightRawData:
    flight: dict
    rate_profile: dict
    invert_lf: dict   # {"in_throttle": bool, "in_yaw": bool, "in_pitch": bool, "in_roll": bool}
    laps: list[Lap]
    rc_t_per_lap: list[np.ndarray]
    rc_sticks_per_lap: list[np.ndarray]
    last_lap_indices: list[int]
    telem_t: np.ndarray
    telem_pos: np.ndarray


class RunRecord(NamedTuple):
    ref_source:   str
    ref_drone:    str
    flight_id:    int
    test_drone:   str
    drone_cond:   str   # same_drone | cross_drone
    lap_selection: str
    n_laps:       int
    smooth_w:     int
    test_lap:     int
    n_frames:     int
    duration_s:   float
    median_err_m: float
    p90_err_m:    float
    jump_rate:    float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wrap_error(raw: np.ndarray, L: float) -> np.ndarray:
    return np.where(raw > L / 2, L - raw, raw)


def _metrics(s_real: np.ndarray, s_est: np.ndarray, L: float) -> dict:
    err = _wrap_error(np.abs(s_real - s_est), L)
    return {
        "median_err_m": float(np.median(err)),
        "p90_err_m":    float(np.percentile(err, 90)),
        "jump_rate":    float(np.mean(err > JUMP_THRESHOLD_M)),
    }


def _lap_arc_length(pos: np.ndarray) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def _compute_s_real(pos: np.ndarray, ref: Reference) -> np.ndarray:
    """Project a sequence of xyz onto the reference arc parameter via NN."""
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(pos, ref.pos)
    except ImportError:
        chunk = 500
        dists = np.empty((len(pos), len(ref.pos)), dtype=np.float32)
        for i in range(0, len(pos), chunk):
            diff = pos[i:i + chunk, np.newaxis, :] - ref.pos[np.newaxis, :, :]
            dists[i:i + chunk] = np.linalg.norm(diff, axis=2)
    return ref.s[np.argmin(dists, axis=1)]


def _condition(ref_drone: str, test_drone: str) -> str:
    return "same_drone" if ref_drone == test_drone else "cross_drone"


# ── Stage 1: load raw flight data ──────────────────────────────────────────────

def _load_flight(flight: dict) -> FlightRawData:
    import pandas as pd

    session_dir = PART_5 / flight["session"]
    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    rate_profile = load_rate_profile(session_dir)
    invert_lf = _load_invert_lf(session_dir)

    last = laps[-N_LAST_LAPS:]
    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    rc_t_per_lap, rc_sticks_per_lap, last_lap_indices = [], [], []
    for lap in last:
        mask = (rc_ts >= lap.t[0]) & (rc_ts < lap.t[-1])
        t_rc = rc_ts[mask]
        st_rc = rc_sticks[mask]
        if len(t_rc) < 2:
            t_rc = np.array([], dtype=float)
            st_rc = np.empty((0, 4), dtype=float)
        rc_t_per_lap.append(t_rc)
        rc_sticks_per_lap.append(st_rc)
        last_lap_indices.append(lap.index)

    telem_t = np.concatenate([lap.t for lap in last])
    telem_pos = np.vstack([lap.pos for lap in last])

    return FlightRawData(
        flight=flight,
        rate_profile=rate_profile,
        invert_lf=invert_lf,
        laps=laps,
        rc_t_per_lap=rc_t_per_lap,
        rc_sticks_per_lap=rc_sticks_per_lap,
        last_lap_indices=last_lap_indices,
        telem_t=telem_t,
        telem_pos=telem_pos,
    )


# ── Stage 2: lap selection & N-laps averaging ──────────────────────────────────

def _pick_lap_indices(
    laps: list[Lap],
    *,
    selection: str,
    n_laps: int,
    smooth_w_for_quality: int = 5,
) -> list[int]:
    """Return 0-based indices of n_laps best laps by the chosen strategy.

    Strategies
    ----------
    fastest      : the n_laps shortest-duration laps (most efficient flight)
    median_time  : laps closest to the median duration (typical, not extremes)
    manual_best  : top-n by select_best_reference (LOO NN-greedy quality)
    """
    n_laps = min(n_laps, len(laps))
    if selection == "fastest":
        order = np.argsort([lap.duration for lap in laps])
        return list(order[:n_laps])
    if selection == "median_time":
        durs = np.array([lap.duration for lap in laps])
        med = float(np.median(durs))
        order = np.argsort(np.abs(durs - med))
        return list(order[:n_laps])
    if selection == "manual_best":
        # full LOO ranking is expensive; cached per (session, smooth_w)
        best_idx, scores = select_best_reference(laps, smooth_w=smooth_w_for_quality)
        order = np.argsort(scores)
        return list(order[:n_laps])
    raise ValueError(f"Unknown lap selection: {selection}")


def _build_avg_reference(
    laps: list[Lap],
    indices: list[int],
    *,
    smooth_w: int,
    rate_profile: dict,
    invert_lf: dict,
    grid_size: int = 1000,
) -> Reference:
    """Build a reference by averaging N laps over a common arc-length grid.

    Each selected lap is resampled onto a unit [0..1] arc-length grid; physical
    observations (Betaflight features) and positions are averaged across laps.
    The final arc-length parameter is recomputed from the averaged positions
    (so units stay in metres and ``ref.L`` reflects the averaged track length).

    ``invert_lf`` — Liftoff stick-sign overrides from the session's
    ``invert.json``; channels marked True are sign-flipped before being mapped
    through the Betaflight curve, matching the GUI's reference build pipeline.
    """
    selected = [laps[i] for i in indices]
    if len(selected) == 1:
        lap = selected[0]
        sticks = _apply_invert(lap.sticks, invert_lf)
        obs = physical_observation_matrix(sticks, rate_profile)
        return Reference.build_from_features(
            t=lap.t.copy(),
            obs=obs,
            pos=lap.pos.copy(),
            smooth_w=smooth_w,
            feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
            rate_profile=rate_profile,
        )

    grid = np.linspace(0.0, 1.0, grid_size)
    obs_stack = []
    pos_stack = []
    for lap in selected:
        s_lap = _lap_arc_length(lap.pos)
        if s_lap[-1] < 1e-6:
            continue
        u_lap = s_lap / s_lap[-1]
        sticks = _apply_invert(lap.sticks, invert_lf)
        obs_lap = physical_observation_matrix(sticks, rate_profile)
        obs_resampled = np.empty((grid_size, obs_lap.shape[1]), dtype=np.float64)
        for c in range(obs_lap.shape[1]):
            obs_resampled[:, c] = np.interp(grid, u_lap, obs_lap[:, c])
        pos_resampled = np.empty((grid_size, 3), dtype=np.float64)
        for c in range(3):
            pos_resampled[:, c] = np.interp(grid, u_lap, lap.pos[:, c])
        obs_stack.append(obs_resampled)
        pos_stack.append(pos_resampled)

    obs_avg = np.mean(np.stack(obs_stack, axis=0), axis=0)
    pos_avg = np.mean(np.stack(pos_stack, axis=0), axis=0)
    t_dummy = np.linspace(0.0, 1.0, grid_size)

    return Reference.build_from_features(
        t=t_dummy,
        obs=obs_avg.astype(np.float32),
        pos=pos_avg,
        smooth_w=smooth_w,
        feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
        rate_profile=rate_profile,
    )


# ── Stage 3: project test flight onto reference ────────────────────────────────

def _compute_lap_caches(frd: FlightRawData, ref: Reference) -> list[dict]:
    telem_s_real = _compute_s_real(frd.telem_pos, ref)
    telem_t = frd.telem_t
    caches = []
    for lap_idx, t_rc, sticks_rc in zip(
        frd.last_lap_indices, frd.rc_t_per_lap, frd.rc_sticks_per_lap
    ):
        if len(t_rc) < 2:
            caches.append(dict(
                lap_index=lap_idx, rc_t=t_rc, rc_sticks=sticks_rc,
                rc_s_real=np.array([], dtype=float), duration_s=0.0,
            ))
            continue
        idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_l = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        rc_s_real = telem_s_real[np.where(closer_l, idx_l, idx_r)]
        caches.append(dict(
            lap_index=lap_idx, rc_t=t_rc, rc_sticks=sticks_rc,
            rc_s_real=rc_s_real, duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


# ── Stage 4: run localizer ─────────────────────────────────────────────────────

def _run_one(
    ref: Reference,
    caches: list[dict],
    rate_profile: dict,
) -> list[dict]:
    loc = OnlineLocalizer(
        ref,
        obs_sigma=OBS_SIGMA,
        process_noise_v=PROCESS_NOISE_V,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=WEIGHTS_NO_THR,
    )
    loc.reset()
    out = []
    for cache in caches:
        if len(cache["rc_t"]) < 2:
            continue
        prev_ts: float | None = None
        s_est = []
        for i in range(len(cache["rc_t"])):
            dt = float(cache["rc_t"][i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(cache["rc_t"][i])
            res = loc.update(cache["rc_sticks"][i].tolist(), dt, rate_profile=rate_profile)
            s_est.append(res.s)
        s_est = np.array(s_est)
        m = _metrics(cache["rc_s_real"], s_est, ref.L)
        out.append(dict(
            test_lap=cache["lap_index"],
            n_frames=len(s_est),
            duration_s=cache["duration_s"],
            **m,
        ))
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

def _build_grid(smoke: bool) -> list[tuple[str, int, int]]:
    if smoke:
        return [
            ("manual_best", 1, 5),
            ("manual_best", 3, 5),
        ]
    return [(sel, n, sw)
            for sel in LAP_SELECTIONS
            for n in N_LAPS_LEVELS
            for sw in SMOOTH_W_LEVELS]


def _build_pairs(smoke: bool) -> list[tuple[dict, dict]]:
    """List (ref_source, test_flight) pairs, excluding same-session leakage."""
    pairs: list[tuple[dict, dict]] = []
    for ref in REF_SOURCES:
        for flight in FLIGHTS:
            if ref["session"] == flight["session"]:
                continue
            pairs.append((ref, flight))
    if smoke:
        # one same_drone + one cross_drone
        pairs = [
            next(p for p in pairs if _condition(p[0]["drone"], p[1]["drone"]) == "same_drone"),
            next(p for p in pairs if _condition(p[0]["drone"], p[1]["drone"]) == "cross_drone"),
        ]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 6: reference building strategy")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 2 (ref,test) × 2 configs only")
    args = ap.parse_args()

    import pandas as pd

    grid = _build_grid(args.smoke)
    pairs = _build_pairs(args.smoke)

    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
    print(f"  configs (lap_sel × n_laps × smooth_w): {len(grid)}")
    print(f"  (ref, test) pairs: {len(pairs)}")
    print(f"  expected records:  {len(grid) * len(pairs) * N_LAST_LAPS}")
    print()

    # Cache per-flight raw data
    flight_cache: dict[int, FlightRawData] = {}
    print("Loading flight data...")
    for flight in FLIGHTS:
        # only load flights that participate as ref-source or test
        if not args.smoke or any(
            flight["flight_id"] == p[1]["flight_id"] for p in pairs
        ) or any(flight["session"] == r["session"] for r in REF_SOURCES):
            print(f"  Flight {flight['flight_id']}: {flight['drone']} × {flight['rate']}")
            flight_cache[flight["flight_id"]] = _load_flight(flight)

    # Cache per-ref-source laps once (for building references)
    ref_source_laps: dict[str, list[Lap]] = {}
    ref_source_rate: dict[str, dict] = {}
    ref_source_invert: dict[str, dict] = {}
    for ref_src in REF_SOURCES:
        matched = next(
            (frd for frd in flight_cache.values() if frd.flight["session"] == ref_src["session"]),
            None,
        )
        if matched is None:
            matched = _load_flight(
                next(f for f in FLIGHTS if f["session"] == ref_src["session"])
            )
            flight_cache[matched.flight["flight_id"]] = matched
        ref_source_laps[ref_src["name"]] = matched.laps
        ref_source_rate[ref_src["name"]] = matched.rate_profile
        ref_source_invert[ref_src["name"]] = matched.invert_lf
        print(f"  Ref source {ref_src['name']}: {len(matched.laps)} laps "
              f"(durations {min(l.duration for l in matched.laps):.2f}–"
              f"{max(l.duration for l in matched.laps):.2f} s, "
              f"invert={matched.invert_lf})")
    print()

    # Cache lap_selection results (heavy: manual_best needs LOO NN-greedy)
    print("Pre-computing lap rankings...")
    sel_cache: dict[tuple[str, str], list[int]] = {}  # (ref_name, selection) -> sorted indices
    for ref_src in REF_SOURCES:
        laps = ref_source_laps[ref_src["name"]]
        for sel in LAP_SELECTIONS if not args.smoke else ["manual_best"]:
            t0 = time.perf_counter()
            order = _pick_lap_indices(laps, selection=sel, n_laps=len(laps))
            sel_cache[(ref_src["name"], sel)] = order
            print(f"  {ref_src['name']} / {sel}: top-3 laps = "
                  f"{[laps[i].index for i in order[:3]]} "
                  f"({time.perf_counter() - t0:.1f}s)")
    print()

    records: list[RunRecord] = []
    total = len(pairs) * len(grid)
    done = 0
    t_start = time.perf_counter()

    for ref_src, flight in pairs:
        frd_test = flight_cache[flight["flight_id"]]
        ref_laps = ref_source_laps[ref_src["name"]]
        ref_rate = ref_source_rate[ref_src["name"]]
        ref_invert = ref_source_invert[ref_src["name"]]
        cond = _condition(ref_src["drone"], flight["drone"])

        for sel, n_laps, sw in grid:
            order = sel_cache[(ref_src["name"], sel)]
            indices = order[:n_laps]
            ref = _build_avg_reference(
                ref_laps, indices, smooth_w=sw,
                rate_profile=ref_rate, invert_lf=ref_invert,
            )
            caches = _compute_lap_caches(frd_test, ref)
            # IMPORTANT: pass the *test* flight's rate_profile to update()
            # so incoming sticks are mapped to physical observations using the
            # correct rate curves of the test drone.
            recs = _run_one(ref, caches, frd_test.rate_profile)
            for r in recs:
                records.append(RunRecord(
                    ref_source=ref_src["name"],
                    ref_drone=ref_src["drone"],
                    flight_id=flight["flight_id"],
                    test_drone=flight["drone"],
                    drone_cond=cond,
                    lap_selection=sel,
                    n_laps=n_laps,
                    smooth_w=sw,
                    test_lap=r["test_lap"],
                    n_frames=r["n_frames"],
                    duration_s=r["duration_s"],
                    median_err_m=r["median_err_m"],
                    p90_err_m=r["p90_err_m"],
                    jump_rate=r["jump_rate"],
                ))
            done += 1
            if done % 20 == 0 or done == total or args.smoke:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {ref_src['name']}→F{flight['flight_id']} "
                      f"({cond}) sel={sel} n={n_laps} sw={sw} "
                      f"| elapsed {elapsed:.0f}s ETA {eta:.0f}s")

    df = pd.DataFrame(records, columns=RunRecord._fields)
    suffix = "_smoke" if args.smoke else ""
    results_csv = OUT_DIR / f"results{suffix}.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nSaved {len(df)} records → {results_csv}")

    if args.smoke:
        print("\n=== Smoke test results ===")
        print(df[[
            "ref_source", "flight_id", "drone_cond",
            "lap_selection", "n_laps", "smooth_w",
            "test_lap", "median_err_m", "p90_err_m", "jump_rate",
        ]].to_string(index=False))
        return

    # ── Aggregation, plots, report ─────────────────────────────────────────────
    summary = (
        df.groupby(["ref_source", "ref_drone", "flight_id", "test_drone",
                    "drone_cond", "lap_selection", "n_laps", "smooth_w"])
        .agg(p90_err_m=("p90_err_m", "mean"),
             median_err_m=("median_err_m", "mean"),
             jump_rate=("jump_rate", "mean"),
             n_test_laps=("test_lap", "count"))
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    cond_summary = (
        df.groupby(["drone_cond", "lap_selection", "n_laps", "smooth_w"])
        .agg(p90_err_m=("p90_err_m", "mean"),
             median_err_m=("median_err_m", "mean"),
             jump_rate=("jump_rate", "mean"),
             n=("test_lap", "count"))
        .reset_index()
    )
    cond_summary.to_csv(OUT_DIR / "condition_summary.csv", index=False)

    _make_plots(df, summary, cond_summary)
    _write_report(df, summary, cond_summary)
    print(f"\nAll outputs in: {OUT_DIR}")


# ── Plots & Report ────────────────────────────────────────────────────────────

def _make_plots(df, summary, cond_summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_colors = {"same_drone": "#27ae60", "cross_drone": "#e74c3c"}

    # 1. p90 vs smooth_w (averaged over lap_selection × n_laps), per condition
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ("same_drone", "cross_drone"):
        sub = cond_summary[cond_summary["drone_cond"] == cond]
        agg = sub.groupby("smooth_w")["p90_err_m"].mean()
        ax.plot(agg.index, agg.values, "o-", color=cond_colors[cond], lw=2,
                label=cond, markersize=8)
        for x, y in zip(agg.index, agg.values):
            ax.text(x, y + 0.2, f"{y:.1f}", ha="center", fontsize=9, color=cond_colors[cond])
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1, label="target 15 m")
    ax.set_xlabel("smooth_w")
    ax.set_ylabel("p90 error, m")
    ax.set_title("Эффект сглаживания референса (среднее по lap_selection × n_laps)")
    ax.set_xticks(SMOOTH_W_LEVELS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "smooth_effect.png", dpi=140)
    plt.close(fig)

    # 2. p90 vs n_laps (averaged over lap_selection × smooth_w), per condition
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in ("same_drone", "cross_drone"):
        sub = cond_summary[cond_summary["drone_cond"] == cond]
        agg = sub.groupby("n_laps")["p90_err_m"].mean()
        ax.plot(agg.index, agg.values, "o-", color=cond_colors[cond], lw=2,
                label=cond, markersize=10)
        for x, y in zip(agg.index, agg.values):
            ax.text(x, y + 0.2, f"{y:.1f}", ha="center", fontsize=10, color=cond_colors[cond])
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1, label="target 15 m")
    ax.set_xlabel("n_laps (число усреднённых лапов)")
    ax.set_ylabel("p90 error, m")
    ax.set_title("Эффект усреднения по N лапам")
    ax.set_xticks(N_LAPS_LEVELS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "nlaps_effect.png", dpi=140)
    plt.close(fig)

    # 3. p90 by lap_selection (avg over smooth_w × n_laps), per condition
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LAP_SELECTIONS))
    w = 0.35
    for i, cond in enumerate(("same_drone", "cross_drone")):
        sub = cond_summary[cond_summary["drone_cond"] == cond]
        vals = [sub[sub["lap_selection"] == s]["p90_err_m"].mean() for s in LAP_SELECTIONS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=cond_colors[cond],
                      label=cond, edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(LAP_SELECTIONS)
    ax.set_ylabel("p90 error, m")
    ax.set_title("Стратегия выбора лапа (среднее по smooth_w × n_laps)")
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lap_selection.png", dpi=140)
    plt.close(fig)

    # 4. Heatmap: p90 (smooth_w × n_laps), one panel per (lap_sel × condition)
    fig, axes = plt.subplots(len(LAP_SELECTIONS), 2, figsize=(11, 11), sharex=True, sharey=True)
    fig.suptitle("p90 error: smooth_w × n_laps (тепловая карта)", fontweight="bold")
    for r, sel in enumerate(LAP_SELECTIONS):
        for c, cond in enumerate(("same_drone", "cross_drone")):
            sub = cond_summary[
                (cond_summary["lap_selection"] == sel)
                & (cond_summary["drone_cond"] == cond)
            ]
            grid = sub.pivot(index="n_laps", columns="smooth_w", values="p90_err_m")
            grid = grid.reindex(index=N_LAPS_LEVELS, columns=SMOOTH_W_LEVELS)
            ax = axes[r, c]
            im = ax.imshow(grid.values, cmap="RdYlGn_r", aspect="auto",
                           vmin=cond_summary["p90_err_m"].min(),
                           vmax=cond_summary["p90_err_m"].max())
            ax.set_xticks(range(len(SMOOTH_W_LEVELS)))
            ax.set_xticklabels(SMOOTH_W_LEVELS)
            ax.set_yticks(range(len(N_LAPS_LEVELS)))
            ax.set_yticklabels(N_LAPS_LEVELS)
            ax.set_title(f"{sel} / {cond}")
            for i in range(len(N_LAPS_LEVELS)):
                for j in range(len(SMOOTH_W_LEVELS)):
                    v = grid.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                                color="black")
            if c == 0:
                ax.set_ylabel("n_laps")
            if r == len(LAP_SELECTIONS) - 1:
                ax.set_xlabel("smooth_w")
    fig.colorbar(im, ax=axes, shrink=0.7, label="p90 error, m")
    fig.savefig(PLOTS_DIR / "heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _write_report(df, summary, cond_summary) -> None:
    same = cond_summary[cond_summary["drone_cond"] == "same_drone"]
    cross = cond_summary[cond_summary["drone_cond"] == "cross_drone"]

    # H1: p90 убывает с smooth_w (cross_drone)
    smooth_trend_cross = cross.groupby("smooth_w")["p90_err_m"].mean()
    slope_smooth_cross = _slope(np.array(smooth_trend_cross.index), smooth_trend_cross.values)
    h1 = "ПОДТВЕРЖДЕНО" if slope_smooth_cross < 0 else "ОТКЛОНЕНО"

    # H2: p90(n=3) < p90(n=1)
    n_trend = cond_summary.groupby(["drone_cond", "n_laps"])["p90_err_m"].mean().unstack()
    h2_same = "ПОДТВЕРЖДЕНО" if n_trend.loc["same_drone", 3] < n_trend.loc["same_drone", 1] else "ОТКЛОНЕНО"
    h2_cross = "ПОДТВЕРЖДЕНО" if n_trend.loc["cross_drone", 3] < n_trend.loc["cross_drone", 1] else "ОТКЛОНЕНО"

    # H3: fastest > median_time
    sel_p90 = cond_summary.groupby("lap_selection")["p90_err_m"].mean()
    h3 = "ПОДТВЕРЖДЕНО" if sel_p90["fastest"] > sel_p90["median_time"] else "ОТКЛОНЕНО"

    # H4: оптимальный smooth_w одинаков для same/cross
    opt_same = same.groupby("smooth_w")["p90_err_m"].mean().idxmin()
    opt_cross = cross.groupby("smooth_w")["p90_err_m"].mean().idxmin()
    h4 = "ПОДТВЕРЖДЕНО" if opt_same == opt_cross else "ОТКЛОНЕНО"

    # Best config per condition
    best_same = same.sort_values("p90_err_m").iloc[0]
    best_cross = cross.sort_values("p90_err_m").iloc[0]

    md = []
    md.append("# Эксперимент 6: Стратегия построения референса — Отчёт\n")
    md.append("## 1. Условия запуска\n")
    md.append("| Параметр | Значение |\n|---|---|\n"
              f"| Трасса | track-002 |\n"
              f"| Ref sources | {len(REF_SOURCES)} (по одному на дрон) |\n"
              f"| (ref, test) пар | {summary[['ref_source','flight_id']].drop_duplicates().shape[0]} |\n"
              f"| Конфигов референса | {len(LAP_SELECTIONS)} × {len(N_LAPS_LEVELS)} × {len(SMOOTH_W_LEVELS)} = "
              f"{len(LAP_SELECTIONS) * len(N_LAPS_LEVELS) * len(SMOOTH_W_LEVELS)} |\n"
              f"| Записей | {len(df)} |\n"
              f"| Фиксировано | RC+Rate, sigma=2.0, pnv=8.0, pns=1.5, weights=no_thr |\n")

    md.append("## 2. Лучшие конфигурации\n")
    md.append("| Условие | lap_selection | n_laps | smooth_w | p90, м | median, м | jump_rate |\n"
              "|---|---|---|---|---|---|---|\n"
              f"| same_drone  | {best_same['lap_selection']} | {int(best_same['n_laps'])} | "
              f"{int(best_same['smooth_w'])} | **{best_same['p90_err_m']:.2f}** | "
              f"{best_same['median_err_m']:.2f} | {best_same['jump_rate']:.3f} |\n"
              f"| cross_drone | {best_cross['lap_selection']} | {int(best_cross['n_laps'])} | "
              f"{int(best_cross['smooth_w'])} | **{best_cross['p90_err_m']:.2f}** | "
              f"{best_cross['median_err_m']:.2f} | {best_cross['jump_rate']:.3f} |\n")

    md.append("## 3. Эффекты факторов (среднее p90, м)\n")
    md.append("### smooth_w (по условию)\n")
    md.append("| smooth_w | same_drone | cross_drone |\n|---|---|---|\n")
    for sw in SMOOTH_W_LEVELS:
        s = same[same["smooth_w"] == sw]["p90_err_m"].mean()
        c = cross[cross["smooth_w"] == sw]["p90_err_m"].mean()
        md.append(f"| {sw} | {s:.2f} | {c:.2f} |\n")
    md.append("\n### n_laps (по условию)\n")
    md.append("| n_laps | same_drone | cross_drone |\n|---|---|---|\n")
    for n in N_LAPS_LEVELS:
        s = same[same["n_laps"] == n]["p90_err_m"].mean()
        c = cross[cross["n_laps"] == n]["p90_err_m"].mean()
        md.append(f"| {n} | {s:.2f} | {c:.2f} |\n")
    md.append("\n### lap_selection (по условию)\n")
    md.append("| lap_selection | same_drone | cross_drone |\n|---|---|---|\n")
    for sel in LAP_SELECTIONS:
        s = same[same["lap_selection"] == sel]["p90_err_m"].mean()
        c = cross[cross["lap_selection"] == sel]["p90_err_m"].mean()
        md.append(f"| {sel} | {s:.2f} | {c:.2f} |\n")

    md.append("\n## 4. Проверка гипотез\n")
    md.append(f"| # | Гипотеза | Результат |\n|---|---|---|\n"
              f"| H1 | ↑ smooth_w → ↓ p90 cross-drone (slope<0) | **{h1}** "
              f"(slope = {slope_smooth_cross:+.2f} м/ед.) |\n"
              f"| H2 | n_laps=3 лучше n_laps=1 (same_drone) | **{h2_same}** |\n"
              f"| H2 | n_laps=3 лучше n_laps=1 (cross_drone) | **{h2_cross}** |\n"
              f"| H3 | `fastest` хуже `median_time` (outlier effect) | **{h3}** |\n"
              f"| H4 | Оптимальный smooth_w одинаков для same/cross | **{h4}** "
              f"(same={opt_same}, cross={opt_cross}) |\n")

    md.append("\n## 5. Файлы\n"
              "- `results.csv` — per-test-lap метрики\n"
              "- `summary.csv` — агрегация по (ref, flight, конфиг)\n"
              "- `condition_summary.csv` — агрегация по (condition, конфиг)\n"
              "- `plots/smooth_effect.png` — p90 vs smooth_w\n"
              "- `plots/nlaps_effect.png` — p90 vs n_laps\n"
              "- `plots/lap_selection.png` — p90 по выбору лапа\n"
              "- `plots/heatmap.png` — тепловая карта (smooth_w × n_laps)\n")

    (OUT_DIR / "report.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

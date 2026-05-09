"""Experiment 7: Temporal Stability.

Research question
-----------------
Does the localizer's accuracy degrade as the number of accumulated laps
grows during a continuous flight? Are the first laps worse than later ones
(filter convergence)?

Design
------
- Track:               track-002 (Part_5_Эксперементальный).
- Long sessions:       MT_s003 (14 laps, Grom), MT_s004 (16 laps, Grom).
- Short Grom sessions: MT_s001 (5 laps), LO_s001 (7 laps).
- Reference modes:
    holdout   — ref = first 5 laps of the test session itself (no test-time
                leakage; tested on laps 6..N).
    external  — ref = avg over laps from OTHER same-drone, same-rate sessions
                (tested on ALL N laps → first laps visible for H3).
    cross     — ref = MT external avg, tested on LO_s001 (cross-drone).
- Filter modes:
    continuous — filter starts cold once, propagates through all test laps.
    reset      — filter.reset() between every test lap (control).
- Reference build (fixed from Exp 6 best config):
    lap_selection=median_time, n_laps=5, smooth_w=5.
- Localizer hyperparameters (from Exp 3):
    obs_sigma=2.0, process_noise_v=8.0, process_noise_s=1.5,
    channel_weights=no_thr [0,1,1,1].

Scenarios (10 runs total)
-------------------------
1.  s003  holdout  continuous     (test laps 6..14, n=9)
2.  s003  holdout  reset
3.  s003  external continuous     (test laps 1..14, n=14)
4.  s003  external reset
5.  s004  holdout  continuous     (test laps 6..16, n=11)
6.  s004  holdout  reset
7.  s004  external continuous     (test laps 1..16, n=16)
8.  s004  external reset
9.  LO_s001 cross  continuous     (test laps 1..7, n=7, cross-drone)
10. LO_s001 cross  reset

Usage (from project root)
-------------------------
    python tools/experiment_temporal.py --smoke    # one fast scenario
    python tools/experiment_temporal.py            # full
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
PART_5 = LIFTOFF / "Part_5_Эксперементальный"

OUT_DIR = Path(__file__).parent / "exp7_temporal"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0

# Optimal hyperparameters (Exp 3) and reference build (Exp 6)
OBS_SIGMA = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5
WEIGHTS_NO_THR = np.array([0.0, 1.0, 1.0, 1.0])
REF_LAP_SELECTION = "median_time"
REF_N_LAPS = 5
REF_SMOOTH_W = 5

# RC channel order in raw rc_channels.parquet
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0

# Maps invert_lf keys → column index in [thr, yaw, pitch, roll]
_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}


# ── Sessions ──────────────────────────────────────────────────────────────────

SESSIONS = {
    "MT_s001": dict(
        drone="MadTrainer", rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
    ),
    "MT_s003": dict(
        drone="MadTrainer", rate="Gromozeka_rate",
        session="2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-003",
    ),
    "MT_s004": dict(
        drone="MadTrainer", rate="Gromozeka_rate",
        session="2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-004",
    ),
    "LO_s001": dict(
        drone="LiftOff_200", rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
    ),
}


# ── Scenarios ─────────────────────────────────────────────────────────────────

# Each scenario describes (test session, ref mode, ref source spec).
# Ref source spec for "external"/"cross": list of (session_id, n_first_laps_to_use)
#   (None for n_first means "all laps").

SCENARIOS = [
    dict(
        scenario="s003_holdout",
        test="MT_s003",
        ref_kind="holdout",
        ref_spec=[("MT_s003", 5)],     # first 5 laps of the test session
        test_lap_start=6,              # 1-based: skip first 5 laps used as ref
        condition="same_drone",
    ),
    dict(
        scenario="s003_external",
        test="MT_s003",
        ref_kind="external",
        ref_spec=[("MT_s001", None), ("MT_s004", 5)],  # 5 + 5 = 10 laps avail
        test_lap_start=1,
        condition="same_drone",
    ),
    dict(
        scenario="s004_holdout",
        test="MT_s004",
        ref_kind="holdout",
        ref_spec=[("MT_s004", 5)],
        test_lap_start=6,
        condition="same_drone",
    ),
    dict(
        scenario="s004_external",
        test="MT_s004",
        ref_kind="external",
        ref_spec=[("MT_s001", None), ("MT_s003", 5)],
        test_lap_start=1,
        condition="same_drone",
    ),
    dict(
        scenario="LO_s001_cross",
        test="LO_s001",
        ref_kind="cross",
        ref_spec=[("MT_s003", 5), ("MT_s004", 5)],
        test_lap_start=1,
        condition="cross_drone",
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

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
                    p50_err_m=float("nan"), jump_rate=float("nan"))
    err = _wrap_error(np.abs(s_real - s_est), L)
    return {
        "median_err_m": float(np.median(err)),
        "p50_err_m":    float(np.median(err)),
        "p90_err_m":    float(np.percentile(err, 90)),
        "jump_rate":    float(np.mean(err > JUMP_THRESHOLD_M)),
    }


def _lap_arc_length(pos: np.ndarray) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def _compute_s_real(pos: np.ndarray, ref: Reference) -> np.ndarray:
    """Project xyz onto the reference arc parameter via NN."""
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


def _slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of OLS fit y ~ x."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _slope_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, two-sided p-value).

    p-value uses scipy.stats.linregress (Wald test). Falls back to NaN if scipy
    is unavailable.
    """
    if len(x) < 3:
        slope, intercept = _slope(x, y)
        return slope, intercept, float("nan")
    try:
        from scipy.stats import linregress
        res = linregress(x, y)
        return float(res.slope), float(res.intercept), float(res.pvalue)
    except Exception:
        slope, intercept = _slope(x, y)
        return slope, intercept, float("nan")


# ── Stage 1: load all laps for a session ───────────────────────────────────────

@dataclass
class SessionData:
    session_id: str
    drone: str
    rate_name: str
    rate_profile: dict
    invert_lf: dict
    laps: list[Lap]                       # ALL laps (after anomaly filter)
    rc_t_per_lap: list[np.ndarray]        # one entry per lap
    rc_sticks_per_lap: list[np.ndarray]
    lap_indices: list[int]                # absolute (1-based) indices from .index


def _load_session(session_id: str) -> SessionData:
    import pandas as pd

    spec = SESSIONS[session_id]
    session_dir = PART_5 / spec["session"]
    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    rate_profile = load_rate_profile(session_dir)
    invert_lf = _load_invert_lf(session_dir)

    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    rc_t_per_lap, rc_sticks_per_lap, lap_indices = [], [], []
    for lap in laps:
        mask = (rc_ts >= lap.t[0]) & (rc_ts < lap.t[-1])
        t_rc = rc_ts[mask]
        st_rc = rc_sticks[mask]
        if len(t_rc) < 2:
            t_rc = np.array([], dtype=float)
            st_rc = np.empty((0, 4), dtype=float)
        rc_t_per_lap.append(t_rc)
        rc_sticks_per_lap.append(st_rc)
        lap_indices.append(lap.index)

    return SessionData(
        session_id=session_id,
        drone=spec["drone"],
        rate_name=spec["rate"],
        rate_profile=rate_profile,
        invert_lf=invert_lf,
        laps=laps,
        rc_t_per_lap=rc_t_per_lap,
        rc_sticks_per_lap=rc_sticks_per_lap,
        lap_indices=lap_indices,
    )


# ── Stage 2: build reference (median_time, n=5, sw=5) ──────────────────────────

def _pick_median_time_indices(laps: list[Lap], n: int) -> list[int]:
    n = min(n, len(laps))
    durs = np.array([lap.duration for lap in laps])
    med = float(np.median(durs))
    order = np.argsort(np.abs(durs - med))
    return list(order[:n])


def _build_reference(
    ref_spec: list[tuple[str, int | None]],
    sessions: dict[str, SessionData],
    *,
    smooth_w: int = REF_SMOOTH_W,
    n_laps: int = REF_N_LAPS,
    grid_size: int = 1000,
) -> Reference:
    """Build a reference by averaging the median_time top-n laps from a union
    of (possibly multi-session) laps. All contributing laps must share the
    same rate_profile and invert_lf — otherwise raise.

    Steps
    -----
    1. Collect candidate laps (each with ``(lap, rate_profile, invert_lf)``).
    2. Verify rate/invert homogeneity (same drone class).
    3. Pick the ``n_laps`` whose duration is closest to the median.
    4. Resample each lap onto a unit [0..1] arc-length grid; average obs+pos.
    """
    candidate_laps: list[Lap] = []
    rate_profile = None
    invert_lf = None

    for sid, n_first in ref_spec:
        sd = sessions[sid]
        sub = sd.laps if n_first is None else sd.laps[:n_first]
        candidate_laps.extend(sub)

        # Homogeneity check via rate name
        if rate_profile is None:
            rate_profile = sd.rate_profile
            invert_lf = sd.invert_lf
            ref_rate_name = sd.rate_name
        else:
            if sd.rate_name != ref_rate_name:
                raise ValueError(
                    f"Heterogeneous rate profiles in ref_spec: {ref_rate_name} vs {sd.rate_name}"
                )
            if sd.invert_lf != invert_lf:
                raise ValueError(
                    f"Heterogeneous invert_lf in ref_spec: {invert_lf} vs {sd.invert_lf}"
                )

    assert rate_profile is not None and invert_lf is not None

    chosen_idx = _pick_median_time_indices(candidate_laps, n_laps)
    selected = [candidate_laps[i] for i in chosen_idx]

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
        t=t_dummy, obs=obs_avg.astype(np.float32), pos=pos_avg,
        smooth_w=smooth_w, feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1,
        rate_profile=rate_profile,
    )


# ── Stage 3: project test session onto reference ───────────────────────────────

def _build_test_caches(
    sd: SessionData,
    ref: Reference,
    test_lap_start_1based: int,
) -> list[dict]:
    """Build a list of per-lap caches starting at lap ``test_lap_start_1based``.

    Lap order in the cache list matches the chronological order in the session
    so the filter sees laps in real flight sequence.
    """
    test_laps = sd.laps[test_lap_start_1based - 1 :]
    test_rc_t = sd.rc_t_per_lap[test_lap_start_1based - 1 :]
    test_rc_st = sd.rc_sticks_per_lap[test_lap_start_1based - 1 :]
    test_lap_idx = sd.lap_indices[test_lap_start_1based - 1 :]

    if not test_laps:
        return []

    # Concatenate telem for projection lookup
    telem_t = np.concatenate([lap.t for lap in test_laps])
    telem_pos = np.vstack([lap.pos for lap in test_laps])
    telem_s_real = _compute_s_real(telem_pos, ref)

    caches = []
    for rel_idx, (lap, t_rc, st_rc, abs_idx) in enumerate(
        zip(test_laps, test_rc_t, test_rc_st, test_lap_idx), start=1
    ):
        if len(t_rc) < 2:
            caches.append(dict(
                test_lap_relative=rel_idx, lap_index=abs_idx,
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
            lap_index=abs_idx,
            rc_t=t_rc, rc_sticks=st_rc, rc_s_real=rc_s_real,
            duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


# ── Stage 4: run filter (continuous OR per-lap reset) ──────────────────────────

def _run(
    ref: Reference,
    caches: list[dict],
    rate_profile: dict,
    *,
    reset_per_lap: bool,
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
        if reset_per_lap:
            loc.reset()
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

class RunRecord(NamedTuple):
    scenario:           str
    test_session:       str
    test_drone:         str
    test_rate:          str
    ref_kind:           str    # holdout | external | cross
    condition:          str    # same_drone | cross_drone
    filter_mode:        str    # continuous | reset
    test_lap_relative:  int    # 1-based position in the test sequence
    lap_index_abs:      int    # absolute lap index in the session (1-based)
    n_frames:           int
    duration_s:         float
    median_err_m:       float
    p90_err_m:          float
    jump_rate:          float


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 7: temporal stability")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: run only the s003_holdout scenario, both modes")
    args = ap.parse_args()

    import pandas as pd

    scenarios = SCENARIOS
    if args.smoke:
        scenarios = [s for s in SCENARIOS if s["scenario"] == "s003_holdout"]

    # Determine which sessions are needed
    needed_ids: set[str] = set()
    for sc in scenarios:
        needed_ids.add(sc["test"])
        for sid, _ in sc["ref_spec"]:
            needed_ids.add(sid)

    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
    print(f"  scenarios: {len(scenarios)} (×2 filter_mode = {2 * len(scenarios)} runs)")
    print(f"  sessions needed: {sorted(needed_ids)}")
    print()

    print("Loading sessions...")
    sessions: dict[str, SessionData] = {}
    for sid in sorted(needed_ids):
        sd = _load_session(sid)
        sessions[sid] = sd
        durs = [round(l.duration, 2) for l in sd.laps]
        print(f"  {sid}: drone={sd.drone} rate={sd.rate_name} "
              f"laps={len(sd.laps)} durs={durs}")
    print()

    records: list[RunRecord] = []
    t_start = time.perf_counter()
    for sc in scenarios:
        sd_test = sessions[sc["test"]]
        for filter_mode in ("continuous", "reset"):
            t0 = time.perf_counter()
            ref = _build_reference(sc["ref_spec"], sessions)
            caches = _build_test_caches(sd_test, ref, sc["test_lap_start"])
            recs = _run(
                ref, caches, sd_test.rate_profile,
                reset_per_lap=(filter_mode == "reset"),
            )
            for r in recs:
                records.append(RunRecord(
                    scenario=sc["scenario"],
                    test_session=sc["test"],
                    test_drone=sd_test.drone,
                    test_rate=sd_test.rate_name,
                    ref_kind=sc["ref_kind"],
                    condition=sc["condition"],
                    filter_mode=filter_mode,
                    test_lap_relative=r["test_lap_relative"],
                    lap_index_abs=r["lap_index"],
                    n_frames=r["n_frames"],
                    duration_s=r["duration_s"],
                    median_err_m=r["median_err_m"],
                    p90_err_m=r["p90_err_m"],
                    jump_rate=r["jump_rate"],
                ))
            elapsed = time.perf_counter() - t0
            n_lap = len(recs)
            mean_p90 = float(np.mean([r["p90_err_m"] for r in recs])) if recs else float("nan")
            print(f"  [{sc['scenario']:>20} / {filter_mode:>10}] "
                  f"{n_lap} laps, mean p90={mean_p90:5.2f} m  ({elapsed:.1f}s)")
    print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")

    df = pd.DataFrame(records, columns=RunRecord._fields)
    suffix = "_smoke" if args.smoke else ""
    results_csv = OUT_DIR / f"results{suffix}.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nSaved {len(df)} per-lap records → {results_csv}")

    if args.smoke:
        print("\n=== Smoke test results ===")
        cols = ["scenario", "filter_mode", "test_lap_relative", "lap_index_abs",
                "n_frames", "median_err_m", "p90_err_m", "jump_rate"]
        print(df[cols].to_string(index=False, float_format="%.2f"))
        return

    # ── Aggregation, plots, report ─────────────────────────────────────────────
    summary = _aggregate(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Saved summary → {OUT_DIR / 'summary.csv'}")

    trends = _trend_analysis(df)
    trends.to_csv(OUT_DIR / "trends.csv", index=False)
    print(f"Saved trends → {OUT_DIR / 'trends.csv'}")

    _make_plots(df, trends)
    _write_report(df, summary, trends)
    print(f"\nAll outputs in: {OUT_DIR}")


# ── Aggregation and trend analysis ─────────────────────────────────────────────

def _aggregate(df):
    import pandas as pd
    summary = (
        df.groupby(["scenario", "test_session", "test_drone", "ref_kind",
                    "condition", "filter_mode"])
        .agg(
            n_laps=("test_lap_relative", "count"),
            mean_p90=("p90_err_m", "mean"),
            median_p90=("p90_err_m", "median"),
            std_p90=("p90_err_m", "std"),
            mean_median_err=("median_err_m", "mean"),
            mean_jump_rate=("jump_rate", "mean"),
        )
        .reset_index()
    )
    return summary


def _trend_analysis(df):
    """Per-(scenario, filter_mode) linear regression of p90 vs lap position
    plus the H3 indicator (mean p90 of relative laps 1-2 vs the rest)."""
    import pandas as pd
    rows = []
    for (sc, fm), grp in df.groupby(["scenario", "filter_mode"]):
        x = grp["test_lap_relative"].to_numpy(dtype=float)
        y = grp["p90_err_m"].to_numpy(dtype=float)
        slope, intercept, pval = _slope_pvalue(x, y)
        # H3 indicator (only meaningful when first 2 relative laps exist
        # AND there's room for "rest" — i.e. n_laps >= 4)
        first2 = grp[grp["test_lap_relative"] <= 2]["p90_err_m"].mean()
        rest = grp[grp["test_lap_relative"] > 2]["p90_err_m"].mean()
        first2_minus_rest = float(first2 - rest) if len(grp) >= 4 else float("nan")
        rows.append(dict(
            scenario=sc,
            filter_mode=fm,
            condition=grp["condition"].iloc[0],
            ref_kind=grp["ref_kind"].iloc[0],
            n_laps=int(len(grp)),
            mean_p90=float(y.mean()),
            slope_m_per_lap=slope,
            intercept_m=intercept,
            slope_pvalue=pval,
            p90_first2=float(first2) if not np.isnan(first2) else float("nan"),
            p90_rest=float(rest) if not np.isnan(rest) else float("nan"),
            p90_first2_minus_rest=first2_minus_rest,
        ))
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _make_plots(df, trends):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_colors = {"same_drone": "#27ae60", "cross_drone": "#e74c3c"}
    mode_styles = {"continuous": "-", "reset": "--"}

    # 1. p90 vs lap_relative — one panel per scenario, both modes overlaid
    scenarios = df["scenario"].drop_duplicates().tolist()
    n = len(scenarios)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)

    for i, sc in enumerate(scenarios):
        ax = axes[i // cols, i % cols]
        sub = df[df["scenario"] == sc]
        cond = sub["condition"].iloc[0]
        for fm in ("continuous", "reset"):
            grp = sub[sub["filter_mode"] == fm].sort_values("test_lap_relative")
            x = grp["test_lap_relative"].to_numpy()
            y = grp["p90_err_m"].to_numpy()
            color = cond_colors[cond]
            ls = mode_styles[fm]
            ax.plot(x, y, marker="o", color=color, linestyle=ls, lw=1.8,
                    label=f"{fm} (mean={y.mean():.1f}m)", alpha=0.85)
            # Linear fit overlay
            tr = trends[(trends["scenario"] == sc) & (trends["filter_mode"] == fm)].iloc[0]
            xfit = np.array([x.min(), x.max()])
            yfit = tr["intercept_m"] + tr["slope_m_per_lap"] * xfit
            ax.plot(xfit, yfit, color=color, linestyle=":", alpha=0.5,
                    label=f"slope={tr['slope_m_per_lap']:+.3f}m/lap (p={tr['slope_pvalue']:.2g})")
        ax.axhline(JUMP_THRESHOLD_M, color="grey", ls=":", lw=1)
        ax.set_title(f"{sc} ({cond})")
        ax.set_xlabel("relative lap #")
        ax.set_ylabel("p90 error, m")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide any extra empty axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Exp 7: p90 error per lap (continuous vs per-lap reset)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS_DIR / "p90_vs_lap.png", dpi=140)
    plt.close(fig)

    # 2. Convergence zoom — first 5 laps for external/cross scenarios (continuous)
    conv_df = df[
        df["ref_kind"].isin(["external", "cross"])
        & (df["filter_mode"] == "continuous")
        & (df["test_lap_relative"] <= 5)
    ]
    if not conv_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sc, grp in conv_df.groupby("scenario"):
            grp = grp.sort_values("test_lap_relative")
            cond = grp["condition"].iloc[0]
            ax.plot(grp["test_lap_relative"], grp["p90_err_m"],
                    marker="o", lw=2, color=cond_colors[cond],
                    label=f"{sc} ({cond})", alpha=0.85)
        ax.set_xlabel("relative lap # (test sequence)")
        ax.set_ylabel("p90 error, m")
        ax.set_title("Filter convergence (first 5 laps, continuous mode, external/cross ref)")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "convergence.png", dpi=140)
        plt.close(fig)

    # 3. Continuous vs reset — bar chart of mean p90 per scenario
    fig, ax = plt.subplots(figsize=(11, 5))
    scs = trends["scenario"].drop_duplicates().tolist()
    x = np.arange(len(scs))
    w = 0.35
    cont_vals = [trends[(trends["scenario"] == s) & (trends["filter_mode"] == "continuous")]
                 ["mean_p90"].iloc[0] for s in scs]
    reset_vals = [trends[(trends["scenario"] == s) & (trends["filter_mode"] == "reset")]
                  ["mean_p90"].iloc[0] for s in scs]
    b1 = ax.bar(x - w/2, cont_vals, w, label="continuous", color="#3498db", edgecolor="black")
    b2 = ax.bar(x + w/2, reset_vals, w, label="reset", color="#f39c12", edgecolor="black")
    for b, v in zip(b1, cont_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    for b, v in zip(b2, reset_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(scs, rotation=20, ha="right")
    ax.set_ylabel("mean p90 across all test laps, m")
    ax.set_title("Effect of filter reset between laps")
    ax.axhline(JUMP_THRESHOLD_M, color="grey", ls=":", lw=1, label="15 m target")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reset_vs_continuous.png", dpi=140)
    plt.close(fig)

    # 4. Slope summary plot — slope ± 95% CI? simpler: bar per (scenario, mode)
    fig, ax = plt.subplots(figsize=(11, 5))
    cont_slopes = [trends[(trends["scenario"] == s) & (trends["filter_mode"] == "continuous")]
                   ["slope_m_per_lap"].iloc[0] for s in scs]
    reset_slopes = [trends[(trends["scenario"] == s) & (trends["filter_mode"] == "reset")]
                    ["slope_m_per_lap"].iloc[0] for s in scs]
    b1 = ax.bar(x - w/2, cont_slopes, w, label="continuous", color="#3498db", edgecolor="black")
    b2 = ax.bar(x + w/2, reset_slopes, w, label="reset", color="#f39c12", edgecolor="black")
    for b, v in zip(b1, cont_slopes):
        y_text = v + (0.01 if v >= 0 else -0.04)
        ax.text(b.get_x() + b.get_width()/2, y_text, f"{v:+.3f}", ha="center", fontsize=8)
    for b, v in zip(b2, reset_slopes):
        y_text = v + (0.01 if v >= 0 else -0.04)
        ax.text(b.get_x() + b.get_width()/2, y_text, f"{v:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(scs, rotation=20, ha="right")
    ax.set_ylabel("slope of p90 vs lap, m/lap")
    ax.set_title("Temporal trend slope (positive = degradation)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "slopes.png", dpi=140)
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(df, summary, trends):
    # Hypothesis tests
    same_cont = trends[(trends["condition"] == "same_drone") & (trends["filter_mode"] == "continuous")]
    cross_cont = trends[(trends["condition"] == "cross_drone") & (trends["filter_mode"] == "continuous")]

    # H1: |slope| < 0.5 m/lap and slope_pvalue > 0.05 OR change <20% over horizon
    # We classify per-scenario, then aggregate.
    h1_per = []
    for _, row in trends.iterrows():
        change_pct = abs(row["slope_m_per_lap"] * (row["n_laps"] - 1)) / max(row["mean_p90"], 1e-3) * 100
        h1_pass = (change_pct < 20.0) or (row["slope_pvalue"] > 0.05)
        h1_per.append(dict(scenario=row["scenario"], filter_mode=row["filter_mode"],
                            slope=row["slope_m_per_lap"], change_pct=change_pct,
                            pvalue=row["slope_pvalue"], pass_h1=h1_pass))
    h1_overall = "ПОДТВЕРЖДЕНО" if all(x["pass_h1"] for x in h1_per) else "ОТКЛОНЕНО"

    # H2: |slope_cross| > |slope_same| (continuous mode only, where it's directly comparable)
    same_abs = float(np.mean(np.abs(same_cont["slope_m_per_lap"]))) if not same_cont.empty else float("nan")
    cross_abs = float(np.mean(np.abs(cross_cont["slope_m_per_lap"]))) if not cross_cont.empty else float("nan")
    h2 = "ПОДТВЕРЖДЕНО" if (cross_abs > same_abs and not np.isnan(cross_abs) and not np.isnan(same_abs)) else "ОТКЛОНЕНО"

    # H3: p90(lap1+2) > p90(rest) in continuous external/cross runs
    h3_runs = trends[
        trends["ref_kind"].isin(["external", "cross"])
        & (trends["filter_mode"] == "continuous")
    ]
    h3_pass_cnt = int((h3_runs["p90_first2_minus_rest"] > 0).sum())
    h3_total = len(h3_runs)
    h3_overall = "ПОДТВЕРЖДЕНО" if h3_pass_cnt > h3_total / 2 else "ОТКЛОНЕНО"

    # Reset effect: mean p90 difference (continuous − reset) per scenario
    reset_effect = []
    for sc in trends["scenario"].drop_duplicates():
        c = trends[(trends["scenario"] == sc) & (trends["filter_mode"] == "continuous")]["mean_p90"].iloc[0]
        r = trends[(trends["scenario"] == sc) & (trends["filter_mode"] == "reset")]["mean_p90"].iloc[0]
        reset_effect.append(dict(scenario=sc, continuous=c, reset=r, delta=c - r))

    md = []
    md.append("# Эксперимент 7: Временна́я стабильность — Отчёт\n\n")
    md.append("## 1. Условия запуска\n\n")
    md.append("| Параметр | Значение |\n|---|---|\n"
              f"| Трасса | track-002 (Part_5_Эксперементальный) |\n"
              f"| Сценариев | {len(SCENARIOS)} ×2 фильтра = {2*len(SCENARIOS)} запусков |\n"
              f"| Per-lap записей | {len(df)} |\n"
              f"| Длинные сессии (Grom rate) | MT_s003 (14 лапов, 385с), MT_s004 (16 лапов, 394с) |\n"
              f"| Cross-drone сессия | LO_s001 (7 лапов, 154с, Grom rate) |\n"
              f"| Реф-конфиг (Exp 6) | median_time, n_laps={REF_N_LAPS}, smooth_w={REF_SMOOTH_W} |\n"
              f"| Локализатор (Exp 3) | sigma=2.0, pnv=8.0, pns=1.5, weights=no_thr |\n\n")

    md.append("## 2. Тренды p90 по лапам (линейная регрессия)\n\n")
    md.append("| Сценарий | filter_mode | n | mean p90 | slope, м/лап | p-value | "
              "Δ_first2 (м) | вывод |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for _, row in trends.iterrows():
        verdict = "стабильно" if abs(row["slope_m_per_lap"]) < 0.3 or row["slope_pvalue"] > 0.05 \
                  else ("деградация" if row["slope_m_per_lap"] > 0 else "сходимость")
        d_first2 = row["p90_first2_minus_rest"]
        d_str = f"{d_first2:+.2f}" if not np.isnan(d_first2) else "—"
        md.append(f"| {row['scenario']} | {row['filter_mode']} | {row['n_laps']} | "
                  f"{row['mean_p90']:.2f} | {row['slope_m_per_lap']:+.3f} | "
                  f"{row['slope_pvalue']:.2g} | {d_str} | {verdict} |\n")
    md.append("\n")

    md.append("## 3. Эффект сброса фильтра (continuous − reset)\n\n")
    md.append("| Сценарий | continuous (mean p90) | reset (mean p90) | Δ, м |\n|---|---|---|---|\n")
    for r in reset_effect:
        md.append(f"| {r['scenario']} | {r['continuous']:.2f} | {r['reset']:.2f} | "
                  f"{r['delta']:+.2f} |\n")
    mean_delta = float(np.mean([r["delta"] for r in reset_effect]))
    md.append(f"\n**Среднее Δ = {mean_delta:+.2f} м** "
              f"({'continuous хуже reset' if mean_delta > 0 else 'continuous лучше или равен reset'}).\n\n")

    md.append("## 4. Проверка гипотез\n\n")
    md.append("| # | Гипотеза | Результат |\n|---|---|---|\n"
              f"| H1 | p90 не деградирует с ростом номера лапа (изменение <20% или p>0.05) | "
              f"**{h1_overall}** ({sum(1 for x in h1_per if x['pass_h1'])}/{len(h1_per)} подсценариев) |\n"
              f"| H2 | |slope_cross| > |slope_same| (continuous) | **{h2}** "
              f"(same={same_abs:.3f}, cross={cross_abs:.3f}) |\n"
              f"| H3 | Первые 2 лапа хуже последующих (continuous, ref включает lap 1) | "
              f"**{h3_overall}** ({h3_pass_cnt}/{h3_total} сценариев) |\n\n")

    md.append("## 5. Per-сценарий H1 разбор\n\n")
    md.append("| Сценарий | filter_mode | slope (м/лап) | Изменение по горизонту | p-value | H1 |\n"
              "|---|---|---|---|---|---|\n")
    for x in h1_per:
        md.append(f"| {x['scenario']} | {x['filter_mode']} | {x['slope']:+.3f} | "
                  f"{x['change_pct']:.1f}% | {x['pvalue']:.2g} | "
                  f"{'✓' if x['pass_h1'] else '✗'} |\n")
    md.append("\n")

    md.append("## 6. Файлы\n\n"
              "- `results.csv` — все per-lap метрики\n"
              "- `summary.csv` — агрегация по (сценарий, filter_mode)\n"
              "- `trends.csv` — slope, p-value, H3 индикатор для каждой связки\n"
              "- `plots/p90_vs_lap.png` — динамика p90 по лапам, оба режима\n"
              "- `plots/convergence.png` — первые 5 лапов крупно (только external/cross continuous)\n"
              "- `plots/reset_vs_continuous.png` — сравнение режимов фильтра\n"
              "- `plots/slopes.png` — наклоны трендов\n")

    (OUT_DIR / "report.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

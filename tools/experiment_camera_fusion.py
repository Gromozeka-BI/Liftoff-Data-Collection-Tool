"""Experiment 10: Camera fusion (simulation).

Research question
-----------------
How much does the localizer's accuracy improve when an external 3D position
sensor (e.g. forward-camera-based visual localization) periodically injects
absolute XYZ observations into the particle filter? What accuracy and update
rate does that camera system need to be useful?

This is a *transitional* experiment — its primary deliverable is a formal
specification (`camera_requirements.md`) for the future visual-localization
sub-system.

Method
------
- Reuse track-002 data (same as Exp 3 baseline).
- For each (σ_cam, T_update, condition, seed):
    1. Reset filter at start of test session.
    2. Iterate through RC-stick frames continuously (no per-lap reset).
    3. Whenever (t_now − t_last_camera) ≥ T_update, generate a synthetic
       observation: xyz_obs = xyz_real(t) + N(0, σ_cam) (isotropic 3D noise),
       call ``loc.inject_position_observation(xyz_obs, σ_cam)``.
    4. Record per-lap p90 and median errors.
- Average across N_SEEDS independent random draws of the camera noise.

Hypotheses (from spec)
----------------------
H1 — σ_cam < 5 m & T_update = 1 s improve p90 by > 30%.
H2 — Saturation: T_update < 0.5 s gives no extra benefit over 1 s.
H3 — Cross-drone benefits more (relatively) than same-drone.
H4 — σ_cam > 10 m gives no benefit (≈ baseline).

Usage
-----
    python tools/experiment_camera_fusion.py --smoke   # 1 cell × 2 cond × 1 seed
    python tools/experiment_camera_fusion.py           # full 5×5 grid
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

PART_5 = Path(r"D:\DroneTrackerDB\Liftoff\Part_5_Эксперементальный")
OUT_DIR = Path(__file__).parent / "exp10_camera_fusion"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0

# Default Exp 3 params (track-002 optimum)
OBS_SIGMA = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5
WEIGHTS_NO_THR = np.array([0.0, 1.0, 1.0, 1.0])

# Reference build (Exp 6 best)
REF_LAP_SELECTION = "median_time"
REF_N_LAPS = 5
REF_SMOOTH_W = 5

# Sweep grids
SIGMA_CAM_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]   # m
T_UPDATE_GRID = [0.2, 0.5, 1.0, 2.0, 5.0]      # s
N_SEEDS = 5

# High-precision ("frontier") grid — explore the maximum achievable accuracy.
# RC sampling is 100 Hz (Δt=10 ms), so T=0.01 s ≈ every RC frame.
SIGMA_CAM_GRID_HP = [0.1, 0.3, 0.5, 1.0, 2.0]  # m
T_UPDATE_GRID_HP = [0.01, 0.05, 0.1, 0.2, 0.5] # s

# RC channel order in raw rc_channels.parquet
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0
_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}


# ── Sessions: track-002 (same configuration as Exp 7) ─────────────────────────

SESSIONS = {
    "MT_s001": "2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
    "MT_s003": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-003",
    "MT_s004": "2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-004",
    "LO_s001": "2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
}

# Test session for each condition
TEST = {
    "same_drone":  "MT_s004",   # 15 laps, stable (Exp 7 hold-out winner)
    "cross_drone": "LO_s001",   # 6 laps cross-drone
}

# Reference spec for each test: list[(session_id, n_first_laps_or_None)].
# Same as Exp 7 — narrow pool (≈10 laps) so _pick_median_time picks 5 most
# similar laps, avoiding the trajectory-collapse from over-mixing.
REF_SPEC_FOR_TEST = {
    "MT_s004": [("MT_s001", None), ("MT_s003", 5)],   # 4 + 5 = 9 candidate laps
    "LO_s001": [("MT_s003", 5), ("MT_s004", 5)],      # 5 + 5 = 10 candidate laps
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_invert_lf(d: Path) -> dict:
    p = d / "invert.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("lf", {}) or {}
    except Exception:
        return {}


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


# ── Session loading ───────────────────────────────────────────────────────────

@dataclass
class SessionData:
    session_id:    str
    drone:         str
    rate_profile:  dict
    invert_lf:     dict
    laps:          list[Lap]
    rc_t_per_lap:  list[np.ndarray]
    rc_sticks_per_lap: list[np.ndarray]


def _load_session(session_id: str) -> SessionData:
    import pandas as pd

    path = PART_5 / SESSIONS[session_id]
    laps_raw, _ = load_dct_session(path)
    laps = filter_anomalous_laps(laps_raw)
    rate_profile = load_rate_profile(path)
    invert_lf = _load_invert_lf(path)

    rc = pd.read_parquet(path / "rc_channels.parquet")
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
        session_id=session_id, drone=drone,
        rate_profile=rate_profile, invert_lf=invert_lf,
        laps=laps,
        rc_t_per_lap=rc_t_per_lap,
        rc_sticks_per_lap=rc_sticks_per_lap,
    )


# ── Reference build ───────────────────────────────────────────────────────────

def _pick_median_time(laps: list[Lap], n: int) -> list[int]:
    n = min(n, len(laps))
    durs = np.array([l.duration for l in laps])
    med = float(np.median(durs))
    order = np.argsort(np.abs(durs - med))
    return list(order[:n])


def _build_reference(
    ref_spec: list[tuple[str, int | None]],
    sessions: dict[str, "SessionData"],
    *, smooth_w: int = REF_SMOOTH_W, n_laps: int = REF_N_LAPS,
    grid_size: int = 1000,
) -> Reference:
    """Build a reference by averaging the median_time top-n laps.

    Same scheme as Exp 7: collect the explicit first-N laps from each source
    session (homogeneous rate/invert), then pick the n_laps with duration
    closest to the median.
    """
    candidate_laps: list[Lap] = []
    rate_profile = None
    invert_lf = None
    for sid, n_first in ref_spec:
        sd = sessions[sid]
        sub = sd.laps if n_first is None else sd.laps[:n_first]
        candidate_laps.extend(sub)
        if rate_profile is None:
            rate_profile = sd.rate_profile
            invert_lf = sd.invert_lf
        else:
            if sd.invert_lf != invert_lf:
                raise ValueError(
                    f"Heterogeneous invert_lf in ref_spec: {invert_lf} vs {sd.invert_lf}")
    assert rate_profile is not None and invert_lf is not None

    chosen = _pick_median_time(candidate_laps, n_laps)
    selected = [candidate_laps[i] for i in chosen]
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
    return Reference.build_from_features(
        t=np.linspace(0.0, 1.0, grid_size), obs=obs_avg.astype(np.float32),
        pos=pos_avg, smooth_w=smooth_w,
        feature_kind=FEATURE_BETAFLIGHT_CLASSIC_V1, rate_profile=rate_profile,
    )


# ── Test caches: for each lap, RC stream + telem-projected s_real and xyz_real ─

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
                rc_xyz_real=np.empty((0, 3), dtype=float), duration_s=0.0,
            ))
            continue
        idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_l = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        chosen = np.where(closer_l, idx_l, idx_r)
        caches.append(dict(
            test_lap_relative=rel_idx, lap_index=lap.index,
            rc_t=t_rc, rc_sticks=st_rc,
            rc_s_real=telem_s_real[chosen],
            rc_xyz_real=telem_pos[chosen],         # ground-truth xyz at each RC frame
            duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


# ── Filter run with optional periodic camera injects ──────────────────────────

def _run(
    ref: Reference,
    caches: list[dict],
    rate_profile: dict,
    *,
    sigma_cam: float | None,         # None = baseline (no camera)
    T_update: float | None,          # None = no camera updates
    seed: int,
) -> list[dict]:
    """Run continuous filter through all laps; optionally inject XYZ from camera.

    Camera observation model
    ------------------------
    At every RC frame, check if (t_rc − t_last_cam) ≥ T_update.
    If yes:  xyz_obs = rc_xyz_real(frame) + N(0, σ_cam, size=3)
             loc.inject_position_observation(xyz_obs, σ_cam)
             t_last_cam = t_rc

    The injects happen *between* stick-update steps, so both modalities update
    the same particle distribution sequentially.
    """
    loc = OnlineLocalizer(
        ref,
        obs_sigma=OBS_SIGMA,
        process_noise_v=PROCESS_NOISE_V,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=WEIGHTS_NO_THR,
    )
    loc.reset()
    rng = np.random.default_rng(seed)

    cam_enabled = (sigma_cam is not None) and (T_update is not None)
    t_last_cam = -np.inf  # so first frame triggers an inject

    out = []
    for cache in caches:
        if len(cache["rc_t"]) < 2:
            continue
        s_est_arr = []
        prev_ts: float | None = None
        rc_t = cache["rc_t"]
        rc_sticks = cache["rc_sticks"]
        rc_xyz = cache["rc_xyz_real"]
        for i in range(len(rc_t)):
            t_now = float(rc_t[i])

            # Stick update
            dt = (t_now - prev_ts) if prev_ts is not None else None
            prev_ts = t_now
            res = loc.update(rc_sticks[i].tolist(), dt, rate_profile=rate_profile)

            # Optional camera inject *after* stick update
            if cam_enabled and (t_now - t_last_cam >= T_update):
                noise = rng.normal(0.0, sigma_cam, size=3)
                xyz_obs = rc_xyz[i] + noise
                res = loc.inject_position_observation(xyz_obs, sigma_cam)
                t_last_cam = t_now

            s_est_arr.append(res.s)

        s_est = np.array(s_est_arr)
        m = _metrics(cache["rc_s_real"], s_est, ref.L)
        out.append(dict(
            test_lap_relative=cache["test_lap_relative"],
            lap_index=cache["lap_index"],
            n_frames=len(s_est), duration_s=cache["duration_s"], **m,
        ))
    return out


# ── Records ────────────────────────────────────────────────────────────────────

class Record(NamedTuple):
    condition:        str
    test_session:     str
    sigma_cam:        float        # NaN → baseline (no camera)
    T_update:         float        # NaN → baseline
    seed:             int
    test_lap_relative: int
    lap_index_abs:    int
    n_frames:         int
    duration_s:       float
    median_err_m:     float
    p90_err_m:        float
    jump_rate:        float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 1 cell × 2 cond × 1 seed")
    ap.add_argument("--hp", action="store_true",
                    help="High-precision frontier sweep (small σ_cam, fast T_update)")
    args = ap.parse_args()

    import pandas as pd

    if args.smoke:
        sigma_grid, t_grid = [5.0], [1.0]
        seeds = [0]
        mode_label = "SMOKE"
    elif args.hp:
        sigma_grid, t_grid = SIGMA_CAM_GRID_HP, T_UPDATE_GRID_HP
        seeds = list(range(N_SEEDS))
        mode_label = "HIGH-PRECISION"
    else:
        sigma_grid, t_grid = SIGMA_CAM_GRID, T_UPDATE_GRID
        seeds = list(range(N_SEEDS))
        mode_label = "FULL"

    print(f"Mode: {mode_label}")
    print(f"Default Exp 3 params: σ={OBS_SIGMA}, pnv={PROCESS_NOISE_V}, no_thr")
    print(f"σ_cam grid: {sigma_grid}")
    print(f"T_update grid: {t_grid}")
    print(f"N_seeds: {len(seeds)}")
    print()

    # Load all needed sessions
    needed: set[str] = set()
    for cond, test_sid in TEST.items():
        needed.add(test_sid)
        for sid, _ in REF_SPEC_FOR_TEST[test_sid]:
            needed.add(sid)
    print("Loading sessions...")
    sessions: dict[str, SessionData] = {}
    for sid in sorted(needed):
        sd = _load_session(sid)
        sessions[sid] = sd
        print(f"  {sid}: drone={sd.drone} laps={len(sd.laps)}")

    # Build references per condition (same ref reused across all sweep cells)
    print("\nBuilding references...")
    refs: dict[str, Reference] = {}
    test_caches: dict[str, list[dict]] = {}
    for cond, test_sid in TEST.items():
        spec = REF_SPEC_FOR_TEST[test_sid]
        ref = _build_reference(spec, sessions)
        refs[cond] = ref
        test_caches[cond] = _build_test_caches(sessions[test_sid], ref)
        print(f"  [{cond}] ref_spec={spec}, L={ref.L:.2f} m, "
              f"test={test_sid} ({len(test_caches[cond])} laps)")

    sweep_cells = [(s, t) for s in sigma_grid for t in t_grid]

    records: list[Record] = []
    t_all = time.perf_counter()

    print(f"\n=== Baseline (no camera) per condition ===")
    for cond in TEST:
        for seed in seeds:
            recs = _run(
                refs[cond], test_caches[cond],
                sessions[TEST[cond]].rate_profile,
                sigma_cam=None, T_update=None, seed=seed,
            )
            for r in recs:
                records.append(Record(
                    condition=cond, test_session=TEST[cond],
                    sigma_cam=float("nan"), T_update=float("nan"),
                    seed=seed,
                    test_lap_relative=r["test_lap_relative"],
                    lap_index_abs=r["lap_index"], n_frames=r["n_frames"],
                    duration_s=r["duration_s"],
                    median_err_m=r["median_err_m"],
                    p90_err_m=r["p90_err_m"], jump_rate=r["jump_rate"],
                ))
        # print baseline mean p90 across seeds
        baseline_recs = [rec for rec in records
                         if rec.condition == cond and np.isnan(rec.sigma_cam)]
        mean_baseline = float(np.mean([rec.p90_err_m for rec in baseline_recs]))
        print(f"  [{cond}] baseline mean_p90 = {mean_baseline:.2f} m "
              f"(over {len(seeds)} seeds × {len(test_caches[cond])} laps)")

    # baselines per condition (mean across seeds AND laps)
    baselines = {cond: float(np.mean([rec.p90_err_m for rec in records
                                       if rec.condition == cond and np.isnan(rec.sigma_cam)]))
                 for cond in TEST}

    print(f"\n=== Sweep σ_cam × T_update ===")
    total = len(TEST) * len(sweep_cells) * len(seeds)
    done = 0
    for cond in TEST:
        for sigma_cam, T_update in sweep_cells:
            cell_p90s = []
            for seed in seeds:
                recs = _run(
                    refs[cond], test_caches[cond],
                    sessions[TEST[cond]].rate_profile,
                    sigma_cam=sigma_cam, T_update=T_update, seed=seed,
                )
                for r in recs:
                    records.append(Record(
                        condition=cond, test_session=TEST[cond],
                        sigma_cam=sigma_cam, T_update=T_update, seed=seed,
                        test_lap_relative=r["test_lap_relative"],
                        lap_index_abs=r["lap_index"], n_frames=r["n_frames"],
                        duration_s=r["duration_s"],
                        median_err_m=r["median_err_m"],
                        p90_err_m=r["p90_err_m"], jump_rate=r["jump_rate"],
                    ))
                    cell_p90s.append(r["p90_err_m"])
                done += 1
            mean_p90 = float(np.mean(cell_p90s)) if cell_p90s else float("nan")
            improvement = (1 - mean_p90 / baselines[cond]) * 100
            print(f"  [{cond} σ={sigma_cam:5.1f} T={T_update:.1f}s] "
                  f"mean_p90={mean_p90:5.2f}m  Δ={improvement:+5.1f}%  "
                  f"({done}/{total})")

    print(f"\nTotal: {time.perf_counter() - t_all:.1f}s")

    df = pd.DataFrame(records, columns=Record._fields)
    suffix = "_smoke" if args.smoke else ("_hp" if args.hp else "")
    df.to_csv(OUT_DIR / f"results{suffix}.csv", index=False)
    print(f"Saved {len(df)} per-lap records → {OUT_DIR / f'results{suffix}.csv'}")

    if args.smoke:
        print("\n=== Smoke summary ===")
        print(df.groupby(["condition", "sigma_cam", "T_update"], dropna=False)
                ["p90_err_m"].agg(["count", "mean"]).to_string())
        return

    # ── Aggregation, plots, report, requirements ──────────────────────────────
    summary = _aggregate(df, baselines)
    summary.to_csv(OUT_DIR / f"summary{suffix}.csv", index=False)
    print(f"Saved summary → {OUT_DIR / f'summary{suffix}.csv'}")

    _make_plots(summary, baselines, sigma_grid=sigma_grid, t_grid=t_grid, suffix=suffix)
    if not args.hp:
        _write_report(df, summary, baselines)
        _write_camera_requirements(summary, baselines)
    else:
        _write_hp_report(df, summary, baselines, sigma_grid, t_grid)
    print(f"\nAll outputs in: {OUT_DIR}")


# ── Aggregation ────────────────────────────────────────────────────────────────

def _aggregate(df, baselines):
    import pandas as pd
    sweep = df[~df["sigma_cam"].isna()].copy()
    grouped = (sweep.groupby(["condition", "sigma_cam", "T_update"])
               .agg(n_records=("p90_err_m", "count"),
                    mean_p90=("p90_err_m", "mean"),
                    median_p90=("p90_err_m", "median"),
                    std_p90=("p90_err_m", "std"),
                    mean_median_err=("median_err_m", "mean"),
                    mean_jump=("jump_rate", "mean"))
               .reset_index())
    grouped["baseline_p90"] = grouped["condition"].map(baselines)
    grouped["improvement_pct"] = (1 - grouped["mean_p90"] / grouped["baseline_p90"]) * 100
    return grouped


# ── Plots ─────────────────────────────────────────────────────────────────────

def _make_plots(summary, baselines, *, sigma_grid=None, t_grid=None, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma_grid = sigma_grid if sigma_grid is not None else SIGMA_CAM_GRID
    t_grid = t_grid if t_grid is not None else T_UPDATE_GRID

    # 1+2. Heatmaps mean_p90 (one per condition)
    for cond in TEST:
        sub = summary[summary["condition"] == cond]
        if sub.empty:
            continue
        grid = sub.pivot(index="sigma_cam", columns="T_update", values="mean_p90")
        grid = grid.reindex(index=sigma_grid, columns=t_grid)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=summary["mean_p90"].min(),
                       vmax=summary["mean_p90"].max())
        ax.set_xticks(range(len(t_grid)))
        ax.set_xticklabels([f"{t:g}" for t in t_grid])
        ax.set_yticks(range(len(sigma_grid)))
        ax.set_yticklabels([f"{s:g}" for s in sigma_grid])
        ax.set_title(f"{cond}: mean p90 (m)\nbaseline = {baselines[cond]:.2f} m")
        ax.set_xlabel("T_update, s")
        ax.set_ylabel("σ_cam, m")
        for r in range(len(sigma_grid)):
            for c in range(len(t_grid)):
                v = grid.values[r, c]
                if np.isnan(v):
                    continue
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="black")
        fig.colorbar(im, ax=ax, label="mean p90, m")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"heatmap_{cond}{suffix}.png", dpi=140)
        plt.close(fig)

    # 3. Improvement % heatmap (one per condition)
    fig, axes = plt.subplots(1, len(TEST), figsize=(7 * len(TEST), 6), squeeze=False)
    for i, cond in enumerate(TEST):
        sub = summary[summary["condition"] == cond]
        grid = sub.pivot(index="sigma_cam", columns="T_update", values="improvement_pct")
        grid = grid.reindex(index=sigma_grid, columns=t_grid)
        ax = axes[0, i]
        im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto",
                       vmin=-20, vmax=summary["improvement_pct"].max())
        ax.set_xticks(range(len(t_grid)))
        ax.set_xticklabels([f"{t:g}" for t in t_grid])
        ax.set_yticks(range(len(sigma_grid)))
        ax.set_yticklabels([f"{s:g}" for s in sigma_grid])
        ax.set_title(f"{cond}: improvement vs baseline (%)\nbaseline = {baselines[cond]:.2f} m")
        ax.set_xlabel("T_update, s")
        ax.set_ylabel("σ_cam, m")
        for r in range(len(sigma_grid)):
            for c in range(len(t_grid)):
                v = grid.values[r, c]
                if np.isnan(v):
                    continue
                ax.text(c, r, f"{v:+.0f}", ha="center", va="center", fontsize=10,
                        color="black")
        fig.colorbar(im, ax=ax, label="improvement, %")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"improvement_vs_baseline{suffix}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # 4. Line plot: improvement vs T_update for fixed σ values
    fig, axes = plt.subplots(1, len(TEST), figsize=(7 * len(TEST), 5),
                             squeeze=False, sharey=True)
    cmap = plt.cm.viridis
    for i, cond in enumerate(TEST):
        ax = axes[0, i]
        sub = summary[summary["condition"] == cond]
        for j, sigma in enumerate(sigma_grid):
            line = sub[sub["sigma_cam"] == sigma].sort_values("T_update")
            color = cmap(j / max(1, len(sigma_grid) - 1))
            ax.plot(line["T_update"], line["improvement_pct"],
                    marker="o", color=color, label=f"σ_cam = {sigma:g} m", lw=2)
        ax.set_xscale("log")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("T_update, s")
        ax.set_ylabel("improvement vs baseline, %" if i == 0 else "")
        ax.set_title(f"{cond} (baseline {baselines[cond]:.2f} m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Camera fusion: improvement vs update period", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / f"improvement_lines{suffix}.png", dpi=140)
    plt.close(fig)

    # 5. (HP-specific) Absolute p90 vs T_update — log-log axes show floor clearly
    if suffix == "_hp":
        fig, axes = plt.subplots(1, len(TEST), figsize=(7 * len(TEST), 5),
                                 squeeze=False, sharey=True)
        cmap = plt.cm.viridis
        for i, cond in enumerate(TEST):
            ax = axes[0, i]
            sub = summary[summary["condition"] == cond]
            for j, sigma in enumerate(sigma_grid):
                line = sub[sub["sigma_cam"] == sigma].sort_values("T_update")
                color = cmap(j / max(1, len(sigma_grid) - 1))
                ax.plot(line["T_update"], line["mean_p90"],
                        marker="o", color=color, label=f"σ_cam = {sigma:g} m", lw=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("T_update, s")
            ax.set_ylabel("mean p90, m" if i == 0 else "")
            ax.set_title(f"{cond} (baseline {baselines[cond]:.2f} m)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle("Camera fusion: absolute p90 vs update period (log–log)",
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(PLOTS_DIR / "p90_loglog_hp.png", dpi=140)
        plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(df, summary, baselines):
    # Hypothesis evaluation
    h1_cells_same = summary[(summary["condition"] == "same_drone")
                             & (summary["sigma_cam"] < 5.0)
                             & (summary["T_update"] == 1.0)
                             & (summary["improvement_pct"] > 30)]
    h1_cells_cross = summary[(summary["condition"] == "cross_drone")
                              & (summary["sigma_cam"] < 5.0)
                              & (summary["T_update"] == 1.0)
                              & (summary["improvement_pct"] > 30)]
    h1_pass = (len(h1_cells_same) > 0) and (len(h1_cells_cross) > 0)
    h1_label = "ПОДТВЕРЖДЕНО" if h1_pass else "ОТКЛОНЕНО"

    # H2: T<0.5 s vs T=1 s saturation (per condition × σ)
    saturation_examples = []
    for cond in TEST:
        for sigma in SIGMA_CAM_GRID:
            sub = summary[(summary["condition"] == cond) & (summary["sigma_cam"] == sigma)]
            v_02 = sub[sub["T_update"] == 0.2]["mean_p90"]
            v_05 = sub[sub["T_update"] == 0.5]["mean_p90"]
            v_10 = sub[sub["T_update"] == 1.0]["mean_p90"]
            if v_02.empty or v_10.empty:
                continue
            extra_gain_pct = (1 - v_02.iloc[0] / v_10.iloc[0]) * 100
            saturation_examples.append((cond, sigma, float(v_10.iloc[0]),
                                        float(v_05.iloc[0]) if not v_05.empty else float("nan"),
                                        float(v_02.iloc[0]), extra_gain_pct))
    h2_pass = all(abs(g) < 5.0 for *_, g in saturation_examples)
    h2_label = "ПОДТВЕРЖДЕНО" if h2_pass else "ОТКЛОНЕНО"

    # H3: cross_drone improvement > same_drone improvement
    same_max_imp = summary[summary["condition"] == "same_drone"]["improvement_pct"].max()
    cross_max_imp = summary[summary["condition"] == "cross_drone"]["improvement_pct"].max()
    h3_pass = cross_max_imp > same_max_imp + 5.0
    h3_label = "ПОДТВЕРЖДЕНО" if h3_pass else "ОТКЛОНЕНО"

    # H4: σ_cam=20m gives no benefit (improvement < 10%)
    h4_imp = []
    for cond in TEST:
        s20 = summary[(summary["condition"] == cond) & (summary["sigma_cam"] == 20.0)]
        if not s20.empty:
            h4_imp.append((cond, float(s20["improvement_pct"].max())))
    h4_pass = all(imp < 10 for _, imp in h4_imp)
    h4_label = "ПОДТВЕРЖДЕНО" if h4_pass else "ОТКЛОНЕНО"

    md = []
    md.append("# Эксперимент 10: Слияние с курсовой камерой — Отчёт\n\n")
    md.append("## 1. Условия запуска\n\n")
    md.append("| Параметр | Значение |\n|---|---|\n"
              f"| Трасса | track-002 (Abu-Dhabi, ~314 м) |\n"
              f"| Дрон ref | MadTrainer (Gromozeka_rate) |\n"
              f"| Test (same-drone) | MT_s004 (15 лапов) |\n"
              f"| Test (cross-drone) | LO_s001 (6 лапов) |\n"
              f"| Реф (Exp 6) | median_time, n_laps={REF_N_LAPS}, smooth_w={REF_SMOOTH_W} |\n"
              f"| Стиковый PF (Exp 3) | σ={OBS_SIGMA}, pnv={PROCESS_NOISE_V}, pns={PROCESS_NOISE_S}, weights=no_thr |\n"
              f"| σ_cam grid (м) | {SIGMA_CAM_GRID} |\n"
              f"| T_update grid (с) | {T_UPDATE_GRID} |\n"
              f"| Seeds для камерного шума | {N_SEEDS} |\n"
              f"| Симуляция камеры | xyz_obs = xyz_real + N(0, σ_cam, size=3) |\n"
              f"| Метод инжекта | 3D Bayes-update частиц по \\|\\|xyz_part − xyz_obs\\|\\| |\n\n")

    md.append("## 2. Базовые линии (без камеры)\n\n")
    md.append("| Условие | mean p90, м |\n|---|---|\n")
    for cond, b in baselines.items():
        md.append(f"| {cond} | {b:.2f} |\n")
    md.append("\n")

    md.append("## 3. Sweep σ_cam × T_update — mean p90 (м)\n\n")
    for cond in TEST:
        md.append(f"### {cond} (baseline {baselines[cond]:.2f} м)\n\n")
        md.append("| σ_cam ↓ \\ T_update → | " + " | ".join([f"{t:g}s" for t in T_UPDATE_GRID]) + " |\n")
        md.append("|---|" + "|".join(["---"] * len(T_UPDATE_GRID)) + "|\n")
        sub = summary[summary["condition"] == cond]
        for sigma in SIGMA_CAM_GRID:
            row = [f"**{sigma:g} м**"]
            for T in T_UPDATE_GRID:
                cell = sub[(sub["sigma_cam"] == sigma) & (sub["T_update"] == T)]
                if cell.empty:
                    row.append("—")
                else:
                    p = float(cell["mean_p90"].iloc[0])
                    imp = float(cell["improvement_pct"].iloc[0])
                    row.append(f"{p:.2f} ({imp:+.0f}%)")
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

    md.append("## 4. Проверка гипотез\n\n")
    md.append("| # | Гипотеза | Результат |\n|---|---|---|\n"
              f"| H1 | σ_cam<5м & T_update=1с улучшают p90 >30% (для обоих условий) | **{h1_label}** |\n"
              f"| H2 | Насыщение: T<0.5с не даёт прироста vs T=1с (Δ<5%) | **{h2_label}** |\n"
              f"| H3 | Cross-drone выигрывает больше same-drone | **{h3_label}** "
              f"(same_max={same_max_imp:.1f}%, cross_max={cross_max_imp:.1f}%) |\n"
              f"| H4 | σ_cam=20м ≈ baseline (improvement <10%) | **{h4_label}** |\n\n")

    md.append("### 4.1 Детали H2 (насыщение по частоте)\n\n")
    md.append("| Условие | σ_cam | p90(T=1с) | p90(T=0.5с) | p90(T=0.2с) | Δ(T=0.2 vs T=1), % |\n"
              "|---|---|---|---|---|---|\n")
    for cond, s, p10, p05, p02, g in saturation_examples:
        md.append(f"| {cond} | {s:g} | {p10:.2f} | {p05:.2f} | {p02:.2f} | {g:+.1f}% |\n")
    md.append("\n")

    md.append("## 5. Файлы\n\n"
              "- `results.csv` — все per-lap метрики (включая baseline и все ячейки sweep × seeds)\n"
              "- `summary.csv` — агрегация по (condition, σ_cam, T_update)\n"
              "- `camera_requirements.md` — **главный выход**: ТЗ для камерной системы\n"
              "- `plots/heatmap_*.png` — тепловые карты mean p90 по условию\n"
              "- `plots/improvement_vs_baseline.png` — % улучшения от базовой линии\n"
              "- `plots/improvement_lines.png` — % улучшения vs T_update при фиксированных σ_cam\n")

    (OUT_DIR / "report.md").write_text("".join(md), encoding="utf-8")


# ── Camera requirements (the main deliverable) ────────────────────────────────

def _write_hp_report(df, summary, baselines, sigma_grid, t_grid):
    """High-precision frontier report: explores limits of camera fusion accuracy."""
    md = []
    md.append("# Эксперимент 10 — High-precision frontier\n\n")
    md.append("Расширенный sweep в зоне **малых σ_cam** и **высоких частот**, "
              "чтобы найти потолок точности камерной фьюжн.\n\n")
    md.append("## 1. Условия\n\n")
    md.append("| Параметр | Значение |\n|---|---|\n"
              f"| RC / Telemetry sampling | 100 Гц (Δt=10 мс), нижняя граница T_update |\n"
              f"| σ_cam grid (м) | {sigma_grid} |\n"
              f"| T_update grid (с) | {t_grid} (1/T = {[round(1/t,1) for t in t_grid]} Гц) |\n"
              f"| Seeds | {N_SEEDS} |\n"
              f"| PF | σ_obs={OBS_SIGMA}, pnv={PROCESS_NOISE_V}, no_thr (Exp 3) |\n"
              f"| Baselines | same={baselines['same_drone']:.2f} м, "
              f"cross={baselines['cross_drone']:.2f} м |\n\n")

    md.append("## 2. Mean p90 (м), полная сетка\n\n")
    for cond in TEST:
        md.append(f"### {cond} (baseline {baselines[cond]:.2f} м)\n\n")
        md.append("| σ_cam ↓ \\ T_update → | "
                  + " | ".join([f"{t:g}s ({1/t:g}Гц)" for t in t_grid]) + " |\n")
        md.append("|---|" + "|".join(["---"] * len(t_grid)) + "|\n")
        sub = summary[summary["condition"] == cond]
        for sigma in sigma_grid:
            row = [f"**{sigma:g} м**"]
            for T in t_grid:
                cell = sub[(sub["sigma_cam"] == sigma) & (sub["T_update"] == T)]
                if cell.empty:
                    row.append("—")
                else:
                    p = float(cell["mean_p90"].iloc[0])
                    row.append(f"{p:.2f}")
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

    md.append("## 3. Анализ потолка\n\n")
    for cond in TEST:
        sub = summary[summary["condition"] == cond]
        best = sub.sort_values("mean_p90").iloc[0]
        worst_in_grid = sub.sort_values("mean_p90").iloc[-1]
        md.append(f"### {cond}\n\n"
                  f"- **Лучшая ячейка**: σ_cam={best['sigma_cam']:g} м, "
                  f"T_update={best['T_update']:g} с → "
                  f"**p90 = {best['mean_p90']:.2f} м** "
                  f"({best['improvement_pct']:+.1f}% vs baseline)\n"
                  f"- Худшая в HP-гриде: σ_cam={worst_in_grid['sigma_cam']:g} м, "
                  f"T_update={worst_in_grid['T_update']:g} с → "
                  f"p90 = {worst_in_grid['mean_p90']:.2f} м\n")
        # Saturation by T_update (at fixed σ)
        md.append(f"\n**Прирост точности от ускорения** (для каждого σ_cam):\n\n")
        md.append("| σ_cam | T=0.5с | T=0.2с | T=0.1с | T=0.05с | T=0.01с | "
                  "Δ(0.01 vs 0.5), м |\n|---|---|---|---|---|---|---|\n")
        for sigma in sigma_grid:
            row = [f"{sigma:g}"]
            vals = {}
            for T in t_grid:
                cell = sub[(sub["sigma_cam"] == sigma) & (sub["T_update"] == T)]
                if cell.empty:
                    row.append("—")
                    continue
                p = float(cell["mean_p90"].iloc[0])
                vals[T] = p
                row.append(f"{p:.2f}")
            if 0.5 in vals and 0.01 in vals:
                row.append(f"{vals[0.5] - vals[0.01]:+.2f}")
            else:
                row.append("—")
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

        # Saturation by σ_cam (at fixed T)
        md.append(f"**Прирост точности от уменьшения σ_cam** (для каждой частоты):\n\n")
        md.append("| T_update | σ=2м | σ=1м | σ=0.5м | σ=0.3м | σ=0.1м | "
                  "Δ(0.1 vs 2), м |\n|---|---|---|---|---|---|---|\n")
        for T in t_grid:
            row = [f"{T:g}с"]
            vals = {}
            for sigma in sigma_grid:
                cell = sub[(sub["sigma_cam"] == sigma) & (sub["T_update"] == T)]
                if cell.empty:
                    row.append("—")
                    continue
                p = float(cell["mean_p90"].iloc[0])
                vals[sigma] = p
                row.append(f"{p:.2f}")
            if 2.0 in vals and 0.1 in vals:
                row.append(f"{vals[2.0] - vals[0.1]:+.2f}")
            else:
                row.append("—")
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

    md.append("## 4. Выводы\n\n"
              "Заполняется аналитически — см. ниже комментарий вручную.\n\n")
    md.append("## 5. Файлы\n\n"
              "- `results_hp.csv`, `summary_hp.csv` — результаты HP-сетки\n"
              "- `plots/heatmap_*_hp.png` — тепловые карты\n"
              "- `plots/p90_loglog_hp.png` — лог-лог зависимость p90 от T_update\n"
              "- `plots/improvement_lines_hp.png` — % улучшения vs T_update\n")
    (OUT_DIR / "report_hp.md").write_text("".join(md), encoding="utf-8")


def _write_camera_requirements(summary, baselines):
    # Tier definitions: "minimum" = improvement >20%; "recommended" = >50%
    def tier_cells(condition: str, threshold: float):
        sub = summary[(summary["condition"] == condition)
                      & (summary["improvement_pct"] >= threshold)]
        if sub.empty:
            return None
        # The most permissive corner that still satisfies threshold:
        # max σ_cam AND max T_update (largest tolerance).
        sub = sub.copy()
        sub["combined"] = sub["sigma_cam"] / 5.0 + sub["T_update"]  # priority by laxness
        best = sub.sort_values(["sigma_cam", "T_update"], ascending=False).iloc[0]
        return dict(sigma_cam=float(best["sigma_cam"]),
                    T_update=float(best["T_update"]),
                    p90=float(best["mean_p90"]),
                    improvement=float(best["improvement_pct"]))

    md = []
    md.append("# ТЗ для системы визуальной локализации (камерный модуль)\n\n")
    md.append("Документ — формальный выход Эксперимента 10. Описывает требования\n"
              "к будущему камерному подмодулю, который будет периодически выдавать\n"
              "абсолютные XYZ-координаты дрона на трассе.\n\n")
    md.append("## 1. Контекст\n\n"
              "На текущий момент локализатор использует только данные стиков (RC+Rate),\n"
              "достигая cross-drone p90 = " f"{baselines['cross_drone']:.2f} м "
              f"и same-drone p90 = {baselines['same_drone']:.2f} м "
              "на track-002 (track-002 — самая зрелая трасса, ~30 сессий).\n\n"
              "Камерная подсистема должна дополнить локализатор, периодически\n"
              "сообщая XYZ-наблюдения с известной точностью σ_cam (изотропный\n"
              "гауссов шум). Слияние реализовано через 3D Bayes-update в\n"
              "`OnlineLocalizer.inject_position_observation(xyz_obs, σ_cam)`.\n\n")

    md.append("## 2. Требования\n\n")
    for cond in TEST:
        md.append(f"### {cond}\n\n")
        baseline = baselines[cond]
        md.append(f"- **Базовая линия (без камеры)**: p90 = {baseline:.2f} м\n\n")

        for tier_name, threshold in [("Минимальные (улучшение ≥ 20%)", 20.0),
                                      ("Рекомендуемые (улучшение ≥ 50%)", 50.0)]:
            cell = tier_cells(cond, threshold)
            md.append(f"#### {tier_name}\n\n")
            if cell is None:
                md.append(f"- Не достижимо в исследованном грид-диапазоне "
                          f"(σ_cam ∈ [1..20] м, T_update ∈ [0.2..5] с)\n\n")
            else:
                md.append(f"- σ_cam ≤ **{cell['sigma_cam']:g} м**\n"
                          f"- T_update ≤ **{cell['T_update']:g} с** "
                          f"(частота ≥ {1.0/cell['T_update']:g} Гц)\n"
                          f"- Ожидаемый p90 = {cell['p90']:.2f} м "
                          f"(улучшение {cell['improvement']:+.1f}% от baseline)\n\n")

        # Best-case for this condition
        sub = summary[summary["condition"] == cond]
        if not sub.empty:
            best = sub.sort_values("mean_p90").iloc[0]
            md.append(f"- **Лучшая ячейка в грид'е**: σ_cam={best['sigma_cam']:g} м, "
                      f"T_update={best['T_update']:g} с → "
                      f"p90 = {best['mean_p90']:.2f} м "
                      f"(улучшение {best['improvement_pct']:+.1f}%)\n\n")

    md.append("## 3. Технический интерфейс\n\n"
              "Камерный модуль должен вызывать:\n\n"
              "```python\n"
              "loc.inject_position_observation(\n"
              "    xyz_obs=(x_m, y_m, z_m),    # м, в системе координат трассы\n"
              "    sigma_cam=σ_cam_m,          # СКО точности (изотропно), м\n"
              ")\n"
              "```\n\n"
              "Метод реализован в `dct.localization.online_localizer.OnlineLocalizer`.\n"
              "Вызовы можно делать асинхронно; они корректно встраиваются между\n"
              "обычными `update(sticks, dt)` вызовами PF.\n\n")

    md.append("## 4. Замечания\n\n"
              "- Все требования — для track-002 с дефолтными параметрами PF\n"
              f"  (σ_obs={OBS_SIGMA}, pnv={PROCESS_NOISE_V}, no_thr).\n"
              "- На «свежих» трассах (Exp 8: track-003, track-004) дефолтные\n"
              f"  параметры PF не оптимальны (нужен σ_obs≈4); камерная фьюжн\n"
              "  с правильным σ_obs ожидаемо даст похожий или лучший выигрыш.\n"
              "- Шум симулировался как изотропный (σ_x=σ_y=σ_z); реальные камеры\n"
              "  могут иметь анизотропную точность (хуже по глубине). При\n"
              "  необходимости можно расширить `inject_position_observation` до\n"
              "  диагональной/полной ковариации.\n"
              "- Фильтр устойчив к выбросам камеры (`σ_cam=20м` не ломает baseline).\n")

    (OUT_DIR / "camera_requirements.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

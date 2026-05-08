"""Offline sweep of ParticleFilter obs_sigma × channel_weights on two RC-only sessions.

Usage (run from the project root):
    python tools/test_localizer_sigma.py

Outputs:
    - Table printed to stdout
    - tools/sigma_test_results.csv

Sessions tested:
    A - Part_2/session-007  (same session the reference was built from, lap 1 only)
    B - Part_2/session-005  (cross-session, different drone RedSheep_200)

Reference: tracks/track-001/references/RedRC_1.npz

Channel order: [Thr, Yaw, Pit, Roll]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.lap_loader import load_rc_only_session
from dct.localization.online_localizer import OnlineLocalizer
from dct.rate_features import load_rate_profile

# ── configuration ─────────────────────────────────────────────────────────────
REF_PATH = ROOT / "tracks" / "track-001" / "references" / "RedRC_1.npz"

SESSION_A = Path(
    r"D:\MAD Hall Track_1\Part_2"
    r"\2026-05-06_pilot-RedSeep_drone-MadTrainer_track-track-001_session-007"
)
SESSION_B = Path(
    r"D:\MAD Hall Track_1\Part_3"
    r"\2026-05-06_pilot-RedSeep_drone-RedSheep_200_track-track-001_session-005"
)

OBS_SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Weight presets: (label, [Thr, Yaw, Pit, Roll])
# fmt: off
WEIGHT_PRESETS: list[tuple[str, list[float]]] = [
    ("all=1",       [1.0, 1.0, 1.0, 1.0]),   # baseline
    ("no_thr",      [0.0, 1.0, 1.0, 1.0]),   # throttle disabled
    ("R>Y>P",       [0.0, 1.0, 0.5, 2.0]),   # user observation: Roll best, Yaw 2nd, Pitch least
    ("R+Y only",    [0.0, 1.0, 0.0, 2.0]),   # drop Pitch entirely
    ("Roll only",   [0.0, 0.0, 0.0, 1.0]),   # single-channel extreme
    ("Y+P+R",       [0.0, 1.0, 1.0, 2.0]),   # no throttle, double Roll
    ("Thr+R",       [1.0, 0.0, 0.0, 1.0]),   # throttle + roll (no angular rates)
]
# fmt: on

# Convergence threshold: fraction of sigma_max that counts as "converged"
CONV_FRAC = 0.20
# Minimum consecutive converged steps to count as "first stable lock"
LOCK_CONSECUTIVE = 3

CSV_OUT = Path(__file__).parent / "sigma_test_results.csv"

# PWM conversion constants (mirror lap_loader internals)
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]  # [thr, yaw, pitch, roll]
_RC_CENTER = 1500.0
_RC_HALF = 500.0


# ── data loading ──────────────────────────────────────────────────────────────

class SessionData(NamedTuple):
    label: str
    t: np.ndarray        # absolute timestamps (s)
    sticks: np.ndarray   # (N, 4) normalized sticks [thr,yaw,pitch,roll]
    rate_profile: dict
    duration_s: float


def _load_session_a(session_path: Path, lap_index: int = 1) -> SessionData:
    """Load a specific lap from a session that has events_edited.parquet."""
    laps, _track = load_rc_only_session(session_path)
    matching = [l for l in laps if l.index == lap_index]
    if not matching:
        available = [l.index for l in laps]
        raise RuntimeError(
            f"Lap {lap_index} not found in {session_path.name}. "
            f"Available: {available}"
        )
    lap = matching[0]
    rp = load_rate_profile(session_path)
    return SessionData(
        label=f"{session_path.name}  [lap {lap.index}, {lap.gate_count} gates]",
        t=lap.t.copy(),
        sticks=lap.sticks.copy().astype(float),
        rate_profile=rp,
        duration_s=float(lap.t[-1] - lap.t[0]),
    )


def _load_session_raw(session_path: Path) -> SessionData:
    """Load full RC data directly from rc_channels.parquet (no lap events needed)."""
    import pandas as pd

    rc = pd.read_parquet(session_path / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    ts = rc["ts_wall"].to_numpy(dtype=float)
    sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        sticks[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF
    rp = load_rate_profile(session_path)
    return SessionData(
        label=f"{session_path.name}  [full session]",
        t=ts,
        sticks=sticks,
        rate_profile=rp,
        duration_s=float(ts[-1] - ts[0]),
    )


# ── particle filter run ───────────────────────────────────────────────────────

class RunMetrics(NamedTuple):
    session_label: str
    obs_sigma: float
    weights_label: str
    weights: list[float]
    n_frames: int
    duration_s: float
    sigma_max_possible: float   # L / (2*pi)
    conv_threshold_m: float     # CONV_FRAC * sigma_max_possible
    sigma_mean: float
    sigma_median: float
    sigma_p10: float            # 10th-percentile (best)
    sigma_p90: float            # 90th-percentile (worst)
    frac_converged: float       # fraction of steps with σ < conv_threshold_m
    t_first_lock_s: float       # time to first stable lock; -1 = never
    prog_std: float             # std of progress (low = stable tracking)


def _run(
    session: SessionData,
    obs_sigma: float,
    weights_label: str,
    weights: list[float],
) -> RunMetrics:
    loc = OnlineLocalizer.from_file(
        REF_PATH,
        obs_sigma=obs_sigma,
        channel_weights=np.asarray(weights, dtype=float),
    )
    loc.reset()

    sigma_max = loc.ref.L / (2.0 * np.pi)
    conv_thr = sigma_max * CONV_FRAC

    sigmas: list[float] = []
    progs: list[float] = []
    t_rel: list[float] = []

    t0 = float(session.t[0])
    prev_ts: float | None = None
    consec = 0
    t_lock = -1.0

    for i, (ts, sticks) in enumerate(zip(session.t, session.sticks)):
        dt = float(ts - prev_ts) if prev_ts is not None else None
        prev_ts = float(ts)

        result = loc.update(sticks.tolist(), dt, rate_profile=session.rate_profile)
        sigma = result.uncertainty_m

        sigmas.append(sigma)
        progs.append(result.progress)
        t_rel.append(float(ts) - t0)

        if sigma < conv_thr:
            consec += 1
            if consec >= LOCK_CONSECUTIVE and t_lock < 0:
                t_lock = t_rel[max(0, i - LOCK_CONSECUTIVE + 1)]
        else:
            consec = 0

    sigmas_arr = np.array(sigmas)
    progs_arr = np.array(progs)

    return RunMetrics(
        session_label=session.label,
        obs_sigma=obs_sigma,
        weights_label=weights_label,
        weights=weights,
        n_frames=len(sigmas),
        duration_s=float(session.t[-1] - session.t[0]),
        sigma_max_possible=sigma_max,
        conv_threshold_m=conv_thr,
        sigma_mean=float(sigmas_arr.mean()),
        sigma_median=float(np.median(sigmas_arr)),
        sigma_p10=float(np.percentile(sigmas_arr, 10)),
        sigma_p90=float(np.percentile(sigmas_arr, 90)),
        frac_converged=float((sigmas_arr < conv_thr).mean()),
        t_first_lock_s=t_lock,
        prog_std=float(progs_arr.std()),
    )


# ── formatting ────────────────────────────────────────────────────────────────

def _bar(frac: float, width: int = 12) -> str:
    filled = round(frac * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_table(results: list[RunMetrics]) -> None:
    sessions = list(dict.fromkeys(r.session_label for r in results))
    sigma_max = results[0].sigma_max_possible

    print()
    L = sigma_max * 2 * np.pi
    print(
        f"Reference: {REF_PATH.name}  "
        f"L={L:.1f}m  sigma_max={sigma_max:.2f}m  "
        f"conv_threshold={sigma_max * CONV_FRAC:.2f}m  ({CONV_FRAC*100:.0f}% of max)"
    )
    print()

    W_COL = 14
    hdr = (
        f"  {'sigma':>5}  {'weights':<{W_COL}}  {'mean':>6}  {'med':>6}  "
        f"{'p10':>6}  {'p90':>6}  {'conv%':>6}  {'lock_s':>6}  {'prog_std':>8}  bar"
    )
    sep = "-" * (len(hdr) + 2)

    for sess in sessions:
        print(sep)
        print(f"Session: {sess}")
        print(hdr)
        print(sep)
        for r in results:
            if r.session_label != sess:
                continue
            lock_str = f"{r.t_first_lock_s:6.2f}" if r.t_first_lock_s >= 0 else "  none"
            w_str = r.weights_label[:W_COL].ljust(W_COL)
            print(
                f"  {r.obs_sigma:5.1f}  {w_str}  "
                f"{r.sigma_mean:6.2f}  {r.sigma_median:6.2f}  "
                f"{r.sigma_p10:6.2f}  {r.sigma_p90:6.2f}  "
                f"{r.frac_converged*100:6.1f}  {lock_str}  "
                f"{r.prog_std:8.4f}  "
                f"{_bar(r.frac_converged)}"
            )
        print()


def _save_csv(results: list[RunMetrics], path: Path) -> None:
    fields = list(RunMetrics._fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r._asdict())
    print(f"CSV saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading sessions...")

    sessions: list[SessionData] = []

    print(f"  A: {SESSION_A.name}")
    try:
        sd = _load_session_a(SESSION_A, lap_index=1)
        sessions.append(sd)
        print(f"     OK: {len(sd.t)} frames, {sd.duration_s:.2f}s")
    except Exception as e:
        print(f"     ERROR: {e}")

    print(f"  B: {SESSION_B.name}")
    try:
        sd = _load_session_raw(SESSION_B)
        sessions.append(sd)
        print(f"     OK: {len(sd.t)} frames, {sd.duration_s:.2f}s")
    except Exception as e:
        print(f"     ERROR: {e}")

    if not sessions:
        print("No sessions loaded. Aborting.")
        return

    if not REF_PATH.exists():
        print(f"ERROR: reference not found: {REF_PATH}")
        return

    print(f"\nReference: {REF_PATH.name}")
    print(
        f"Sweep: {len(WEIGHT_PRESETS)} weight presets "
        f"x {len(OBS_SIGMAS)} sigmas "
        f"x {len(sessions)} sessions"
    )
    print("Running...\n")

    results: list[RunMetrics] = []
    for sess in sessions:
        for w_label, w_vals in WEIGHT_PRESETS:
            for sigma in OBS_SIGMAS:
                r = _run(sess, sigma, weights_label=w_label, weights=w_vals)
                results.append(r)
                lock_str = f"{r.t_first_lock_s:.2f}s" if r.t_first_lock_s >= 0 else "none"
                print(
                    f"  [{w_label:<10}]  sigma={sigma:.1f}  "
                    f"conv={r.frac_converged*100:5.1f}%  "
                    f"med={r.sigma_median:.2f}m  lock={lock_str:>6}  "
                    f"{sess.label[:35]}"
                )

    _print_table(results)
    _save_csv(results, CSV_OUT)


if __name__ == "__main__":
    main()

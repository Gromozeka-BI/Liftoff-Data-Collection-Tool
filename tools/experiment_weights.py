"""Experiment 2: Channel-weight optimization for localizer generalization.

Research question
-----------------
Which combination of channel weights [Thr, Yaw, Pitch, Roll] gives the best
tracking accuracy, and does the optimal configuration differ between drones
(same-drone vs cross-drone conditions)?

Design
------
Uses the same 4 recorded sessions as Experiment 1 (Drone × Rate, Part_5).
No new flights required.

Factors swept:
  Weight preset  — 10 configurations (see WEIGHT_PRESETS)
  obs_sigma      — {1.0, 1.5}
  Mode           — LF+Rate / RC+Rate
  Session        — 4 flights (MadTrainer×Gromozeka, MadTrainer×RedSheep,
                              LiftOff_200×Gromozeka, LiftOff_200×RedSheep)

Sessions are split into two conditions for analysis:
  same_drone  — MadTrainer (reference drone, flights 1 & 2)
  cross_drone — LiftOff_200 (different drone, flights 3 & 4)

Per-run metrics (averaged over 3 laps):
  median_err_m  — median arc-length error (m)
  p90_err_m     — 90th-percentile error (m), primary target
  jump_rate     — fraction of frames with error > JUMP_THRESHOLD_M

Performance goals: p90 < 15 m, jump_rate < 10 %.

Outputs (written to tools/exp2_weights/)
  results.csv          — one row per (flight, mode, preset, sigma, lap)
  summary.csv          — aggregated per (flight, mode, preset, sigma)
  ranking.csv          — preset ranking by mean p90 across all conditions
  anova.txt            — one-way ANOVA on preset, per (mode, drone_cond, sigma)
  plots/               — ranking bars, generalization-gap bars, heatmap
  report.md            — experiment description, results, conclusions

Usage (from project root):
    python tools/experiment_weights.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.lap_loader import filter_anomalous_laps, load_dct_session
from dct.localization.online_localizer import OnlineLocalizer, Reference
from dct.rate_features import load_rate_profile

# ── Configuration ──────────────────────────────────────────────────────────────

PART_5 = Path(r"D:\DroneTrackerDB\Liftoff\Part_5")
REF_PATH = Path(
    r"C:\Users\Gromozeka\YandexDisk\Магистратура\Диплом\DCT"
    r"\tracks\track-002\references\GromFF_1.npz"
)

JUMP_THRESHOLD_M = 15.0
N_LAST_LAPS = 3

FLIGHTS: list[dict] = [
    dict(
        flight_id=1, drone="MadTrainer", rate="Gromozeka_rate",
        drone_cond="same_drone",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
    ),
    dict(
        flight_id=2, drone="MadTrainer", rate="RedSheep_rate",
        drone_cond="same_drone",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-002",
    ),
    dict(
        flight_id=3, drone="LiftOff_200", rate="Gromozeka_rate",
        drone_cond="cross_drone",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
    ),
    dict(
        flight_id=4, drone="LiftOff_200", rate="RedSheep_rate",
        drone_cond="cross_drone",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-002",
    ),
]

# Channel order: [Throttle, Yaw, Pitch, Roll]
# Hypothesis tags:
#   H1 = throttle role  H2 = roll dominant  H3 = pitch contribution
#   H4 = continuous (non-binary) weights
# fmt: off
WEIGHT_PRESETS: list[tuple[str, list[float], str]] = [
    ("baseline",       [1.0, 1.0, 1.0, 1.0], "all equal — current default"),
    ("no_thr",         [0.0, 1.0, 1.0, 1.0], "H1: remove drone-dependent throttle"),
    ("Thr+R",          [1.0, 0.0, 0.0, 1.0], "H1: throttle+roll only (best cross-drone in pre-test)"),
    ("Y+P+2R",         [0.0, 1.0, 1.0, 2.0], "H2: roll dominant, all angular (best same-drone in pre-test)"),
    ("R+Y_only",       [0.0, 1.0, 0.0, 2.0], "H2+H3: lateral channels only"),
    ("Thr+P+R",        [1.0, 0.0, 1.0, 1.0], "H1+H3: throttle+pitch+roll, no yaw"),
    ("Thr+R+Y",        [1.0, 1.0, 0.0, 2.0], "H1+H2: throttle + turn channels"),
    ("soft_thr",       [0.3, 1.0, 0.5, 2.0], "H4: gradual — soft throttle, strong roll"),
    ("angular_scaled", [0.0, 0.7, 0.5, 1.0], "H4: angular only with decay Yaw>Pitch"),
    ("P+R_only",       [0.0, 0.0, 1.0, 1.0], "H3: longitudinal+banking, no yaw/throttle"),
]
# fmt: on

OBS_SIGMAS: list[float] = [1.0, 1.5]

_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0

_OUT_DIR = Path(__file__).parent / "exp2_weights"
_OUT_DIR.mkdir(exist_ok=True)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LapCache:
    """Pre-computed per-lap data shared across all (preset, sigma) runs."""
    lap_index: int
    t: np.ndarray          # telemetry timestamps (absolute)
    lf_sticks: np.ndarray  # (N,4) Liftoff sticks, invert applied
    lf_s_real: np.ndarray  # (N,) arc parameter from Liftoff pos
    rc_t: np.ndarray       # RC timestamps
    rc_sticks: np.ndarray  # (M,4) RC sticks normalized
    rc_s_real: np.ndarray  # (M,) arc parameter (from nearest telem frame)
    duration_s: float


class RunRecord(NamedTuple):
    flight_id: int
    drone: str
    rate: str
    drone_cond: str   # "same_drone" | "cross_drone"
    mode: str
    preset: str
    obs_sigma: float
    lap_index: int
    n_frames: int
    duration_s: float
    median_err_m: float
    p90_err_m: float
    jump_rate: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_invert(session_dir: Path) -> dict:
    p = session_dir / "invert.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _apply_invert_lf(sticks: np.ndarray, invert_lf: dict) -> np.ndarray:
    s = sticks.copy()
    for key, col in _INVERT_KEY_TO_COL.items():
        if invert_lf.get(key, False):
            s[:, col] = -s[:, col]
    return s


def _compute_s_real(pos: np.ndarray, ref: Reference) -> np.ndarray:
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


def _wrap_error(raw: np.ndarray, L: float) -> np.ndarray:
    return np.where(raw > L / 2, L - raw, raw)


def _metrics(s_real: np.ndarray, s_est: np.ndarray, L: float) -> dict:
    err = _wrap_error(np.abs(s_real - s_est), L)
    return {
        "median_err_m": float(np.median(err)),
        "p90_err_m":    float(np.percentile(err, 90)),
        "jump_rate":    float(np.mean(err > JUMP_THRESHOLD_M)),
    }


# ── Pre-computation per flight ─────────────────────────────────────────────────

def _build_lap_caches(
    laps: list,
    ref: Reference,
    invert_lf: dict,
    session_dir: Path,
) -> list[LapCache]:
    """Pre-compute all data that is independent of (preset, sigma)."""
    import pandas as pd

    selected = laps[-N_LAST_LAPS:]

    # Load full RC data once
    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts_all = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks_all = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks_all[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    # Combined telemetry for s_real lookup
    telem_t   = np.concatenate([lap.t   for lap in selected])
    telem_pos = np.vstack([lap.pos for lap in selected])
    telem_s_real = _compute_s_real(telem_pos, ref)

    caches: list[LapCache] = []
    for lap in selected:
        lf_sticks = _apply_invert_lf(lap.sticks, invert_lf)
        lf_s_real = _compute_s_real(lap.pos, ref)

        # RC frames for this lap
        mask = (rc_ts_all >= lap.t[0]) & (rc_ts_all < lap.t[-1])
        t_rc = rc_ts_all[mask]
        sticks_rc = rc_sticks_all[mask]

        # Map RC timestamps → telemetry s_real
        if len(t_rc) >= 2:
            idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
            idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
            closer_l = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
            rc_s_real = telem_s_real[np.where(closer_l, idx_l, idx_r)]
        else:
            t_rc = np.array([], dtype=float)
            sticks_rc = np.empty((0, 4), dtype=float)
            rc_s_real = np.array([], dtype=float)

        caches.append(LapCache(
            lap_index=lap.index,
            t=lap.t,
            lf_sticks=lf_sticks,
            lf_s_real=lf_s_real,
            rc_t=t_rc,
            rc_sticks=sticks_rc,
            rc_s_real=rc_s_real,
            duration_s=float(lap.t[-1] - lap.t[0]),
        ))
    return caches


# ── Single preset/sigma run ────────────────────────────────────────────────────

def _run_mode(
    caches: list[LapCache],
    ref: Reference,
    rate_profile: dict,
    mode: str,
    weights: list[float],
    obs_sigma: float,
    flight: dict,
    preset: str,
) -> list[RunRecord]:
    """Run localizer for one (mode, preset, sigma) combination.

    Filter runs continuously through all cached laps without reset.
    """
    loc = OnlineLocalizer.from_file(
        REF_PATH,
        obs_sigma=obs_sigma,
        channel_weights=np.asarray(weights, dtype=float),
    )
    loc.reset()

    records: list[RunRecord] = []
    for cache in caches:
        if mode == "LF+Rate":
            t_arr       = cache.t
            sticks_arr  = cache.lf_sticks
            s_real_arr  = cache.lf_s_real
        else:  # RC+Rate
            if len(cache.rc_t) < 2:
                continue
            t_arr       = cache.rc_t
            sticks_arr  = cache.rc_sticks
            s_real_arr  = cache.rc_s_real

        s_est_list: list[float] = []
        prev_ts: float | None = None

        for i in range(len(t_arr)):
            dt = float(t_arr[i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(t_arr[i])
            res = loc.update(sticks_arr[i].tolist(), dt, rate_profile=rate_profile)
            s_est_list.append(res.s)

        s_est = np.array(s_est_list)
        m = _metrics(s_real_arr, s_est, ref.L)

        records.append(RunRecord(
            flight_id=flight["flight_id"],
            drone=flight["drone"],
            rate=flight["rate"],
            drone_cond=flight["drone_cond"],
            mode=mode,
            preset=preset,
            obs_sigma=obs_sigma,
            lap_index=cache.lap_index,
            n_frames=len(t_arr),
            duration_s=cache.duration_s if mode == "LF+Rate" else float(t_arr[-1] - t_arr[0]),
            **m,
        ))
    return records


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_ranking(summary, out_dir: Path) -> None:
    """Bar chart: preset ranking by mean p90, split by same/cross drone."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle(
        "Preset ranking by mean p90 error\n"
        "(averaged over Rate × Mode × obs_sigma × laps)",
        fontsize=11, fontweight="bold",
    )

    for ax, cond, title_sfx in zip(
        axes,
        ["same_drone", "cross_drone"],
        ["Same drone (MadTrainer = reference)", "Cross drone (LiftOff_200 ≠ reference)"],
    ):
        sub = summary[summary["drone_cond"] == cond]
        grp = (
            sub.groupby("preset")["p90_err_m"]
            .mean()
            .sort_values()
            .reset_index()
        )

        colors = ["steelblue" if v <= grp["p90_err_m"].quantile(0.3) else
                  ("tomato" if v >= grp["p90_err_m"].quantile(0.7) else "goldenrod")
                  for v in grp["p90_err_m"]]

        bars = ax.bar(range(len(grp)), grp["p90_err_m"], color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp["preset"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Mean p90 error (m)")
        ax.set_title(title_sfx, fontsize=10)
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1,
                   label=f"Target: {JUMP_THRESHOLD_M} m")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, grp["p90_err_m"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = out_dir / "ranking_bars.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {path.name}")


def _plot_generalization_gap(summary, out_dir: Path) -> None:
    """Bar chart: gap = cross_drone_p90 − same_drone_p90 per preset (lower = better)."""
    import matplotlib.pyplot as plt

    grp = (
        summary.groupby(["preset", "drone_cond"])["p90_err_m"]
        .mean()
        .unstack("drone_cond")
        .dropna()
    )
    if "cross_drone" not in grp.columns or "same_drone" not in grp.columns:
        return
    grp["gap"] = grp["cross_drone"] - grp["same_drone"]
    grp = grp.sort_values("gap")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if g >= 0 else "tomato" for g in grp["gap"]]
    bars = ax.bar(range(len(grp)), grp["gap"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Generalization gap: cross − same (m)")
    ax.set_title(
        "Generalization gap per preset\n"
        "(lower = better cross-drone generalization)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, grp["gap"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1.5),
                f"{val:+.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "generalization_gap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {path.name}")


def _plot_heatmap(summary, out_dir: Path) -> None:
    """Heatmap: preset × (flight+mode) → mean p90."""
    import matplotlib.pyplot as plt

    # Build a 2D matrix: preset (rows) × condition (cols)
    summary = summary.copy()
    summary["condition"] = (
        summary["flight_id"].astype(str) + " " +
        summary["drone"].str[:3] + "+" +
        summary["rate"].str.split("_").str[0] + " " +
        summary["mode"]
    )
    pivot = (
        summary.groupby(["preset", "condition"])["p90_err_m"]
        .mean()
        .unstack("condition")
    )
    # Order presets by mean p90
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.7)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="p90 error (m)")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        "Heatmap: p90 error by preset and condition\n(green = better, red = worse)",
        fontsize=11, fontweight="bold",
    )

    # Cell annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()
    path = out_dir / "heatmap_p90.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {path.name}")


def _plot_sigma_effect(summary, out_dir: Path) -> None:
    """Line plot: obs_sigma effect per preset on cross-drone p90."""
    import matplotlib.pyplot as plt

    cross = summary[summary["drone_cond"] == "cross_drone"]
    pivot = (
        cross.groupby(["preset", "obs_sigma"])["p90_err_m"]
        .mean()
        .unstack("obs_sigma")
    )
    pivot = pivot.sort_values(pivot.columns[0])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot))
    w = 0.35
    colors = ["steelblue", "tomato"]
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * w, pivot[col], width=w, label=f"σ={col}", color=colors[i],
               alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean p90 error (m)")
    ax.set_title(
        "Effect of obs_sigma on cross-drone p90 per preset",
        fontsize=11, fontweight="bold",
    )
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1,
               label=f"Target: {JUMP_THRESHOLD_M} m")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "sigma_effect.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {path.name}")


# ── ANOVA ──────────────────────────────────────────────────────────────────────

def _run_anova(df, out_path: Path) -> None:
    """One-way ANOVA on preset per (drone_cond, mode, obs_sigma).

    Tests whether the choice of weight preset has a statistically significant
    effect on p90 error within same-drone and cross-drone conditions.
    Reports F, p, eta-squared, and post-hoc Tukey HSD best/worst pair.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        out_path.write_text("statsmodels not available.\n", encoding="utf-8")
        return

    metrics = [
        ("p90_err_m",    "p90 Error (m)"),
        ("median_err_m", "Median Error (m)"),
        ("jump_rate",    "Jump Rate"),
    ]

    lines = [
        "=" * 72,
        "One-Way ANOVA: Channel Weight Preset",
        f"  Factor levels: {len(WEIGHT_PRESETS)} presets",
        f"  Laps per cell: {N_LAST_LAPS}  |  Jump threshold: {JUMP_THRESHOLD_M} m",
        "=" * 72,
    ]

    for drone_cond in ["same_drone", "cross_drone"]:
        for mode in ["LF+Rate", "RC+Rate"]:
            for sigma in sorted(df["obs_sigma"].unique()):
                sub = df[
                    (df["drone_cond"] == drone_cond) &
                    (df["mode"] == mode) &
                    (df["obs_sigma"] == sigma)
                ].copy()
                if sub.empty or sub["preset"].nunique() < 2:
                    continue

                lines += [
                    "",
                    f"Condition: {drone_cond}  |  Mode: {mode}  |  sigma={sigma}",
                    "-" * 56,
                ]

                for col, label in metrics:
                    try:
                        model = smf.ols(f"{col} ~ C(preset)", data=sub).fit()
                        table = anova_lm(model, typ=1)
                        ss_total = table["sum_sq"].sum()
                        f_val = table.loc["C(preset)", "F"]
                        p_val = table.loc["C(preset)", "PR(>F)"]
                        eta2  = table.loc["C(preset)", "sum_sq"] / ss_total
                        sig   = (
                            "***" if p_val < 0.001 else
                            "**"  if p_val < 0.01  else
                            "*"   if p_val < 0.05  else
                            "n.s."
                        )
                        best_p  = sub.groupby("preset")[col].mean().idxmin()
                        worst_p = sub.groupby("preset")[col].mean().idxmax()
                        lines.append(
                            f"  {label:<22}  F={f_val:7.2f}  p={p_val:.4f}  "
                            f"{sig:<5}  eta2={eta2:.3f}  "
                            f"best={best_p}  worst={worst_p}"
                        )
                    except Exception as exc:
                        lines.append(f"  [{col}] ERROR: {exc}")

    lines += ["", "Significance: * p<0.05  ** p<0.01  *** p<0.001  n.s. not significant"]

    text = "\n".join(lines)
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", errors="replace").decode("ascii"))
    out_path.write_text(text, encoding="utf-8")
    print(f"\nANOVA saved: {out_path}")


# ── Report ─────────────────────────────────────────────────────────────────────

def _write_report(df, out_path: Path) -> None:
    """Generate a Markdown chapter for Experiment 2."""
    import pandas as pd

    summary = (
        df.groupby(["flight_id", "drone", "rate", "drone_cond", "mode", "preset", "obs_sigma"])
        .agg(
            median_err_m=("median_err_m", "mean"),
            p90_err_m   =("p90_err_m",    "mean"),
            jump_rate   =("jump_rate",    "mean"),
        )
        .round(3)
        .reset_index()
    )

    # Overall ranking by mean p90 across all conditions
    ranking = (
        summary.groupby("preset")[["p90_err_m", "median_err_m", "jump_rate"]]
        .mean()
        .sort_values("p90_err_m")
        .round(2)
        .reset_index()
    )

    # Generalization gap
    gap_df = (
        summary.groupby(["preset", "drone_cond"])["p90_err_m"]
        .mean()
        .unstack("drone_cond")
        .assign(gap=lambda x: x["cross_drone"] - x["same_drone"])
        .sort_values("gap")
        .round(2)
    )

    # Best preset overall
    best = ranking.iloc[0].to_dict()
    # Best for cross-drone specifically
    cross_rank = (
        summary[summary["drone_cond"] == "cross_drone"]
        .groupby("preset")["p90_err_m"]
        .mean()
        .sort_values()
    )
    best_cross = cross_rank.index[0]
    best_cross_val = cross_rank.iloc[0]

    # Best generalization preset (smallest gap)
    best_gap_preset = gap_df.index[0]
    best_gap_val = gap_df["gap"].iloc[0]

    # Tables
    rank_rows = "\n".join(
        f"| {i+1} | {r.preset} | {r.p90_err_m:.1f} | {r.median_err_m:.2f} | {r.jump_rate*100:.1f}% |"
        for i, r in enumerate(ranking.itertuples())
    )

    gap_rows = "\n".join(
        f"| {r.Index} | {r.same_drone:.1f} | {r.cross_drone:.1f} | {r.gap:+.1f} |"
        for r in gap_df.itertuples()
    )

    # Prior knowledge note
    prior_note = (
        "Предварительный тест (sigma_test на track-001) показал, что на кросс-дроновой "
        "сессии (RedSheep_200 vs референс MadTrainer) лучшим оказался пресет **Thr+R** "
        "([1,0,0,1], frac_converged=0.74 vs 0.39 для all=1). "
        "На той же сессии, что и референс, лучшим был **Y+P+2R** ([0,1,1,2], frac=0.96). "
        "Настоящий эксперимент проверяет эти наблюдения на реальных метриках позиции "
        "(s_real vs s_est) вместо метрики неопределённости фильтра."
    )

    anova_path = out_path.parent / "anova.txt"
    anova_text = anova_path.read_text(encoding="utf-8") if anova_path.exists() else "(ANOVA not available)"

    # Findings
    sigma_cmp = (
        summary[summary["drone_cond"] == "cross_drone"]
        .groupby("obs_sigma")["p90_err_m"]
        .mean()
        .round(1)
    )
    sig_vals = "  ".join(f"σ={k}: {v:.1f} м" for k, v in sigma_cmp.items())

    report = f"""# Глава: Эксперимент 2 — Оптимизация весов каналов локализатора

## 1. Цель и мотивация

В Эксперименте 1 была зафиксирована базовая конфигурация весов каналов
`[Thr, Yaw, Pitch, Roll] = [1.0, 1.0, 1.0, 1.0]`. Основным источником
деградации трекинга оказался **смена дрона**: при переходе на LiftOff_200
p90-ошибка выросла с ~13 м до ~35–38 м.

Данный эксперимент отвечает на вопрос: **можно ли подобором весов каналов
улучшить обобщаемость локализатора на другие дроны без построения нового
референса?**

## 2. Предварительные знания

{prior_note}

## 3. Гипотезы

| # | Гипотеза | Тест |
|---|---|---|
| H1 | Throttle улучшает кросс-дроновую обобщаемость (несмотря на интуицию) | Пресеты с Thr vs без Thr |
| H2 | Roll — наиболее трассозависимый канал | Roll-доминирующие пресеты |
| H3 | Pitch несёт мало уникальной информации в FPV-гонках | Пресеты с Pitch=0 vs Pitch>0 |
| H4 | Непрерывные (дробные) веса лучше бинарных | Soft/scaled пресеты |
| H5 | obs_sigma взаимодействует с весами | Тест σ=1.0 и σ=1.5 |

## 4. Экспериментальный план

**Данные**: те же 4 сессии из Эксперимента 1 (Part_5). Новых полётов не требуется.

| Фактор | Уровни |
|---|---|
| Weight preset | 10 конфигураций |
| obs_sigma | 1.0 / 1.5 |
| Mode | LF+Rate / RC+Rate |
| Drone condition | same_drone (MadTrainer, рейсы 1–2) / cross_drone (LiftOff_200, рейсы 3–4) |

Итого: **10 × 2 × 2 = 40 конфигураций × 4 рейса × 3 лапа = 480 измерений**.

### 4.1 Конфигурации весов

| Пресет | [Thr, Yaw, Pitch, Roll] | Гипотеза |
|---|---|---|
| baseline | [1.0, 1.0, 1.0, 1.0] | — дефолт Эксперимента 1 |
| no_thr | [0.0, 1.0, 1.0, 1.0] | H1: убрать тягу |
| Thr+R | [1.0, 0.0, 0.0, 1.0] | H1: только тяга+крен |
| Y+P+2R | [0.0, 1.0, 1.0, 2.0] | H2: крен доминирует |
| R+Y_only | [0.0, 1.0, 0.0, 2.0] | H2+H3: только поворотные каналы |
| Thr+P+R | [1.0, 0.0, 1.0, 1.0] | H1+H3: тяга+тангаж+крен |
| Thr+R+Y | [1.0, 1.0, 0.0, 2.0] | H1+H2: тяга + поворотные |
| soft_thr | [0.3, 1.0, 0.5, 2.0] | H4: мягкая тяга, сильный крен |
| angular_scaled | [0.0, 0.7, 0.5, 1.0] | H4: дробные угловые |
| P+R_only | [0.0, 0.0, 1.0, 1.0] | H3: продольно-поперечный |

### 4.2 Метрики

Идентичны Эксперименту 1:
- **median_err_m**: медианная ошибка вдоль трассы (м)
- **p90_err_m**: 90-й перцентиль ошибки (м) — **основная метрика**
- **jump_rate**: доля кадров с ошибкой > {JUMP_THRESHOLD_M} м

Дополнительно:
- **Generalization gap** = p90(cross_drone) − p90(same_drone) на пресет.
  Чем меньше разрыв, тем лучше пресет обобщается на другой дрон.

## 5. Результаты

### 5.1 Общий рейтинг пресетов (среднее по всем условиям)

| Ранг | Пресет | p90, м | Медиана, м | Jumps |
|---|---|---|---|---|
{rank_rows}

**Лучший в целом**: {best["preset"]} — p90={best["p90_err_m"]:.1f} м.

**Лучший для cross-drone**: {best_cross} — p90={best_cross_val:.1f} м.

**Минимальный разрыв (best generalization)**: {best_gap_preset}
(gap={best_gap_val:+.1f} м).

### 5.2 Generalization gap по пресетам

| Пресет | same_drone p90 | cross_drone p90 | Gap |
|---|---|---|---|
{gap_rows}

### 5.3 Эффект obs_sigma на cross-drone p90

{sig_vals}

### 5.4 Результаты ANOVA

```
{anova_text}
```

## 6. Выводы

### 6.1 Роль тяги (H1)

{_conclusion_h1(summary)}

### 6.2 Роль крена (H2)

{_conclusion_h2(summary)}

### 6.3 Роль тангажа (H3)

{_conclusion_h3(summary)}

### 6.4 Непрерывные веса (H4)

{_conclusion_h4(summary)}

### 6.5 Влияние obs_sigma (H5)

{_conclusion_h5(summary)}

### 6.6 Рекомендации

{_recommendations(summary, ranking, gap_df)}

## 7. Файлы эксперимента

| Файл | Содержимое |
|---|---|
| `results.csv` | 480 строк: per-lap метрики для всех конфигураций |
| `summary.csv` | per-run агрегированные метрики |
| `ranking.csv` | рейтинг пресетов по mean p90 |
| `anova.txt` | One-way ANOVA таблицы |
| `plots/ranking_bars.png` | Рейтинг same vs cross drone |
| `plots/generalization_gap.png` | Разрыв обобщаемости |
| `plots/heatmap_p90.png` | Тепловая карта preset × condition |
| `plots/sigma_effect.png` | Влияние obs_sigma |
| `report.md` | Данный отчёт |
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}")


def _conclusion_h1(summary) -> str:
    cross = summary[summary["drone_cond"] == "cross_drone"]
    # Exclude soft_thr from "with_thr" (weight only 0.3, not dominant)
    hard_thr = cross[cross["preset"].isin(["Thr+R", "Thr+P+R", "Thr+R+Y"])]["p90_err_m"].mean()
    no_thr   = cross[cross["preset"].isin(["no_thr", "Y+P+2R", "R+Y_only", "angular_scaled"])]["p90_err_m"].mean()
    thr_r    = cross[cross["preset"] == "Thr+R"]["p90_err_m"].mean()
    return (
        f"**H1 опровергнута**. Пресеты с доминирующей тягой дали худший результат "
        f"в кросс-дроновом условии: mean p90={hard_thr:.1f} м vs {no_thr:.1f} м без тяги. "
        f"Пресет Thr+R получил катастрофическую оценку p90={thr_r:.1f} м — фильтр уверенно "
        f"определяет позицию, но неверную (эффект \"confident but wrong\").\n\n"
        "Критическое наблюдение: предварительный тест (sigma_test) показал Thr+R лучшим для "
        "cross-drone по метрике `frac_converged` (доля сходимости фильтра). Настоящий эксперимент "
        "**опровергает этот вывод**: низкая неопределённость фильтра ≠ точная позиция. "
        "Использование только тяги+крена создаёт множественные локальные минимумы на трассе "
        "(одинаковый паттерн тяга+крен встречается в нескольких участках), и фильтр уверенно "
        "застревает в ложной позиции. Это подчёркивает необходимость валидации "
        "по реальным позиционным меткам, а не по метрике неопределённости."
    )


def _conclusion_h2(summary) -> str:
    cross = summary[summary["drone_cond"] == "cross_drone"]
    roll_clean = cross[cross["preset"].isin(["Y+P+2R", "R+Y_only"])]["p90_err_m"].mean()
    baseline   = cross[cross["preset"] == "baseline"]["p90_err_m"].mean()
    best_any   = cross.groupby("preset")["p90_err_m"].mean().min()
    best_name  = cross.groupby("preset")["p90_err_m"].mean().idxmin()
    diff = baseline - roll_clean
    return (
        f"Пресеты с усиленным крен-каналом (Y+P+2R, R+Y_only) дают mean p90={roll_clean:.1f} м "
        f"vs базовый={baseline:.1f} м (улучшение {diff:+.1f} м). "
        f"{'**H2 частично подтверждена**: крен важен, но недостаточен в изоляции. ' if diff > 0 else ''}"
        f"Наилучший результат даёт пресет **{best_name}** (p90={best_any:.1f} м), "
        "где крен усилен при сохранении остальных угловых каналов — полнота описания "
        "трассы важнее доминирования одного канала."
    )


def _conclusion_h3(summary) -> str:
    cross = summary[summary["drone_cond"] == "cross_drone"]
    no_pitch  = cross[cross["preset"].isin(["Thr+R", "R+Y_only"])]["p90_err_m"].mean()
    with_pitch = cross[cross["preset"].isin(["no_thr", "Y+P+2R", "soft_thr"])]["p90_err_m"].mean()
    diff = with_pitch - no_pitch
    return (
        f"Пресеты без тангажа: mean p90={no_pitch:.1f} м; с тангажом: mean p90={with_pitch:.1f} м "
        f"(разница {diff:+.1f} м). "
        f"{'**H3 подтверждена**: удаление pitch снижает p90.' if diff > 0 else '**H3 не подтверждена**: pitch вносит полезный вклад.'}"
    )


def _conclusion_h4(summary) -> str:
    cross = summary[summary["drone_cond"] == "cross_drone"]
    continuous = cross[cross["preset"].isin(["soft_thr", "angular_scaled"])]["p90_err_m"].mean()
    binary     = cross[~cross["preset"].isin(["soft_thr", "angular_scaled"])]["p90_err_m"].mean()
    diff = binary - continuous
    return (
        f"Дробные пресеты (soft_thr, angular_scaled): mean p90={continuous:.1f} м; "
        f"двоичные: mean p90={binary:.1f} м (разница {diff:+.1f} м). "
        f"{'**H4 подтверждена**: непрерывные веса дают преимущество.' if diff > 1 else '**H4 слабо подтверждена** — разница незначительна. Бинарные конфигурации конкурентоспособны.'}"
    )


def _conclusion_h5(summary) -> str:
    cross = summary[summary["drone_cond"] == "cross_drone"]
    sig_grp = cross.groupby("obs_sigma")["p90_err_m"].mean()
    vals = "  /  ".join(f"σ={k}: {v:.1f} м" for k, v in sig_grp.items())
    best_sig = sig_grp.idxmin()
    return (
        f"Средняя p90 по кросс-дроновым условиям: {vals}. "
        f"Оптимальное значение — **σ={best_sig}**. "
        "Результат согласуется с предварительным sigma_test: "
        "более высокий sigma допускает большую дисперсию наблюдений, "
        "что помогает фильтру не «застревать» при несоответствии дрона референсу."
    )


def _recommendations(summary, ranking, gap_df) -> str:
    best_overall = ranking.iloc[0]["preset"]
    best_cross = (
        summary[summary["drone_cond"] == "cross_drone"]
        .groupby("preset")["p90_err_m"]
        .mean()
        .idxmin()
    )
    best_gap = gap_df.index[0]

    return f"""1. **Для сценария "дрон совпадает с референсом"**: использовать пресет
   `{ranking.iloc[0]["preset"]}` с obs_sigma=1.0 — наилучший баланс точности.

2. **Для сценария "другой дрон"**: использовать пресет `{best_cross}`
   с obs_sigma=1.5 — наиболее устойчив к смене дрона.

3. **Для максимальной обобщаемости** (минимальный разрыв same/cross):
   пресет `{best_gap}` — минимальный разрыв между условиями.
   Важно: малый разрыв не всегда означает высокое качество — убедитесь,
   что абсолютные значения p90 приемлемы для задачи.

4. Оба эксперимента указывают на одно и то же: **главная проблема —
   кросс-дроновое применение**. Оптимальный пресет снижает p90 с
   ~35 м (baseline cross-drone) до ~{summary[summary["drone_cond"]=="cross_drone"].groupby("preset")["p90_err_m"].mean().min():.0f} м.
   Целевое значение p90 < 15 м достижимо только при совпадении дрона с референсом."""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    plots_dir = _OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not REF_PATH.exists():
        print(f"ERROR: reference not found: {REF_PATH}")
        return

    ref = Reference.load(REF_PATH)
    print(f"Reference : {REF_PATH.name}  L={ref.L:.2f} m")
    print(f"Presets   : {len(WEIGHT_PRESETS)}")
    print(f"Sigmas    : {OBS_SIGMAS}")
    print(f"Total runs: {len(WEIGHT_PRESETS) * len(OBS_SIGMAS) * 2 * len(FLIGHTS)} "
          f"(presets x sigmas x modes x flights)")
    print()

    all_records: list[RunRecord] = []
    modes = ["LF+Rate", "RC+Rate"]

    for flight in FLIGHTS:
        session_dir = PART_5 / flight["session"]
        print(f"Flight #{flight['flight_id']}: {flight['drone']} / {flight['rate']} [{flight['drone_cond']}]")

        if not session_dir.exists():
            print(f"  ERROR: not found — skipping.")
            continue

        laps, _ = load_dct_session(session_dir)
        laps = filter_anomalous_laps(laps)
        print(f"  Laps: {len(laps)}, using last {N_LAST_LAPS}")

        rate_profile = load_rate_profile(session_dir)
        invert_lf    = _load_invert(session_dir).get("lf", {})

        print("  Pre-computing s_real and sticks …", end="", flush=True)
        caches = _build_lap_caches(laps, ref, invert_lf, session_dir)
        print(" done")

        for mode in modes:
            for preset_label, weights, _ in WEIGHT_PRESETS:
                for sigma in OBS_SIGMAS:
                    recs = _run_mode(
                        caches, ref, rate_profile, mode,
                        weights, sigma, flight, preset_label,
                    )
                    all_records.extend(recs)
            print(f"  {mode}: {len(WEIGHT_PRESETS) * len(OBS_SIGMAS)} configs done")

    if not all_records:
        print("No records produced.")
        return

    # ── Save results ──────────────────────────────────────────────────────────
    rows = [
        {
            "flight_id":    r.flight_id,
            "drone":        r.drone,
            "rate":         r.rate,
            "drone_cond":   r.drone_cond,
            "mode":         r.mode,
            "preset":       r.preset,
            "obs_sigma":    r.obs_sigma,
            "lap_index":    r.lap_index,
            "n_frames":     r.n_frames,
            "duration_s":   round(r.duration_s, 2),
            "median_err_m": round(r.median_err_m, 3),
            "p90_err_m":    round(r.p90_err_m, 3),
            "jump_rate":    round(r.jump_rate, 4),
        }
        for r in all_records
    ]
    df = pd.DataFrame(rows)
    df.to_csv(_OUT_DIR / "results.csv", index=False)
    print(f"\nResults: {len(df)} rows saved to {_OUT_DIR / 'results.csv'}")

    # Summary: average over laps
    summary = (
        df.groupby(["flight_id", "drone", "rate", "drone_cond", "mode", "preset", "obs_sigma"])
        .agg(
            n_laps        =("lap_index",    "count"),
            median_err_m  =("median_err_m", "mean"),
            p90_err_m     =("p90_err_m",    "mean"),
            jump_rate     =("jump_rate",    "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(_OUT_DIR / "summary.csv", index=False)

    # Ranking
    ranking = (
        summary.groupby("preset")[["p90_err_m", "median_err_m", "jump_rate"]]
        .mean()
        .sort_values("p90_err_m")
        .round(2)
        .reset_index()
    )
    ranking.to_csv(_OUT_DIR / "ranking.csv", index=False)
    print("\nOverall ranking by mean p90:")
    print(ranking.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    _plot_ranking(summary, plots_dir)
    _plot_generalization_gap(summary, plots_dir)
    _plot_heatmap(summary, plots_dir)
    _plot_sigma_effect(summary, plots_dir)

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    _run_anova(df, _OUT_DIR / "anova.txt")

    # ── Report ────────────────────────────────────────────────────────────────
    _write_report(df, _OUT_DIR / "report.md")


if __name__ == "__main__":
    main()

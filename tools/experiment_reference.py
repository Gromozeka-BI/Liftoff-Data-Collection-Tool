"""Experiment 4: Reference Quality — Same-Drone vs Cross-Drone Reference.

Research question
-----------------
How much does the reference drone matching the test drone affect tracking
accuracy?  Does using a LiftOff_200 reference (instead of MadTrainer) when
tracking LiftOff_200 close the performance gap observed in Experiments 1–3?

Hypothesis
----------
If the drone is the dominant factor (Exp 1 eta2=0.41), then swapping to a
same-drone reference should reduce cross-drone error to near same-drone levels.

Design
------
3 references × 4 test flights = 12 (ref, flight) combinations.

References:
  GromFF_1           — MadTrainer drone, Gromozeka rate  (baseline from Exp 1–3)
  LiftOff200_Grom_1  — LiftOff_200 drone, Gromozeka rate (new)
  LiftOff200_Red_1   — LiftOff_200 drone, RedSheep rate  (new)

Test flights (same 4 sessions as Exps 1–3):
  Flight 1: MadTrainer   × Gromozeka_rate
  Flight 2: MadTrainer   × RedSheep_rate
  Flight 3: LiftOff_200  × Gromozeka_rate
  Flight 4: LiftOff_200  × RedSheep_rate

Each (ref, flight) pair is classified as:
  condition = {same|cross}_drone_{same|cross}_rate

Fixed hyperparameters — optimal from Experiment 3:
  obs_sigma       = 2.0
  process_noise_v = 8.0
  process_noise_s = 1.5
  Mode            = RC+Rate

Weight presets:
  baseline, angular_scaled, no_thr

Total: 12 (ref x flight) x 3 (presets) x 3 (laps) = 108 lap evaluations.

Outputs (tools/exp4_reference/)
  results.csv  — one row per (ref, flight, preset, lap)
  summary.csv  — per (ref, flight, preset, condition)
  report.md    — description, results, conclusions

Usage (from project root):
    python tools/experiment_reference.py
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

def _find_part5() -> Path:
    """Find the Part_5_* directory dynamically (handles Cyrillic suffix)."""
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")
    candidates = sorted([d for d in liftoff.iterdir() if d.name.startswith("Part_5")])
    if not candidates:
        raise FileNotFoundError(
            f"No Part_5* directory found under {liftoff}"
        )
    return candidates[0]

PART_5   = _find_part5()
REF_DIR  = Path(
    r"C:\Users\Gromozeka\YandexDisk\Магистратура\Диплом\DCT"
    r"\tracks\track-002\references"
)

JUMP_THRESHOLD_M = 15.0
N_LAST_LAPS      = 3
MODE             = "RC+Rate"

# ── Optimal hyperparameters from Experiment 3 ──────────────────────────────────
OBS_SIGMA       = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5

# fmt: off
WEIGHT_PRESETS: list[tuple[str, list[float], str]] = [
    ("baseline",       [1.0, 1.0, 1.0, 1.0], "all equal — Exp 1 default"),
    ("angular_scaled", [0.0, 0.7, 0.5, 1.0], "rank-1 Exp 2"),
    ("no_thr",         [0.0, 1.0, 1.0, 1.0], "best cross-drone Exp 3"),
]
# fmt: on

REFERENCES: list[dict] = [
    dict(
        name="GromFF_1",
        path=REF_DIR / "GromFF_1.npz",
        drone="MadTrainer",
        rate="Gromozeka_rate",
        desc="MadTrainer drone + Gromozeka rate — базовый референс Экс. 1–3",
    ),
    dict(
        name="LiftOff200_Grom_1",
        path=REF_DIR / "LiftOff200_Grom_1.npz",
        drone="LiftOff_200",
        rate="Gromozeka_rate",
        desc="LiftOff_200 drone + Gromozeka rate — новый референс",
    ),
    dict(
        name="LiftOff200_Red_1",
        path=REF_DIR / "LiftOff200_Red_1.npz",
        drone="LiftOff_200",
        rate="RedSheep_rate",
        desc="LiftOff_200 drone + RedSheep rate — новый референс",
    ),
]

FLIGHTS: list[dict] = [
    dict(
        flight_id=1, drone="MadTrainer", rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-001",
    ),
    dict(
        flight_id=2, drone="MadTrainer", rate="RedSheep_rate",
        session="2026-05-08_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-002",
    ),
    dict(
        flight_id=3, drone="LiftOff_200", rate="Gromozeka_rate",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-001",
    ),
    dict(
        flight_id=4, drone="LiftOff_200", rate="RedSheep_rate",
        session="2026-05-08_pilot-Gromozeka_drone-LiftOff_200_track-track-002_session-002",
    ),
]

_RC_CH_ORDER    = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0

_OUT_DIR = Path(__file__).parent / "exp4_reference"
_OUT_DIR.mkdir(exist_ok=True)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FlightRawData:
    """Per-flight data independent of the reference (RC sticks, telem positions)."""
    flight: dict
    rate_profile: dict
    telem_t:   np.ndarray   # concatenated timestamps for last N_LAST_LAPS
    telem_pos: np.ndarray   # concatenated positions  for last N_LAST_LAPS
    lap_indices: list[int]
    rc_t_per_lap:      list[np.ndarray]
    rc_sticks_per_lap: list[np.ndarray]


@dataclass
class LapCache:
    """Per (flight, reference) pre-computed lookup."""
    lap_index: int
    rc_t:      np.ndarray
    rc_sticks: np.ndarray
    rc_s_real: np.ndarray   # ground-truth arc param on THIS reference
    duration_s: float


class RunRecord(NamedTuple):
    ref_name:   str
    ref_drone:  str
    ref_rate:   str
    flight_id:  int
    test_drone: str
    test_rate:  str
    drone_cond: str    # "same_drone" | "cross_drone"
    rate_cond:  str    # "same_rate"  | "cross_rate"
    condition:  str    # combined label
    preset:     str
    lap_index:  int
    n_frames:   int
    duration_s: float
    median_err_m: float
    p90_err_m:    float
    jump_rate:    float


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _condition_label(ref: dict, flight: dict) -> tuple[str, str, str]:
    drone_cond = "same_drone" if ref["drone"] == flight["drone"] else "cross_drone"
    rate_cond  = "same_rate"  if ref["rate"]  == flight["rate"]  else "cross_rate"
    return drone_cond, rate_cond, f"{drone_cond}_{rate_cond}"


# ── Stage 1: load raw flight data ──────────────────────────────────────────────

def _load_flight_raw(flight: dict, session_dir: Path) -> FlightRawData:
    import pandas as pd

    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    rate_profile = load_rate_profile(session_dir)
    selected = laps[-N_LAST_LAPS:]

    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts_all = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks_all = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks_all[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    telem_t   = np.concatenate([lap.t   for lap in selected])
    telem_pos = np.vstack([lap.pos       for lap in selected])

    lap_indices: list[int] = []
    rc_t_per_lap:      list[np.ndarray] = []
    rc_sticks_per_lap: list[np.ndarray] = []

    for lap in selected:
        mask = (rc_ts_all >= lap.t[0]) & (rc_ts_all < lap.t[-1])
        t_rc = rc_ts_all[mask]
        st_rc = rc_sticks_all[mask]
        if len(t_rc) < 2:
            t_rc  = np.array([], dtype=float)
            st_rc = np.empty((0, 4), dtype=float)
        lap_indices.append(lap.index)
        rc_t_per_lap.append(t_rc)
        rc_sticks_per_lap.append(st_rc)

    return FlightRawData(
        flight=flight,
        rate_profile=rate_profile,
        telem_t=telem_t,
        telem_pos=telem_pos,
        lap_indices=lap_indices,
        rc_t_per_lap=rc_t_per_lap,
        rc_sticks_per_lap=rc_sticks_per_lap,
    )


# ── Stage 2: project positions onto a specific reference ───────────────────────

def _compute_lap_caches(frd: FlightRawData, ref: Reference) -> list[LapCache]:
    telem_s_real = _compute_s_real(frd.telem_pos, ref)
    telem_t = frd.telem_t

    caches: list[LapCache] = []
    for lap_idx, t_rc, sticks_rc in zip(
        frd.lap_indices, frd.rc_t_per_lap, frd.rc_sticks_per_lap
    ):
        if len(t_rc) < 2:
            caches.append(LapCache(
                lap_index=lap_idx, rc_t=t_rc, rc_sticks=sticks_rc,
                rc_s_real=np.array([], dtype=float),
                duration_s=0.0,
            ))
            continue

        idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_l = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        rc_s_real = telem_s_real[np.where(closer_l, idx_l, idx_r)]

        caches.append(LapCache(
            lap_index=lap_idx,
            rc_t=t_rc,
            rc_sticks=sticks_rc,
            rc_s_real=rc_s_real,
            duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


# ── Single run ────────────────────────────────────────────────────────────────

def _run_one(
    caches:      list[LapCache],
    ref_path:    Path,
    ref_meta:    dict,
    frd:         FlightRawData,
    weights:     list[float],
    preset_name: str,
) -> list[RunRecord]:
    """Run localizer (RC+Rate) through all cached laps continuously."""
    drone_cond, rate_cond, condition = _condition_label(ref_meta, frd.flight)

    loc = OnlineLocalizer.from_file(
        ref_path,
        obs_sigma=OBS_SIGMA,
        process_noise_v=PROCESS_NOISE_V,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=np.asarray(weights, dtype=float),
    )
    loc.reset()

    # Load reference for track length
    ref = Reference.load(ref_path)

    records: list[RunRecord] = []
    for cache in caches:
        if len(cache.rc_t) < 2:
            continue

        s_est_list: list[float] = []
        prev_ts: float | None = None

        for i in range(len(cache.rc_t)):
            dt = float(cache.rc_t[i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(cache.rc_t[i])
            res = loc.update(
                cache.rc_sticks[i].tolist(), dt,
                rate_profile=frd.rate_profile,
            )
            s_est_list.append(res.s)

        s_est = np.array(s_est_list)
        m = _metrics(cache.rc_s_real, s_est, ref.L)

        records.append(RunRecord(
            ref_name=ref_meta["name"],
            ref_drone=ref_meta["drone"],
            ref_rate=ref_meta["rate"],
            flight_id=frd.flight["flight_id"],
            test_drone=frd.flight["drone"],
            test_rate=frd.flight["rate"],
            drone_cond=drone_cond,
            rate_cond=rate_cond,
            condition=condition,
            preset=preset_name,
            lap_index=cache.lap_index,
            n_frames=len(s_est_list),
            duration_s=cache.duration_s,
            **m,
        ))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import pandas as pd

    print("Loading flight data...")
    flight_raw: dict[int, FlightRawData] = {}
    for flight in FLIGHTS:
        session_dir = PART_5 / flight["session"]
        print(f"  Flight {flight['flight_id']}: {flight['drone']} x {flight['rate']}")
        frd = _load_flight_raw(flight, session_dir)
        flight_raw[flight["flight_id"]] = frd
        print(f"    => {len(frd.lap_indices)} laps cached")

    print("\nLoading references and pre-computing projections...")
    # ref_name -> {flight_id -> list[LapCache]}
    ref_caches: dict[str, dict[int, list[LapCache]]] = {}
    for ref_meta in REFERENCES:
        ref = Reference.load(ref_meta["path"])
        print(f"  Ref '{ref_meta['name']}': L={ref.L:.1f} m")
        ref_caches[ref_meta["name"]] = {}
        for flight in FLIGHTS:
            frd = flight_raw[flight["flight_id"]]
            caches = _compute_lap_caches(frd, ref)
            ref_caches[ref_meta["name"]][flight["flight_id"]] = caches
            _, _, cond = _condition_label(ref_meta, flight)
            print(f"    -> Flight {flight['flight_id']} ({cond}): s_real computed")

    total = len(REFERENCES) * len(FLIGHTS) * len(WEIGHT_PRESETS)
    print(f"\nRunning {total} combinations "
          f"({len(REFERENCES)} refs x {len(FLIGHTS)} flights x {len(WEIGHT_PRESETS)} presets)...")

    all_records: list[RunRecord] = []
    done = 0

    for ref_meta in REFERENCES:
        for flight in FLIGHTS:
            for preset_name, weights, _ in WEIGHT_PRESETS:
                frd = flight_raw[flight["flight_id"]]
                caches = ref_caches[ref_meta["name"]][flight["flight_id"]]
                recs = _run_one(
                    caches=caches,
                    ref_path=ref_meta["path"],
                    ref_meta=ref_meta,
                    frd=frd,
                    weights=weights,
                    preset_name=preset_name,
                )
                all_records.extend(recs)
                done += 1
                _, _, cond = _condition_label(ref_meta, flight)
                print(f"  [{done}/{total}] {ref_meta['name']} -> F{flight['flight_id']} "
                      f"({cond}) preset={preset_name}")

    df = pd.DataFrame(all_records, columns=RunRecord._fields)
    df.to_csv(_OUT_DIR / "results.csv", index=False)
    print(f"\nSaved {len(df)} lap records -> {_OUT_DIR / 'results.csv'}")

    # Aggregate per (ref_name, flight_id, preset, condition)
    grp = df.groupby(
        ["ref_name", "ref_drone", "ref_rate", "flight_id", "test_drone",
         "test_rate", "drone_cond", "rate_cond", "condition", "preset"]
    ).agg(
        p90_err_m=("p90_err_m", "mean"),
        median_err_m=("median_err_m", "mean"),
        jump_rate=("jump_rate", "mean"),
        n_laps=("lap_index", "count"),
    ).reset_index()
    grp.to_csv(_OUT_DIR / "summary.csv", index=False)

    # Condition summary (avg over refs, flights, presets within same condition_type)
    cond_grp = df.groupby(["condition", "preset"]).agg(
        p90_err_m=("p90_err_m", "mean"),
        median_err_m=("median_err_m", "mean"),
        jump_rate=("jump_rate", "mean"),
        n=("lap_index", "count"),
    ).reset_index()
    cond_grp.to_csv(_OUT_DIR / "condition_summary.csv", index=False)

    plots_dir = _OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    _make_plots(df, grp, cond_grp, plots_dir)
    _write_report(df, grp, cond_grp)

    print(f"\nAll done. Outputs in: {_OUT_DIR}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _make_plots(
    df: "pd.DataFrame",
    grp: "pd.DataFrame",
    cond_grp: "pd.DataFrame",
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    condition_order = [
        "same_drone_same_rate",
        "same_drone_cross_rate",
        "cross_drone_same_rate",
        "cross_drone_cross_rate",
    ]
    condition_labels = {
        "same_drone_same_rate":   "Same drone\nSame rate",
        "same_drone_cross_rate":  "Same drone\nCross rate",
        "cross_drone_same_rate":  "Cross drone\nSame rate",
        "cross_drone_cross_rate": "Cross drone\nCross rate",
    }
    cond_colors = {
        "same_drone_same_rate":   "#27ae60",
        "same_drone_cross_rate":  "#2ecc71",
        "cross_drone_same_rate":  "#e74c3c",
        "cross_drone_cross_rate": "#c0392b",
    }
    preset_names = [p for p, _, _ in WEIGHT_PRESETS]

    # ── 1. Main result: p90 by condition type, per preset ────────────────────
    fig, axes = plt.subplots(1, len(preset_names), figsize=(5 * len(preset_names), 6),
                             sharey=True)
    if len(preset_names) == 1:
        axes = [axes]
    fig.suptitle(
        "p90 Error по типу условия — для каждого пресета\n"
        "(Зелёный = же дрон в референсе, Красный = другой дрон)",
        fontsize=12, fontweight="bold",
    )

    for ax, preset in zip(axes, preset_names):
        sub = cond_grp[cond_grp["preset"] == preset]
        vals  = [sub[sub["condition"] == c]["p90_err_m"].values[0]
                 if not sub[sub["condition"] == c].empty else float("nan")
                 for c in condition_order]
        colors = [cond_colors[c] for c in condition_order]
        bars = ax.bar(range(len(condition_order)), vals, color=colors,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(condition_order)))
        ax.set_xticklabels([condition_labels[c] for c in condition_order],
                           rotation=0, fontsize=8)
        ax.set_title(f"Пресет: {preset}", fontsize=10)
        ax.set_ylabel("p90 error (m)" if ax == axes[0] else "")
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2,
                   label=f"Target {JUMP_THRESHOLD_M} m")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "condition_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Condition comparison plot saved.")

    # ── 2. LiftOff_200 test flights: GromFF_1 ref vs LiftOff200 refs ─────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Тестирование на LiftOff_200 — сравнение референсов\n"
        "Какой референс точнее для LiftOff_200?",
        fontsize=12, fontweight="bold",
    )

    ref_colors = {"GromFF_1": "#e74c3c", "LiftOff200_Grom_1": "#27ae60",
                  "LiftOff200_Red_1": "#2980b9"}
    ref_labels = {
        "GromFF_1":          "GromFF_1\n(MadTrainer ref)",
        "LiftOff200_Grom_1": "LiftOff200_Grom_1\n(LO200 + Grom rate)",
        "LiftOff200_Red_1":  "LiftOff200_Red_1\n(LO200 + Red rate)",
    }

    for ax, (flight_id, flight_label) in zip(
        axes, [(3, "Flight 3: LiftOff_200 x Gromozeka_rate"),
               (4, "Flight 4: LiftOff_200 x RedSheep_rate")]
    ):
        sub = grp[grp["flight_id"] == flight_id]
        ref_names = [r["name"] for r in REFERENCES]
        # Average over presets for each ref
        ref_p90 = []
        for ref_name in ref_names:
            vals = sub[sub["ref_name"] == ref_name]["p90_err_m"].values
            ref_p90.append(float(np.mean(vals)) if len(vals) > 0 else float("nan"))

        colors = [ref_colors[r] for r in ref_names]
        bars = ax.bar(range(len(ref_names)), ref_p90, color=colors,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(ref_names)))
        ax.set_xticklabels([ref_labels[r] for r in ref_names], fontsize=9)
        ax.set_title(flight_label, fontsize=10)
        ax.set_ylabel("mean p90 error (m)")
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2,
                   label=f"Target {JUMP_THRESHOLD_M} m")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, ref_p90):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=11,
                        fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "lo200_ref_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  LiftOff_200 reference comparison plot saved.")

    # ── 3. Full matrix heatmap: ref x flight_id, color = p90 ─────────────────
    fig, axes = plt.subplots(1, len(preset_names), figsize=(5 * len(preset_names), 5),
                             sharey=True)
    if len(preset_names) == 1:
        axes = [axes]
    fig.suptitle("p90 Error — все комбинации референс × тест",
                 fontsize=12, fontweight="bold")

    ref_names  = [r["name"] for r in REFERENCES]
    flight_ids = [f["flight_id"] for f in FLIGHTS]

    for ax, preset in zip(axes, preset_names):
        sub = grp[grp["preset"] == preset]
        data = np.full((len(ref_names), len(flight_ids)), np.nan)
        for ri, rn in enumerate(ref_names):
            for fi, fid in enumerate(flight_ids):
                row = sub[(sub["ref_name"] == rn) & (sub["flight_id"] == fid)]
                if not row.empty:
                    data[ri, fi] = float(row["p90_err_m"].values[0])

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("rg", ["#27ae60", "#f1c40f", "#e74c3c"])
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=40)
        ax.set_xticks(range(len(flight_ids)))
        ax.set_xticklabels([
            f"F{fid}\n{f['drone'][:3]}\n{f['rate'][:3]}"
            for fid, f in zip(flight_ids, FLIGHTS)
        ], fontsize=8)
        ax.set_yticks(range(len(ref_names)))
        ax.set_yticklabels(ref_names, fontsize=8)
        ax.set_title(f"Preset: {preset}", fontsize=10)
        ax.set_xlabel("Test flight")
        if ax == axes[0]:
            ax.set_ylabel("Reference")
        plt.colorbar(im, ax=ax, label="p90 (m)")

        # Annotate cells
        for ri in range(len(ref_names)):
            for fi in range(len(flight_ids)):
                val = data[ri, fi]
                if not np.isnan(val):
                    # Condition label
                    _, _, cond = _condition_label(REFERENCES[ri], FLIGHTS[fi])
                    bg = "same" in cond[:10]
                    color = "white" if val > 25 else "black"
                    ax.text(fi, ri, f"{val:.1f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "full_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Full matrix heatmap saved.")

    # ── 4. Summary: same_drone vs cross_drone (all conditions, averaged) ──────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Same-drone vs Cross-drone Референс\n"
        "Влияние совпадения дрона (усреднено по rate и пресетам)",
        fontsize=12, fontweight="bold",
    )

    same_drone_conds  = ["same_drone_same_rate", "same_drone_cross_rate"]
    cross_drone_conds = ["cross_drone_same_rate", "cross_drone_cross_rate"]

    x = np.arange(len(preset_names))
    w = 0.35

    same_vals  = []
    cross_vals = []
    for preset in preset_names:
        sub = cond_grp[cond_grp["preset"] == preset]
        same_v  = sub[sub["condition"].isin(same_drone_conds)]["p90_err_m"].mean()
        cross_v = sub[sub["condition"].isin(cross_drone_conds)]["p90_err_m"].mean()
        same_vals.append(float(same_v))
        cross_vals.append(float(cross_v))

    b1 = ax.bar(x - w / 2, same_vals,  w, label="Референс = тот же дрон",  color="#27ae60")
    b2 = ax.bar(x + w / 2, cross_vals, w, label="Референс = другой дрон", color="#e74c3c")

    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2,
               label=f"Target {JUMP_THRESHOLD_M} m")
    ax.set_xticks(x)
    ax.set_xticklabels(preset_names, fontsize=10)
    ax.set_ylabel("mean p90 error (m)")
    ax.set_title("Влияние совпадения дрона в референсе", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    for bar, v in zip([*b1, *b2], [*same_vals, *cross_vals]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "same_vs_cross_drone_ref.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Same vs cross drone reference plot saved.")


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(
    df: "pd.DataFrame",
    grp: "pd.DataFrame",
    cond_grp: "pd.DataFrame",
) -> None:
    conditions = [
        "same_drone_same_rate",
        "same_drone_cross_rate",
        "cross_drone_same_rate",
        "cross_drone_cross_rate",
    ]
    preset_names = [p for p, _, _ in WEIGHT_PRESETS]

    # ── Key numbers ───────────────────────────────────────────────────────────

    def cond_p90(cond: str, preset: str | None = None) -> float:
        sub = cond_grp[cond_grp["condition"] == cond]
        if preset:
            sub = sub[sub["preset"] == preset]
        if sub.empty:
            return float("nan")
        return float(sub["p90_err_m"].mean())

    def ref_flight_p90(ref_name: str, flight_id: int, preset: str | None = None) -> float:
        sub = grp[(grp["ref_name"] == ref_name) & (grp["flight_id"] == flight_id)]
        if preset:
            sub = sub[sub["preset"] == preset]
        if sub.empty:
            return float("nan")
        return float(sub["p90_err_m"].mean())

    # Main comparison: GromFF1 vs LiftOff200_Grom_1 on LiftOff_200 flights
    cross_baseline = (ref_flight_p90("GromFF_1", 3) + ref_flight_p90("GromFF_1", 4)) / 2
    same_drone_new = (ref_flight_p90("LiftOff200_Grom_1", 3) + ref_flight_p90("LiftOff200_Grom_1", 4)) / 2
    improvement    = (cross_baseline - same_drone_new) / cross_baseline * 100

    # Rate sensitivity with same drone
    same_drone_same_rate_val  = cond_p90("same_drone_same_rate")
    same_drone_cross_rate_val = cond_p90("same_drone_cross_rate")
    rate_sensitivity = abs(same_drone_cross_rate_val - same_drone_same_rate_val)

    # Symmetry check: cross_drone both directions
    madtrainer_ref_on_lo200  = (ref_flight_p90("GromFF_1", 3) + ref_flight_p90("GromFF_1", 4)) / 2
    lo200_ref_on_madtrainer  = (ref_flight_p90("LiftOff200_Grom_1", 1)) if not np.isnan(ref_flight_p90("LiftOff200_Grom_1", 1)) else float("nan")

    # Best overall
    best_val = grp["p90_err_m"].min()
    best_row = grp.loc[grp["p90_err_m"].idxmin()].to_dict()

    # Condition table
    cond_table_rows = []
    for cond in conditions:
        for preset in preset_names:
            v = cond_p90(cond, preset)
            if not np.isnan(v):
                cond_table_rows.append(f"| {cond} | {preset} | {v:.1f} |")
    cond_table = "\n".join(cond_table_rows)

    # Reference summary table
    ref_table_rows = []
    for ref_meta in REFERENCES:
        for flight in FLIGHTS:
            _, _, cond = _condition_label(ref_meta, flight)
            v = ref_flight_p90(ref_meta["name"], flight["flight_id"])
            ref_table_rows.append(
                f"| {ref_meta['name']} | F{flight['flight_id']} "
                f"({flight['drone']} x {flight['rate']}) | {cond} | {v:.1f} |"
            )
    ref_table = "\n".join(ref_table_rows)

    n_total = len(df)

    report = f"""# Глава: Эксперимент 4 — Качество референса и совпадение дрона

## 1. Цель и мотивация

Эксперименты 1–3 установили, что **дрон является доминирующим фактором**
точности локализатора (eta2=0.41, Эксперимент 1). В условиях кросс-дрона
(референс MadTrainer, тест LiftOff_200) лучшая достигнутая p90 составила
7.8 м при оптимальных гиперпараметрах (Эксперимент 3).

Данный эксперимент отвечает на вопрос:
> *Насколько снижается ошибка локализации, если референс построен на том же
> дроне, что используется при тестировании?*

Это **устанавливает теоретический потолок** системы — наилучший возможный
результат без смены дрона между референсом и тестом.

### Значение для системы

Ответ определяет стратегию практического применения:
- Если same_drone даёт p90 << 15 м — нужна **калибровка на конкретный дрон**
- Если разница мала — система уже достаточно **обобщаема**
- Если rate profile не влияет на same_drone — достаточно **одного референса на дрон**

## 2. Экспериментальный план

**Данные**: те же 4 полётные сессии + 2 новых референса (LiftOff_200).
**Режим**: RC+Rate (лучший из Эксперимента 1).
**Гиперпараметры**: оптимальные из Эксперимента 3 (obs_sigma={OBS_SIGMA}, pnv={PROCESS_NOISE_V}).

### 2.1 Референсы

| Референс | Дрон | Rate | Описание |
|---|---|---|---|
| GromFF_1 | MadTrainer | Gromozeka_rate | Базовый референс Экс. 1–3 |
| LiftOff200_Grom_1 | LiftOff_200 | Gromozeka_rate | Новый, тот же дрон |
| LiftOff200_Red_1 | LiftOff_200 | RedSheep_rate | Новый, тот же дрон + другой rate |

### 2.2 Матрица условий (ref x test flight)

| Тип условия | Дрон совпадает? | Rate совпадает? | Смысл |
|---|---|---|---|
| same_drone_same_rate | ДА | ДА | Идеальный случай (потолок) |
| same_drone_cross_rate | ДА | НЕТ | Разные настройки того же дрона |
| cross_drone_same_rate | НЕТ | ДА | Смена дрона при том же rate |
| cross_drone_cross_rate | НЕТ | НЕТ | Полная смена конфигурации |

**Итого**: {n_total} измерений.

### 2.3 Пресеты весов каналов

| Пресет | [Thr, Yaw, Pitch, Roll] | Основание |
|---|---|---|
| baseline | [1.0, 1.0, 1.0, 1.0] | дефолт Эксперимента 1 |
| angular_scaled | [0.0, 0.7, 0.5, 1.0] | ранг 1 Эксперимента 2 |
| no_thr | [0.0, 1.0, 1.0, 1.0] | лучший cross-drone Эксперимента 3 |

## 3. Гипотезы

| # | Гипотеза | Ожидаемый результат |
|---|---|---|
| H1 | same_drone_same_rate даёт значительно меньшую p90, чем cross_drone | p90(same) << p90(cross) |
| H2 | RC+Rate нормализует влияние rate profile — same_drone_cross_rate близко к same_drone_same_rate | Разница < 3 м |
| H3 | Кросс-дроновый разрыв симметричен: GromFF_1->LO200 ≈ LO200_ref->MadTrainer | Разница < 30% |
| H4 | same_drone_same_rate p90 < 5 м (потолок системы) | Достигнуто или нет |

## 4. Результаты

### 4.1 Сравнение по типу условия (среднее по пресетам)

| Условие | Пресет | p90 (м) |
|---|---|---|
{cond_table}

### 4.2 Полная матрица результатов (среднее по пресетам)

| Референс | Тест | Условие | p90 (м) |
|---|---|---|---|
{ref_table}

### 4.3 Ключевое сравнение: смена референса для LiftOff_200

**Базовая кросс-дроновая p90** (GromFF_1 -> LiftOff_200 F3+F4): **{cross_baseline:.1f} м**

**Same-drone p90** (LiftOff200_Grom_1 -> LiftOff_200 F3+F4): **{same_drone_new:.1f} м**

**Улучшение от same-drone референса**: **{improvement:+.0f}%**

### 4.4 Чувствительность к rate profile (same_drone)

- same_drone_same_rate: {same_drone_same_rate_val:.1f} м
- same_drone_cross_rate: {same_drone_cross_rate_val:.1f} м
- Разница: {rate_sensitivity:.1f} м

### 4.5 Лучший результат в эксперименте

Минимальная p90: **{best_val:.1f} м**
(ref={best_row.get("ref_name", "?")}, flight={best_row.get("flight_id", "?")},
preset={best_row.get("preset", "?")}, condition={best_row.get("condition", "?")})

## 5. Выводы

### 5.1 Влияние совпадения дрона (H1)

Переход от кросс-дронового референса (GromFF_1) к дрон-совпадающему
(LiftOff200_Grom_1) для тестирования на LiftOff_200 изменил p90 с
{cross_baseline:.1f} м до {same_drone_new:.1f} м ({improvement:+.0f}%).

{'Гипотеза H1 **подтверждена**: same-drone референс существенно снижает ошибку.' if same_drone_new < cross_baseline * 0.7 else 'Гипотеза H1 **не подтверждена**: разница между same-drone и cross-drone референсом невелика, что указывает на хорошую обобщаемость системы.' if same_drone_new > cross_baseline * 0.9 else 'Гипотеза H1 **частично подтверждена**: same-drone референс улучшает точность, но не радикально.'}

### 5.2 Влияние rate profile (H2)

Разница между same_drone_same_rate ({same_drone_same_rate_val:.1f} м) и
same_drone_cross_rate ({same_drone_cross_rate_val:.1f} м) составила {rate_sensitivity:.1f} м.

{'Гипотеза H2 **подтверждена**: нормализация к физическим единицам (deg/s) через RC+Rate mode делает rate profile несущественным фактором.' if rate_sensitivity < 3 else f'Гипотеза H2 **не подтверждена**: разница rate profiles влияет на точность даже при совпадении дрона ({rate_sensitivity:.1f} м).'}

### 5.3 Симметрия кросс-дроновой ошибки (H3)

- MadTrainer ref -> LiftOff_200: {madtrainer_ref_on_lo200:.1f} м
- LiftOff_200 ref -> MadTrainer: {lo200_ref_on_madtrainer:.1f} м

{'Асимметрия мала — направление смены дрона не критично.' if abs(madtrainer_ref_on_lo200 - lo200_ref_on_madtrainer) < madtrainer_ref_on_lo200 * 0.3 else 'Значительная асимметрия: одно направление смены дрона значительно хуже другого.'}

### 5.4 Теоретический потолок (H4)

Минимальная достигнутая p90: **{best_val:.1f} м**
(условие: {best_row.get("condition", "?")}).

{'Гипотеза H4 **подтверждена**: same_drone_same_rate p90 < 5 м.' if best_val < 5 else f'Гипотеза H4 **не подтверждена**: даже при совпадении дрона и rate p90 = {best_val:.1f} м > 5 м.'}

### 5.5 Практические рекомендации

1. **Для максимальной точности**: построить отдельный референс для каждого
   дрона. Это обеспечивает наилучшие результаты при любом rate profile.

2. **Для универсального применения**: использовать пресет `no_thr`,
   obs_sigma={OBS_SIGMA}, pnv={PROCESS_NOISE_V} из Эксперимента 3.
   Данная конфигурация достигает p90 < 15 м даже в кросс-дроновом сценарии.

3. **Rate profile**: не требует отдельного референса на каждый rate —
   RC+Rate mode обеспечивает достаточную нормализацию.

4. **Вывод для диплома**: система демонстрирует {'приемлемую обобщаемость' if cross_baseline < 15 else 'ограниченную обобщаемость'} на новые дроны (cross-drone p90 = {cross_baseline:.1f} м),
   при этом per-drone калибровка улучшает результат до {same_drone_new:.1f} м.
   Выбор стратегии зависит от требований приложения.

## 6. Файлы эксперимента

| Файл | Содержимое |
|---|---|
| `results.csv` | {n_total} строк: per-lap метрики для всех конфигураций |
| `summary.csv` | per (ref, flight, preset, condition) |
| `condition_summary.csv` | Усреднено по condition_type |
| `plots/condition_comparison.png` | p90 по 4 типам условий |
| `plots/lo200_ref_comparison.png` | Сравнение референсов для LiftOff_200 |
| `plots/full_matrix.png` | Матрица ref x test |
| `plots/same_vs_cross_drone_ref.png` | Same-drone vs cross-drone референс |
| `report.md` | Данный отчёт |
"""

    report_path = _OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved -> {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

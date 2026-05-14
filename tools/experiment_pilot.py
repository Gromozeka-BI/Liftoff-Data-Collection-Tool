"""Experiment 5: Cross-Pilot Generalization on track-002.

Исследовательский вопрос
------------------------
Может ли локализатор, построенный на основе референса пилота A,
точно отслеживать полёт другого пилота на той же трассе?
Является ли стиль пилота более значимым фактором, чем тип дрона (Exp 1, eta2=0.41)?

Пилоты
------
Gromozeka  — Пилот 1, референс GromFF_1     [существует]
RedSeep    — Пилот 2, референс RedSeep_S2   [построен]
PlatOnAir  — Пилот 3, референс PlatOnAir_1  [построен]

Все трое используют дрон MadTrainer; у каждого свой профиль рейтов Betaflight.

Дизайн
------
3 референса × 3 тестовых полёта × 3 пресета весов × N_LAST_LAPS кругов.

Референсы (построены из специальных сессий):
  GromFF_1    — Part_5 / session-001, круг 4  (Gromozeka_rate,  ~25 с/круг)
  RedSeep_S2  — Part_9 / session-002, круг 3  (RedSheep_rate,   ~18 с/круг)
  PlatOnAir_1 — Part_10 / session-027, круг 5 (PlatOnAir_rate,  ~17 с/круг)

Тестовые сессии (та же сессия, что для референса, или близкая по скорости):
  Gromozeka:  Part_5 / session-004  (15 кругов, ~24 с, Gromozeka_rate)
  RedSeep:    Part_9 / session-002  (8 кругов,  ~17–18 с, RedSheep_rate)
  PlatOnAir:  Part_10 / session-027 (7 кругов,  ~16–17 с, PlatOnAir_rate)

Важно: для RedSeep и PlatOnAir та же сессия используется для референса и теста
(разные круги). Это гарантирует совпадение пространственной траектории и скорости.

Метки условий:
  same_pilot  — референс и тест от одного пилота
  cross_pilot — референс и тест от разных пилотов

Фиксированные гиперпараметры (оптимум из Exp 3):
  obs_sigma       = 2.0
  process_noise_v = 8.0
  process_noise_s = 1.5
  Режим           = RC+Rate (собственный профиль рейтов каждого пилота)

Примечание по v_init: вычисляется автоматически из длины трассы и числа кадров
референса (L / lap_time). Дефолт 10 м/с не подходит для быстрых пилотов (17–18 м/с).

Выходные файлы (tools/exp5_pilot/):
  results.csv           — одна строка на (реф, полёт, пресет, круг)
  summary.csv           — агрегация по (реф, полёт, пресет)
  condition_summary.csv — агрегация по (условие, пресет)
  plots/                — графики
  report.md             — финальный отчёт (вручную, на русском)

Запуск (из корня проекта):
    python tools/experiment_pilot.py
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

# ── Configuration ─────────────────────────────────────────────────────────────

def _find_part(prefix: str) -> Path:
    liftoff = Path(r"D:\DroneTrackerDB\Liftoff")
    candidates = sorted([d for d in liftoff.iterdir() if d.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No {prefix}* directory found under {liftoff}")
    return candidates[0]


PART_5  = _find_part("Part_5")
PART_9  = _find_part("Part_9")
PART_10 = _find_part("Part_10")

REF_DIR  = ROOT / "tracks" / "track-002" / "references"
_OUT_DIR = Path(__file__).parent / "exp5_pilot"
_OUT_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0
N_LAST_LAPS      = 3

# ── Optimal hyperparameters (Exp 3) ───────────────────────────────────────────
OBS_SIGMA       = 2.0
PROCESS_NOISE_V = 8.0
PROCESS_NOISE_S = 1.5

# fmt: off
WEIGHT_PRESETS: list[tuple[str, list[float]]] = [
    ("baseline",       [1.0, 1.0, 1.0, 1.0]),
    ("angular_scaled", [0.0, 0.7, 0.5, 1.0]),
    ("no_thr",         [0.0, 1.0, 1.0, 1.0]),
]
# fmt: on

REFERENCES: list[dict] = [
    dict(
        name="GromFF_1",
        path=REF_DIR / "GromFF_1.npz",
        pilot="Gromozeka",
        rate="Gromozeka_rate",
        desc="Gromozeka — Part_5/session-001 круг 4 (~25 с/круг)",
    ),
    dict(
        name="RedSeep_S2",
        path=REF_DIR / "RedSeep_S2.npz",
        pilot="RedSeep",
        rate="RedSheep_rate",
        desc="RedSeep — Part_9/session-002 лучший круг (~18 с/круг)",
    ),
    dict(
        name="PlatOnAir_1",
        path=REF_DIR / "PlatOnAir_1.npz",
        pilot="PlatOnAir",
        rate="PlatOnAir_rate",
        desc="PlatOnAir — Part_10/session-027 лучший круг (~17 с/круг)",
    ),
]

# Тестовые сессии: для RedSeep и PlatOnAir — та же сессия, что и для референса.
FLIGHTS: list[dict] = [
    dict(
        flight_id="Grom-S4",
        pilot="Gromozeka",
        rate="Gromozeka_rate",
        session="2026-05-09_pilot-Gromozeka_drone-MadTrainer_track-track-002_session-004",
        base=PART_5,
        desc="Gromozeka тест session-004 (15 кругов, ~24 с каждый)",
    ),
    dict(
        flight_id="Red-S2",
        pilot="RedSeep",
        rate="RedSheep_rate",
        session="2026-05-10_pilot-RedSeep_drone-MadTrainer_track-track-002_session-002",
        base=PART_9,
        desc="RedSeep тест session-002 (8 кругов, ~17–18 с) — та же сессия, что у референса",
    ),
    dict(
        flight_id="Plat-S27",
        pilot="PlatOnAir",
        rate="PlatOnAir_rate",
        session="2026-05-10_pilot-PlatOnAir_drone-MadTrainer_track-track-002_session-027",
        base=PART_10,
        desc="PlatOnAir тест session-027 (7 кругов, ~16–17 с) — та же сессия, что у референса",
    ),
]

_RC_CH_ORDER      = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FlightRawData:
    flight:            dict
    rate_profile:      dict
    telem_t:           np.ndarray
    telem_pos:         np.ndarray
    lap_indices:       list[int]
    rc_t_per_lap:      list[np.ndarray]
    rc_sticks_per_lap: list[np.ndarray]


@dataclass
class LapCache:
    lap_index:  int
    rc_t:       np.ndarray
    rc_sticks:  np.ndarray
    rc_s_real:  np.ndarray
    duration_s: float


class RunRecord(NamedTuple):
    ref_name:     str
    ref_pilot:    str
    ref_rate:     str
    flight_id:    str
    test_pilot:   str
    test_rate:    str
    pilot_cond:   str    # "same_pilot" | "cross_pilot"
    preset:       str
    lap_index:    int
    n_frames:     int
    duration_s:   float
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


def _pilot_cond(ref: dict, flight: dict) -> str:
    return "same_pilot" if ref["pilot"] == flight["pilot"] else "cross_pilot"


# ── Stage 1: load raw flight data ──────────────────────────────────────────────

def _load_flight_raw(flight: dict) -> FlightRawData:
    import pandas as pd

    session_dir = flight["base"] / flight["session"]
    laps, _ = load_dct_session(session_dir)
    laps = filter_anomalous_laps(laps)
    rate_profile = load_rate_profile(session_dir)
    selected = laps[-N_LAST_LAPS:]

    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts_all     = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks_all = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks_all[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    telem_t   = np.concatenate([lap.t   for lap in selected])
    telem_pos = np.vstack([lap.pos       for lap in selected])

    lap_indices:       list[int]          = []
    rc_t_per_lap:      list[np.ndarray]   = []
    rc_sticks_per_lap: list[np.ndarray]   = []

    for lap in selected:
        mask  = (rc_ts_all >= lap.t[0]) & (rc_ts_all < lap.t[-1])
        t_rc  = rc_ts_all[mask]
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


# ── Stage 2: project positions onto a reference ────────────────────────────────

def _compute_lap_caches(frd: FlightRawData, ref: Reference) -> list[LapCache]:
    telem_s_real = _compute_s_real(frd.telem_pos, ref)
    telem_t      = frd.telem_t

    caches: list[LapCache] = []
    for lap_idx, t_rc, sticks_rc in zip(
        frd.lap_indices, frd.rc_t_per_lap, frd.rc_sticks_per_lap
    ):
        if len(t_rc) < 2:
            caches.append(LapCache(
                lap_index=lap_idx, rc_t=t_rc, rc_sticks=sticks_rc,
                rc_s_real=np.array([], dtype=float), duration_s=0.0,
            ))
            continue

        idx_r = np.clip(np.searchsorted(telem_t, t_rc), 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_l  = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        rc_s_real = telem_s_real[np.where(closer_l, idx_l, idx_r)]

        caches.append(LapCache(
            lap_index=lap_idx,
            rc_t=t_rc,
            rc_sticks=sticks_rc,
            rc_s_real=rc_s_real,
            duration_s=float(t_rc[-1] - t_rc[0]),
        ))
    return caches


# ── Single run ─────────────────────────────────────────────────────────────────

def _estimate_v_init(ref: Reference, rc_hz: float = 100.0) -> float:
    """Оценить скорость дрона из референса: L / (n_frames / RC_Hz).

    Дефолтное v_init=10 м/с не подходит для быстрых пилотов (17–18 м/с) —
    частицы не успевают за дроном и фильтр расходится.
    """
    lap_time_s = ref.sticks_norm.shape[0] / rc_hz
    return float(ref.L / lap_time_s)


def _run_one(
    caches:      list[LapCache],
    ref_meta:    dict,
    frd:         FlightRawData,
    weights:     list[float],
    preset_name: str,
) -> list[RunRecord]:
    pilot_cond = _pilot_cond(ref_meta, frd.flight)

    ref    = Reference.load(ref_meta["path"])
    v_init = _estimate_v_init(ref)

    loc = OnlineLocalizer.from_file(
        ref_meta["path"],
        obs_sigma=OBS_SIGMA,
        process_noise_v=PROCESS_NOISE_V,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=np.asarray(weights, dtype=float),
        v_init_mps=v_init,
        v_init_std=max(3.0, v_init * 0.25),
    )
    loc.reset()

    records: list[RunRecord] = []
    for cache in caches:
        if len(cache.rc_t) < 2:
            continue

        s_est_list: list[float] = []
        prev_ts: float | None   = None

        for i in range(len(cache.rc_t)):
            dt      = float(cache.rc_t[i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(cache.rc_t[i])
            res     = loc.update(
                cache.rc_sticks[i].tolist(), dt,
                rate_profile=frd.rate_profile,
            )
            s_est_list.append(res.s)

        s_est = np.array(s_est_list)
        m     = _metrics(cache.rc_s_real, s_est, ref.L)

        records.append(RunRecord(
            ref_name=ref_meta["name"],
            ref_pilot=ref_meta["pilot"],
            ref_rate=ref_meta["rate"],
            flight_id=frd.flight["flight_id"],
            test_pilot=frd.flight["pilot"],
            test_rate=frd.flight["rate"],
            pilot_cond=pilot_cond,
            preset=preset_name,
            lap_index=cache.lap_index,
            n_frames=len(s_est_list),
            duration_s=cache.duration_s,
            **m,
        ))
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import pandas as pd

    print("=== Experiment 5: Cross-Pilot Generalization ===\n")
    print("Checking inputs...")
    ok = True
    for r in REFERENCES:
        status = "[OK]" if r["path"].exists() else "[MISSING]"
        print(f"  Ref {r['name']}: {status}  ({r['path'].name})")
        if not r["path"].exists():
            ok = False
    for f in FLIGHTS:
        session_dir = f["base"] / f["session"]
        status = "[OK]" if session_dir.exists() else "[MISSING]"
        print(f"  Flight {f['flight_id']}: {status}  ({f['session'][-20:]})")
        if not session_dir.exists():
            ok = False
    if not ok:
        print("\nERROR: Some inputs are missing. Cannot proceed.")
        return

    print("\nLoading flight data...")
    flight_raw: dict[str, FlightRawData] = {}
    for flight in FLIGHTS:
        print(f"  {flight['flight_id']}: {flight['pilot']} x {flight['rate']}")
        frd = _load_flight_raw(flight)
        flight_raw[flight["flight_id"]] = frd
        print(f"    => {len(frd.lap_indices)} laps (last {N_LAST_LAPS})")

    print("\nPre-computing s_real projections...")
    ref_caches: dict[str, dict[str, list[LapCache]]] = {}
    for ref_meta in REFERENCES:
        ref = Reference.load(ref_meta["path"])
        print(f"  Ref '{ref_meta['name']}': L={ref.L:.1f} m  ({ref_meta['pilot']})")
        ref_caches[ref_meta["name"]] = {}
        for flight in FLIGHTS:
            frd    = flight_raw[flight["flight_id"]]
            caches = _compute_lap_caches(frd, ref)
            ref_caches[ref_meta["name"]][flight["flight_id"]] = caches
            cond   = _pilot_cond(ref_meta, flight)
            print(f"    -> {flight['flight_id']} ({cond})")

    total = len(REFERENCES) * len(FLIGHTS) * len(WEIGHT_PRESETS)
    print(f"\nRunning {total} combinations "
          f"({len(REFERENCES)} refs x {len(FLIGHTS)} flights x {len(WEIGHT_PRESETS)} presets)...")

    all_records: list[RunRecord] = []
    done = 0
    for ref_meta in REFERENCES:
        for flight in FLIGHTS:
            for preset_name, weights in WEIGHT_PRESETS:
                frd    = flight_raw[flight["flight_id"]]
                caches = ref_caches[ref_meta["name"]][flight["flight_id"]]
                recs   = _run_one(
                    caches=caches,
                    ref_meta=ref_meta,
                    frd=frd,
                    weights=weights,
                    preset_name=preset_name,
                )
                all_records.extend(recs)
                done += 1
                cond = _pilot_cond(ref_meta, flight)
                print(f"  [{done}/{total}] {ref_meta['name']} -> {flight['flight_id']} "
                      f"({cond}) preset={preset_name}  laps={len(recs)}")

    df = pd.DataFrame(all_records, columns=RunRecord._fields)
    df.to_csv(_OUT_DIR / "results.csv", index=False)
    print(f"\nSaved {len(df)} lap records -> {_OUT_DIR / 'results.csv'}")

    grp = df.groupby(
        ["ref_name", "ref_pilot", "ref_rate",
         "flight_id", "test_pilot", "test_rate",
         "pilot_cond", "preset"]
    ).agg(
        p90_err_m=("p90_err_m", "mean"),
        median_err_m=("median_err_m", "mean"),
        jump_rate=("jump_rate", "mean"),
        n_laps=("lap_index", "count"),
    ).reset_index()
    grp.to_csv(_OUT_DIR / "summary.csv", index=False)

    cond_grp = df.groupby(["pilot_cond", "preset"]).agg(
        p90_err_m=("p90_err_m", "mean"),
        median_err_m=("median_err_m", "mean"),
        jump_rate=("jump_rate", "mean"),
        n=("lap_index", "count"),
    ).reset_index()
    cond_grp.to_csv(_OUT_DIR / "condition_summary.csv", index=False)

    plots_dir = _OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    _make_plots(df, grp, cond_grp, plots_dir)

    print(f"\nAll done. Outputs: {_OUT_DIR}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _make_plots(
    df:       "pd.DataFrame",
    grp:      "pd.DataFrame",
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

    pilots       = [r["pilot"] for r in REFERENCES]
    ref_names    = [r["name"]  for r in REFERENCES]
    flight_ids   = [f["flight_id"] for f in FLIGHTS]
    preset_names = [p for p, _ in WEIGHT_PRESETS]

    pilot_colors = {
        "Gromozeka": "#2980b9",
        "RedSeep":   "#e74c3c",
        "PlatOnAir": "#27ae60",
    }

    # ── 1. Условие: тот же пилот vs другой, по пресетам ──────────────────────
    fig, axes = plt.subplots(1, len(preset_names),
                             figsize=(5 * len(preset_names), 6), sharey=True)
    if len(preset_names) == 1:
        axes = [axes]
    fig.suptitle(
        "p90 Error: Тот же пилот vs Другой пилот — по пресетам",
        fontsize=13, fontweight="bold",
    )
    cond_order  = ["same_pilot", "cross_pilot"]
    cond_colors = {"same_pilot": "#27ae60", "cross_pilot": "#e74c3c"}
    cond_labels = {"same_pilot": "Тот же пилот\n(потолок)", "cross_pilot": "Другой пилот"}

    for ax, preset in zip(axes, preset_names):
        sub  = cond_grp[cond_grp["preset"] == preset]
        vals = []
        for c in cond_order:
            row = sub[sub["pilot_cond"] == c]
            vals.append(float(row["p90_err_m"].values[0]) if not row.empty else float("nan"))

        bars = ax.bar(range(len(cond_order)), vals,
                      color=[cond_colors[c] for c in cond_order],
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels([cond_labels[c] for c in cond_order], fontsize=10)
        ax.set_title(f"Пресет: {preset}", fontsize=10)
        ax.set_ylabel("mean p90 error (m)" if ax == axes[0] else "")
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2,
                   label=f"Цель {JUMP_THRESHOLD_M} м")
        ax.axhline(9.7, color="#e67e22", ls=":", lw=1.2,
                   label="Cross-drone Exp4 (9.7 м)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "condition_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] condition_comparison.png")

    # ── 2. Детализация по пилотам: кто кого отслеживает, лучший пресет ───────
    best_preset = "no_thr"
    sub = grp[grp["preset"] == best_preset]

    fig, axes = plt.subplots(1, len(ref_names),
                             figsize=(5 * len(ref_names), 6), sharey=True)
    fig.suptitle(
        f"p90 для каждого тест-пилота при каждом референсе\n(пресет={best_preset})",
        fontsize=12, fontweight="bold",
    )

    for ax, ref_meta in zip(axes, REFERENCES):
        rsub       = sub[sub["ref_name"] == ref_meta["name"]]
        test_pilots = [f["pilot"] for f in FLIGHTS]
        vals       = []
        for fp in test_pilots:
            row = rsub[rsub["test_pilot"] == fp]
            vals.append(float(row["p90_err_m"].values[0]) if not row.empty else float("nan"))

        bars = ax.bar(range(len(test_pilots)), vals,
                      color=[pilot_colors[fp] for fp in test_pilots],
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(test_pilots)))
        ax.set_xticklabels(test_pilots, fontsize=9)
        ax.set_title(f"Реф: {ref_meta['name']}\n({ref_meta['pilot']})", fontsize=10)
        ax.set_ylabel("p90 error (m)" if ax == axes[0] else "")
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2, label="Цель 15 м")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v, fp in zip(bars, vals, test_pilots):
            if not np.isnan(v):
                is_same = fp == ref_meta["pilot"]
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}{'*' if is_same else ''}",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    fig.text(0.5, 0.01, "* = тот же пилот (потолок)", ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(plots_dir / "per_pilot_breakdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] per_pilot_breakdown.png")

    # ── 3. Тепловая карта 3×3: реф-пилот × тест-пилот ────────────────────────
    fig, axes = plt.subplots(1, len(preset_names),
                             figsize=(5 * len(preset_names), 5), sharey=True)
    if len(preset_names) == 1:
        axes = [axes]
    fig.suptitle("Матрица p90: Пилот-референс × Пилот-тест",
                 fontsize=12, fontweight="bold")

    for ax, preset in zip(axes, preset_names):
        psub = grp[grp["preset"] == preset]
        data = np.full((len(pilots), len(pilots)), np.nan)
        for ri, ref_meta in enumerate(REFERENCES):
            for fi, flight in enumerate(FLIGHTS):
                row = psub[
                    (psub["ref_name"]  == ref_meta["name"]) &
                    (psub["flight_id"] == flight["flight_id"])
                ]
                if not row.empty:
                    data[ri, fi] = float(row["p90_err_m"].values[0])

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("rg", ["#27ae60", "#f1c40f", "#e74c3c"])
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=40)
        ax.set_xticks(range(len(pilots)))
        ax.set_xticklabels(pilots, fontsize=9)
        ax.set_yticks(range(len(pilots)))
        ax.set_yticklabels(pilots, fontsize=9)
        ax.set_title(f"Пресет: {preset}", fontsize=10)
        ax.set_xlabel("Тест-пилот")
        if ax == axes[0]:
            ax.set_ylabel("Пилот-референс")
        plt.colorbar(im, ax=ax, label="p90 (м)")

        for ri in range(len(pilots)):
            for fi in range(len(pilots)):
                v = data[ri, fi]
                if not np.isnan(v):
                    color = "white" if v > 25 else "black"
                    ax.text(fi, ri, f"{v:.1f}", ha="center", va="center",
                            fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "pilot_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] pilot_matrix.png")

    # ── 4. Сравнение пресетов: same vs cross, сгруппированные столбцы ────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Тот же пилот vs Другой пилот — по пресетам\n"
        "Влияние весов канала на кросс-пилотную обобщаемость",
        fontsize=12, fontweight="bold",
    )
    x = np.arange(len(preset_names))
    w = 0.35
    same_vals  = []
    cross_vals = []
    for preset in preset_names:
        sub     = cond_grp[cond_grp["preset"] == preset]
        same_v  = sub[sub["pilot_cond"] == "same_pilot"]["p90_err_m"]
        cross_v = sub[sub["pilot_cond"] == "cross_pilot"]["p90_err_m"]
        same_vals.append(float(same_v.values[0])  if len(same_v)  > 0 else float("nan"))
        cross_vals.append(float(cross_v.values[0]) if len(cross_v) > 0 else float("nan"))

    b1 = ax.bar(x - w / 2, same_vals,  w, color="#27ae60",
                label="Тот же пилот",  edgecolor="white")
    b2 = ax.bar(x + w / 2, cross_vals, w, color="#e74c3c",
                label="Другой пилот", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(preset_names, fontsize=11)
    ax.set_ylabel("mean p90 error (м)", fontsize=11)
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2,
               label=f"Цель {JUMP_THRESHOLD_M} м")
    ax.axhline(9.7, color="#e67e22", ls=":", lw=1.2,
               label="Cross-drone Exp4 (9.7 м)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    for bar, v in list(zip(b1, same_vals)) + list(zip(b2, cross_vals)):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "preset_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] preset_comparison.png")


if __name__ == "__main__":
    main()

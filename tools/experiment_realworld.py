"""Experiment 9: Real-World Track Validation (Cross-Reality Localization).

Исследовательский вопрос
------------------------
Сохраняется ли точность системы (p90 < 15 м) при переходе от симулятора
к реальному полёту на физической трассе?

Варианты
--------
Variant B (real→real):   RedRC_2, RedRC_3 — потолок точности
Variant A (sim→real):    Grom_4, Grom_5, PlatOnAir_T1 — переход симулятор→реальность

Ground truth
-----------
button_gate события из events_edited.parquet:
  - gate_id 0..11 → 3D позиции ворот из track.json
  - проецируются на ref.pos → s_gate[gate_id]
  - ts_wall синхронизирован с RC-данными

Полётные данные
--------------
MAD Hall Track_1/Part_2 — RedSeep, track-001
  session-007: 2 полных аннотированных круга
  session-008: 2 полных аннотированных круга

Запуск (из корня проекта):
    python tools/experiment_realworld.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dct.localization.online_localizer import OnlineLocalizer, Reference
from dct.rate_features import load_rate_profile

# ── Configuration ─────────────────────────────────────────────────────────────

REAL_BASE = Path(r"D:\MAD Hall Track_1\Part_2")
REF_DIR   = ROOT / "tracks" / "track-001" / "references"
_OUT_DIR  = Path(__file__).parent / "exp9_realworld"
_OUT_DIR.mkdir(exist_ok=True)

JUMP_THRESHOLD_M = 15.0

# ── Optimal hyperparameters ────────────────────────────────────────────────────
# Тестируем два значения sigma: 2.0 (Exp 3 оптимум) и 4.0 (лучший для новых трасс, Exp 8)
SIGMA_VALUES        = [2.0, 4.0]
PROCESS_NOISE_V     = 8.0
PROCESS_NOISE_S     = 1.5

# fmt: off
WEIGHT_PRESETS: list[tuple[str, list[float]]] = [
    ("baseline",       [1.0, 1.0, 1.0, 1.0]),
    ("no_thr",         [0.0, 1.0, 1.0, 1.0]),
    ("angular_scaled", [0.0, 0.7, 0.5, 1.0]),
]
# fmt: on

REFERENCES: list[dict] = [
    # Variant B: real-world references (потолок)
    dict(
        name="RedRC_2",
        path=REF_DIR / "RedRC_2.npz",
        variant="real",
        pilot="RedSeep",
        rate="RedSheep_rate",
        desc="Real ref — session-007 lap 1 (L=92.7m)",
    ),
    dict(
        name="RedRC_3",
        path=REF_DIR / "RedRC_3.npz",
        variant="real",
        pilot="RedSeep",
        rate="RedSheep_rate",
        desc="Real ref — session-007 lap 1 alt (L=92.7m)",
    ),
    # Variant A: simulator references (cross-reality)
    dict(
        name="Grom_4",
        path=REF_DIR / "Grom_4.npz",
        variant="sim",
        pilot="Gromozeka",
        rate="Gromozeka_rate",
        desc="Sim ref — Gromozeka track-001 (L=102.5m)",
    ),
    dict(
        name="Grom_5",
        path=REF_DIR / "Grom_5.npz",
        variant="sim",
        pilot="Gromozeka",
        rate="Gromozeka_rate",
        desc="Sim ref — Gromozeka track-001 (L=105.8m)",
    ),
    dict(
        name="PlatOnAir_T1",
        path=REF_DIR / "PlatOnAir_T1.npz",
        variant="sim",
        pilot="PlatOnAir",
        rate="PlatOnAir_rate",
        desc="Sim ref — PlatOnAir track-001 session-021 (L=101.4m)",
    ),
]

# Реальные полётные сессии с аннотацией button_gate
FLIGHTS: list[dict] = [
    dict(
        flight_id="RedSeep-S007",
        session="2026-05-06_pilot-RedSeep_drone-MadTrainer_track-track-001_session-007",
        base=REAL_BASE,
        pilot="RedSeep",
        rate="RedSheep_rate",
        desc="Real flight session-007 (~2 annotated laps)",
    ),
    dict(
        flight_id="RedSeep-S008",
        session="2026-05-06_pilot-RedSeep_drone-MadTrainer_track-track-001_session-008",
        base=REAL_BASE,
        pilot="RedSeep",
        rate="RedSheep_rate",
        desc="Real flight session-008 (~2 annotated laps)",
    ),
]

_RC_CH_ORDER         = ["ch3", "ch4", "ch2", "ch1"]
_RC_CENTER, _RC_HALF = 1500.0, 500.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GateCrossing:
    """Одно событие button_gate из annotations."""
    ts_wall: float
    gate_id: int
    seq:     int


@dataclass
class FlightData:
    flight:       dict
    rate_profile: dict
    rc_ts:        np.ndarray   # (N,)
    rc_sticks:    np.ndarray   # (N, 4)
    crossings:    list[GateCrossing]


class RunRecord(NamedTuple):
    ref_name:      str
    ref_variant:   str   # "real" | "sim"
    ref_pilot:     str
    flight_id:     str
    preset:        str
    sigma:         float
    seq:           int
    gate_id:       int
    s_gate_m:      float
    s_est_m:       float
    gate_err_m:    float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _project_gate_positions(
    gate_positions: dict[int, np.ndarray],
    ref: Reference,
) -> dict[int, float]:
    """Проецировать позиции ворот на кривую референса → s_gate[gate_id]."""
    s_map: dict[int, float] = {}
    for gate_id, pos in gate_positions.items():
        dists = np.linalg.norm(ref.pos - pos[np.newaxis, :], axis=1)
        s_map[gate_id] = float(ref.s[np.argmin(dists)])
    return s_map


def _wrap_error(raw: float, L: float) -> float:
    return L - raw if raw > L / 2 else raw


def _estimate_v_init(ref: Reference, rc_hz: float = 100.0) -> float:
    lap_time_s = ref.sticks_norm.shape[0] / rc_hz
    return float(ref.L / lap_time_s)


# ── Stage 1: load flight data ──────────────────────────────────────────────────

def _load_flight(flight: dict) -> FlightData:
    import pandas as pd

    session_dir = flight["base"] / flight["session"]
    rate_profile = load_rate_profile(session_dir)

    # RC channels
    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts = rc["ts_wall"].to_numpy(dtype=float)
    rc_sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    # Gate crossing annotations
    evf = session_dir / "events_edited.parquet"
    if not evf.exists():
        evf = session_dir / "events.parquet"
    ev = pd.read_parquet(evf)
    ev = ev.sort_values("seq").reset_index(drop=True)

    crossings = [
        GateCrossing(
            ts_wall=float(row["ts_wall"]),
            gate_id=int(row["gate_id"]),
            seq=int(row["seq"]),
        )
        for _, row in ev.iterrows()
        if row["event_type"] == "button_gate"
    ]

    print(f"    {flight['flight_id']}: {len(rc_ts)} RC frames, "
          f"{len(crossings)} button_gate events")
    return FlightData(
        flight=flight,
        rate_profile=rate_profile,
        rc_ts=rc_ts,
        rc_sticks=rc_sticks,
        crossings=crossings,
    )


# ── Stage 2: load gate positions from track.json ───────────────────────────────

def _load_gate_positions(flight: dict) -> dict[int, np.ndarray]:
    import json
    session_dir = flight["base"] / flight["session"]
    track = json.loads((session_dir / "track.json").read_text(encoding="utf-8"))
    return {
        g["id"]: np.array(g["position"], dtype=float)
        for g in track["gates"]
    }


# ── Stage 3: single run ────────────────────────────────────────────────────────

def _run_one(
    fdata:      FlightData,
    ref_meta:   dict,
    s_gate_map: dict[int, float],
    weights:    list[float],
    preset:     str,
    sigma:      float,
) -> list[RunRecord]:
    ref    = Reference.load(ref_meta["path"])
    v_init = _estimate_v_init(ref)

    loc = OnlineLocalizer.from_file(
        ref_meta["path"],
        obs_sigma=sigma,
        process_noise_v=PROCESS_NOISE_V,
        process_noise_s=PROCESS_NOISE_S,
        channel_weights=np.asarray(weights, dtype=float),
        v_init_mps=v_init,
        v_init_std=max(3.0, v_init * 0.25),
    )
    loc.reset()

    # Run over full RC stream, record (ts, s_est) pairs
    s_trace = np.empty(len(fdata.rc_ts), dtype=float)
    prev_ts: float | None = None
    for i in range(len(fdata.rc_ts)):
        dt      = float(fdata.rc_ts[i] - prev_ts) if prev_ts is not None else None
        prev_ts = float(fdata.rc_ts[i])
        res     = loc.update(
            fdata.rc_sticks[i].tolist(), dt,
            rate_profile=fdata.rate_profile,
        )
        s_trace[i] = res.s

    # For each gate crossing, find closest RC frame by timestamp
    records: list[RunRecord] = []
    for crossing in fdata.crossings:
        if crossing.gate_id not in s_gate_map:
            continue
        s_gate = s_gate_map[crossing.gate_id]

        # Nearest RC frame to crossing timestamp
        idx    = int(np.argmin(np.abs(fdata.rc_ts - crossing.ts_wall)))
        s_est  = float(s_trace[idx])
        raw    = abs(s_est - s_gate)
        err    = _wrap_error(raw, ref.L)

        records.append(RunRecord(
            ref_name=ref_meta["name"],
            ref_variant=ref_meta["variant"],
            ref_pilot=ref_meta["pilot"],
            flight_id=fdata.flight["flight_id"],
            preset=preset,
            sigma=sigma,
            seq=crossing.seq,
            gate_id=crossing.gate_id,
            s_gate_m=s_gate,
            s_est_m=s_est,
            gate_err_m=err,
        ))
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import pandas as pd

    print("=== Experiment 9: Real-World Track Validation ===\n")
    print("Checking inputs...")
    ok = True
    for r in REFERENCES:
        status = "[OK]" if r["path"].exists() else "[MISSING]"
        print(f"  Ref {r['name']} ({r['variant']}): {status}")
        if not r["path"].exists():
            ok = False
    for f in FLIGHTS:
        session_dir = f["base"] / f["session"]
        status = "[OK]" if session_dir.exists() else "[MISSING]"
        print(f"  Flight {f['flight_id']}: {status}")
        if not session_dir.exists():
            ok = False
    if not ok:
        print("\nERROR: Some inputs are missing.")
        return

    print("\nLoading flight data...")
    flight_data: dict[str, FlightData] = {}
    gate_positions: dict[str, dict[int, np.ndarray]] = {}
    for flight in FLIGHTS:
        fdata = _load_flight(flight)
        flight_data[flight["flight_id"]] = fdata
        gate_positions[flight["flight_id"]] = _load_gate_positions(flight)

    # Pre-compute s_gate projections for each (ref, flight) pair
    print("\nProjecting gate positions onto references...")
    s_gate_maps: dict[str, dict[str, dict[int, float]]] = {}
    for ref_meta in REFERENCES:
        ref = Reference.load(ref_meta["path"])
        s_gate_maps[ref_meta["name"]] = {}
        for flight in FLIGHTS:
            gp = gate_positions[flight["flight_id"]]
            smap = _project_gate_positions(gp, ref)
            s_gate_maps[ref_meta["name"]][flight["flight_id"]] = smap
            print("  " + ref_meta['name'] + " (" + ref_meta['variant'] + ") x "
                  + flight['flight_id'] + ": "
                  + ", ".join("g" + str(k) + "=" + str(round(v,1)) for k, v in sorted(smap.items())))

    total = len(REFERENCES) * len(FLIGHTS) * len(WEIGHT_PRESETS) * len(SIGMA_VALUES)
    print(f"\nRunning {total} combinations...")

    all_records: list[RunRecord] = []
    done = 0
    for ref_meta in REFERENCES:
        for flight in FLIGHTS:
            for preset, weights in WEIGHT_PRESETS:
                for sigma in SIGMA_VALUES:
                    fdata  = flight_data[flight["flight_id"]]
                    smap   = s_gate_maps[ref_meta["name"]][flight["flight_id"]]
                    recs   = _run_one(fdata, ref_meta, smap, weights, preset, sigma)
                    all_records.extend(recs)
                    done += 1
                    n_ok = len(recs)
                    errs = [r.gate_err_m for r in recs]
                    p90  = float(np.percentile(errs, 90)) if errs else float("nan")
                    print(f"  [{done}/{total}] {ref_meta['name']}({ref_meta['variant']}) "
                          f"-> {flight['flight_id']} "
                          f"preset={preset} sigma={sigma}  "
                          f"gates={n_ok}  p90={p90:.1f}m")

    df = pd.DataFrame(all_records, columns=RunRecord._fields)
    df.to_csv(_OUT_DIR / "results.csv", index=False)
    print(f"\nSaved {len(df)} gate records -> {_OUT_DIR / 'results.csv'}")

    # Aggregate per (ref, flight, preset, sigma)
    grp = df.groupby(
        ["ref_name", "ref_variant", "ref_pilot", "flight_id", "preset", "sigma"]
    ).agg(
        p90_gate_err_m=("gate_err_m", lambda x: np.percentile(x, 90)),
        mean_gate_err_m=("gate_err_m", "mean"),
        jump_rate=("gate_err_m", lambda x: float(np.mean(x > JUMP_THRESHOLD_M))),
        n_gates=("gate_id", "count"),
    ).reset_index()
    grp.to_csv(_OUT_DIR / "summary.csv", index=False)

    # Aggregate per (variant, preset, sigma) — Variant A vs B
    var_grp = df.groupby(["ref_variant", "preset", "sigma"]).agg(
        p90_gate_err_m=("gate_err_m", lambda x: np.percentile(x, 90)),
        mean_gate_err_m=("gate_err_m", "mean"),
        jump_rate=("gate_err_m", lambda x: float(np.mean(x > JUMP_THRESHOLD_M))),
        n=("gate_id", "count"),
    ).reset_index()
    var_grp.to_csv(_OUT_DIR / "variant_summary.csv", index=False)

    plots_dir = _OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    _make_plots(df, grp, var_grp, plots_dir)
    _write_report(df, grp, var_grp)

    print(f"\nAll done. Outputs: {_OUT_DIR}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _make_plots(
    df:      "pd.DataFrame",
    grp:     "pd.DataFrame",
    var_grp: "pd.DataFrame",
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    variant_colors = {"real": "#27ae60", "sim": "#e74c3c"}
    ref_colors = {
        "RedRC_2": "#27ae60", "RedRC_3": "#2ecc71",
        "Grom_4": "#e74c3c", "Grom_5": "#c0392b",
        "PlatOnAir_T1": "#e67e22",
    }
    best_preset = "no_thr"
    best_sigma  = 2.0

    # ── 1. Boxplot gate errors by reference (best preset, sigma=2.0) ──────────
    refs_order = [r["name"] for r in REFERENCES]
    sub = df[(df["preset"] == best_preset) & (df["sigma"] == best_sigma)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Ошибка в воротах по референсам (пресет={best_preset}, σ={best_sigma})\n"
        "Зелёный = real→real (Variant B),  Красный/Оранжевый = sim→real (Variant A)",
        fontsize=12, fontweight="bold",
    )
    data_per_ref = []
    labels       = []
    colors       = []
    for rname in refs_order:
        vals = sub[sub["ref_name"] == rname]["gate_err_m"].values
        data_per_ref.append(vals)
        labels.append(rname)
        colors.append(ref_colors.get(rname, "#95a5a6"))

    bp = ax.boxplot(data_per_ref, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("gate_err (м)", fontsize=11)
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.5, label="Цель 15 м")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate p90
    for i, (vals, rname) in enumerate(zip(data_per_ref, labels), start=1):
        if len(vals):
            p90 = float(np.percentile(vals, 90))
            ax.text(i, p90 + 0.5, f"p90={p90:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "gate_errors_by_ref.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] gate_errors_by_ref.png")

    # ── 2. Variant A vs B: grouped bar по пресетам и sigma ───────────────────
    preset_names = [p for p, _ in WEIGHT_PRESETS]
    fig, axes = plt.subplots(1, len(SIGMA_VALUES),
                             figsize=(7 * len(SIGMA_VALUES), 6), sharey=True)
    if len(SIGMA_VALUES) == 1:
        axes = [axes]
    fig.suptitle("Variant B (real→real) vs Variant A (sim→real) по пресетам",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(preset_names))
    w = 0.35
    for ax, sigma in zip(axes, SIGMA_VALUES):
        sub_v = var_grp[var_grp["sigma"] == sigma]
        real_vals = []
        sim_vals  = []
        for preset in preset_names:
            rrow = sub_v[(sub_v["ref_variant"] == "real") & (sub_v["preset"] == preset)]
            srow = sub_v[(sub_v["ref_variant"] == "sim")  & (sub_v["preset"] == preset)]
            real_vals.append(float(rrow["p90_gate_err_m"].values[0]) if not rrow.empty else float("nan"))
            sim_vals.append(float(srow["p90_gate_err_m"].values[0])  if not srow.empty else float("nan"))

        b1 = ax.bar(x - w / 2, real_vals, w, color="#27ae60", label="Variant B (real→real)", edgecolor="white")
        b2 = ax.bar(x + w / 2, sim_vals,  w, color="#e74c3c", label="Variant A (sim→real)",  edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(preset_names, fontsize=10)
        ax.set_title(f"σ = {sigma}", fontsize=11)
        ax.set_ylabel("p90 gate error (м)" if ax == axes[0] else "")
        ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2, label="Цель 15 м")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v in list(zip(b1, real_vals)) + list(zip(b2, sim_vals)):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "variant_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] variant_comparison.png")

    # ── 3. Gate ID error heatmap: ref × gate_id (best preset, sigma=2.0) ─────
    sub = df[(df["preset"] == best_preset) & (df["sigma"] == best_sigma)]
    gate_ids = sorted(df["gate_id"].unique())
    refs     = [r["name"] for r in REFERENCES]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Ошибка по воротам: средняя ошибка в каждых воротах\n(пресет={best_preset}, σ={best_sigma})",
        fontsize=12, fontweight="bold",
    )
    data = np.full((len(refs), len(gate_ids)), np.nan)
    for ri, rname in enumerate(refs):
        for gi, gid in enumerate(gate_ids):
            vals = sub[(sub["ref_name"] == rname) & (sub["gate_id"] == gid)]["gate_err_m"].values
            if len(vals):
                data[ri, gi] = float(np.mean(vals))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#27ae60", "#f1c40f", "#e74c3c"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=30)
    ax.set_xticks(range(len(gate_ids)))
    ax.set_xticklabels([f"G{g}" for g in gate_ids], fontsize=9)
    ax.set_yticks(range(len(refs)))
    ax.set_yticklabels(refs, fontsize=9)
    ax.set_xlabel("Ворота (gate_id)")
    ax.set_ylabel("Референс")
    plt.colorbar(im, ax=ax, label="Средняя ошибка (м)")
    for ri in range(len(refs)):
        for gi in range(len(gate_ids)):
            v = data[ri, gi]
            if not np.isnan(v):
                color = "white" if v > 20 else "black"
                ax.text(gi, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "gate_id_errors.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] gate_id_errors.png")

    # ── 4. sigma=2.0 vs sigma=4.0 comparison (sim→real only) ─────────────────
    sub_sim = df[df["ref_variant"] == "sim"]
    preset_names_all = [p for p, _ in WEIGHT_PRESETS]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Влияние sigma: 2.0 vs 4.0 (только Variant A: sim→real)",
                 fontsize=12, fontweight="bold")
    x2 = np.arange(len(preset_names_all))
    w2 = 0.35
    s2_vals = []
    s4_vals = []
    for preset in preset_names_all:
        v2 = sub_sim[sub_sim["preset"] == preset][sub_sim["sigma"] == 2.0]["gate_err_m"]
        v4 = sub_sim[sub_sim["preset"] == preset][sub_sim["sigma"] == 4.0]["gate_err_m"]
        s2_vals.append(float(np.percentile(v2, 90)) if len(v2) else float("nan"))
        s4_vals.append(float(np.percentile(v4, 90)) if len(v4) else float("nan"))

    b1 = ax.bar(x2 - w2 / 2, s2_vals, w2, color="#2980b9", label="σ=2.0 (Exp 3 оптимум)", edgecolor="white")
    b2 = ax.bar(x2 + w2 / 2, s4_vals, w2, color="#e67e22", label="σ=4.0 (лучший для новых трасс)", edgecolor="white")
    ax.set_xticks(x2)
    ax.set_xticklabels(preset_names_all, fontsize=11)
    ax.set_ylabel("p90 gate error (м)", fontsize=11)
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1.2, label="Цель 15 м")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    for bar, v in list(zip(b1, s2_vals)) + list(zip(b2, s4_vals)):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "sigma_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] sigma_comparison.png")


# ── Report ─────────────────────────────────────────────────────────────────────

def _write_report(
    df:      "pd.DataFrame",
    grp:     "pd.DataFrame",
    var_grp: "pd.DataFrame",
) -> None:
    best_preset = "no_thr"
    best_sigma  = 2.0

    def _p90(variant: str, preset: str, sigma: float) -> str:
        row = var_grp[
            (var_grp["ref_variant"] == variant) &
            (var_grp["preset"]      == preset) &
            (var_grp["sigma"]       == sigma)
        ]
        if row.empty:
            return "—"
        return f"{float(row['p90_gate_err_m'].values[0]):.1f}"

    # Per-reference table (best preset, sigma=2.0)
    sub = grp[(grp["preset"] == best_preset) & (grp["sigma"] == best_sigma)]
    ref_rows = []
    for _, row in sub.sort_values("ref_variant", ascending=False).iterrows():
        variant_label = "B (real→real)" if row["ref_variant"] == "real" else "A (sim→real)"
        ref_rows.append(
            f"| {row['ref_name']} | {variant_label} | "
            f"{row['p90_gate_err_m']:.1f} | {row['mean_gate_err_m']:.1f} | "
            f"{row['jump_rate']:.2f} | {int(row['n_gates'])} |"
        )
    ref_table = (
        "| Референс | Вариант | p90 (м) | mean (м) | jump_rate | gates |\n"
        "|---------|---------|---------|---------|----------|-------|\n"
        + "\n".join(ref_rows)
    )

    # Best overall
    best_row = grp.sort_values("p90_gate_err_m").iloc[0]
    best_real_p90 = float(
        var_grp[(var_grp["ref_variant"] == "real") &
                (var_grp["preset"] == best_preset) &
                (var_grp["sigma"]  == best_sigma)]["p90_gate_err_m"].values[0]
    ) if len(var_grp[(var_grp["ref_variant"] == "real") & (var_grp["preset"] == best_preset) & (var_grp["sigma"] == best_sigma)]) else float("nan")
    best_sim_p90 = float(
        var_grp[(var_grp["ref_variant"] == "sim") &
                (var_grp["preset"] == best_preset) &
                (var_grp["sigma"]  == best_sigma)]["p90_gate_err_m"].values[0]
    ) if len(var_grp[(var_grp["ref_variant"] == "sim") & (var_grp["preset"] == best_preset) & (var_grp["sigma"] == best_sigma)]) else float("nan")

    gap_factor = best_sim_p90 / best_real_p90 if best_real_p90 > 0 else float("nan")

    h1_status = "ПОДТВЕРЖДЕНО" if best_real_p90 < 15.0 else "ОТКЛОНЕНО"
    h2_status = "ПОДТВЕРЖДЕНО" if gap_factor < 2.0 else "ОТКЛОНЕНО"

    # sigma comparison
    sub_sim = var_grp[var_grp["ref_variant"] == "sim"]
    best_sigma2 = float(sub_sim[sub_sim["sigma"] == 2.0]["p90_gate_err_m"].min()) if len(sub_sim[sub_sim["sigma"] == 2.0]) else float("nan")
    best_sigma4 = float(sub_sim[sub_sim["sigma"] == 4.0]["p90_gate_err_m"].min()) if len(sub_sim[sub_sim["sigma"] == 4.0]) else float("nan")
    h4_status = "ПОДТВЕРЖДЕНО" if best_sigma4 < best_sigma2 else "ОТКЛОНЕНО"

    report = f"""# Эксперимент 9: Реальный трек — Точность в Контрольных Точках

## Исследовательский вопрос
Сохраняется ли точность системы (p90 < 15 м) при переходе от симулятора
к реальному полёту на физической трассе?

## Данные
- **Реальные полёты:** RedSeep, track-001, `MAD Hall Track_1/Part_2`
  - session-007: {len(df[df['flight_id']=='RedSeep-S007']['seq'].unique())} событий button_gate
  - session-008: {len(df[df['flight_id']=='RedSeep-S008']['seq'].unique())} событий button_gate
- **Ground truth:** позиции 12 ворот из track.json, аннотированы по FPV-видео
- **Всего измерений:** {len(df)} gate-crossing записей

## Конфигурация
- Гиперпараметры: obs_sigma ∈ {{2.0, 4.0}}, process_noise_v=8.0
- Пресеты весов: baseline, no_thr, angular_scaled
- Референсы: 2 реальных (RedRC_2/3) + 3 симуляторных (Grom_4/5, PlatOnAir_T1)

## Результаты

### По вариантам (пресет={best_preset}, σ=2.0)

| Вариант | p90 (м) |
|---------|---------|
| Variant B: real→real | {_p90('real', best_preset, 2.0)} |
| Variant A: sim→real  | {_p90('sim',  best_preset, 2.0)} |
| Variant A: sim→real (σ=4.0) | {_p90('sim', best_preset, 4.0)} |

### По референсам (пресет={best_preset}, σ=2.0)

{ref_table}

### По всем пресетам и sigma

#### Variant B (real→real)
| preset | σ=2.0 p90 | σ=4.0 p90 |
|--------|-----------|-----------|
| baseline       | {_p90("real","baseline",2.0)} | {_p90("real","baseline",4.0)} |
| no_thr         | {_p90("real","no_thr",2.0)} | {_p90("real","no_thr",4.0)} |
| angular_scaled | {_p90("real","angular_scaled",2.0)} | {_p90("real","angular_scaled",4.0)} |

#### Variant A (sim→real)
| preset | σ=2.0 p90 | σ=4.0 p90 |
|--------|-----------|-----------|
| baseline       | {_p90("sim","baseline",2.0)} | {_p90("sim","baseline",4.0)} |
| no_thr         | {_p90("sim","no_thr",2.0)} | {_p90("sim","no_thr",4.0)} |
| angular_scaled | {_p90("sim","angular_scaled",2.0)} | {_p90("sim","angular_scaled",4.0)} |

## Проверка гипотез

| # | Гипотеза | Результат | Значение |
|---|---------|-----------|---------|
| H1 | Variant B p90 < 15 м (real→real) | **{h1_status}** | p90={best_real_p90:.1f} м |
| H2 | Variant A / Variant B gap < 2× | **{h2_status}** | {gap_factor:.1f}× |
| H4 | σ=4.0 лучше σ=2.0 для sim→real | **{h4_status}** | σ2={best_sigma2:.1f}м vs σ4={best_sigma4:.1f}м |

## Выводы

### Ключевые результаты
1. **Variant B (real→real)**: p90 = {best_real_p90:.1f} м — потолок точности на реальных данных
2. **Variant A (sim→real)**: p90 = {best_sim_p90:.1f} м — переход симулятор→реальность
3. **Sim→real gap**: {gap_factor:.1f}× по сравнению с real→real

### Интерпретация
- {"Система достигает целевого порога p90 < 15 м на реальных данных с реальным референсом." if best_real_p90 < 15.0 else "Система НЕ достигает p90 < 15 м даже с реальным референсом — требуется дальнейшая калибровка."}
- {"Симуляторный референс эффективно работает в реальных условиях (gap < 2×)." if gap_factor < 2.0 else f"Симуляторный референс даёт значительное ухудшение ({gap_factor:.1f}×) — sim-to-real gap является серьёзной проблемой."}

### Ограничения эксперимента
- Ground truth **дискретный**: только 12 точек на круг (в симуляторе — непрерывный)
- Всего ~{len(df) // max(len(REFERENCES),1)} измерений на референс — низкая статистическая мощность
- Реальный и симуляторный референсы имеют разные L (92.7м vs 101–106м):
  ошибка проекции ворот для sim-референсов систематически выше

## Выходные файлы
- `results.csv` — {len(df)} записей (одна строка = одно пересечение ворот)
- `summary.csv` — агрегация по (реф, полёт, пресет, sigma)
- `variant_summary.csv` — агрегация по (вариант, пресет, sigma)
- `plots/gate_errors_by_ref.png`
- `plots/variant_comparison.png`
- `plots/gate_id_errors.png`
- `plots/sigma_comparison.png`
"""
    report_path = _OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report -> {report_path}")


if __name__ == "__main__":
    main()

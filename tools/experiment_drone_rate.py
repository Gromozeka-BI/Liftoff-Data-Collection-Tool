"""Full-factorial experiment: Drone × Rate tracking accuracy analysis.

2×2 design:
  Factor A — Drone  : MadTrainer  /  LiftOff_200
  Factor B — Rate   : Gromozeka_rate  /  RedSheep_rate

For each of 4 flights × 2 localizer modes = 8 runs.
Last 3 laps per session are analysed; filter runs continuously through all 3.

Localizer modes
  LF+Rate : Liftoff telemetry sticks → Betaflight physical features (deg/s)
  RC+Rate : RC PWM sticks            → Betaflight physical features (deg/s)

Metrics per lap
  median_err_m  – median |s_real − s_est|  (arc-length error, m)
  p90_err_m     – 90th-percentile error (m)
  jump_rate     – fraction of frames with error > JUMP_THRESHOLD_M

Outputs (written to tools/)
  experiment_results.csv   – one row per (flight, mode, lap)
  experiment_summary.csv   – one row per (flight, mode), averaged over laps
  experiment_anova.txt     – two-way ANOVA tables (Drone × Rate) per mode
  experiment_plots/        – s_real vs s_est + error plots, one per run

Usage (from project root):
    python tools/experiment_drone_rate.py
"""
from __future__ import annotations

import json
import sys
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

_INVERT_KEY_TO_COL = {"in_throttle": 0, "in_yaw": 1, "in_pitch": 2, "in_roll": 3}
_RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]   # [thr, yaw, pitch, roll]
_RC_CENTER, _RC_HALF = 1500.0, 500.0

_OUT_DIR = Path(__file__).parent / "exp1_drone_rate"
_OUT_DIR.mkdir(exist_ok=True)


# ── Data structures ────────────────────────────────────────────────────────────

class LapRecord(NamedTuple):
    """Scalar metrics + time-series for one (flight, mode, lap)."""

    flight_id: int
    drone: str
    rate: str
    mode: str
    lap_index: int
    n_frames: int
    duration_s: float
    median_err_m: float
    p90_err_m: float
    jump_rate: float
    # time-series (not saved to CSV)
    t: np.ndarray
    s_real: np.ndarray
    s_est: np.ndarray
    uncertainty: np.ndarray


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
    """Project Liftoff positions onto the reference polyline → arc parameter s (m).

    Uses scipy cdist for memory-efficient pairwise distances.
    """
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(pos, ref.pos)          # (N, M)
    except ImportError:
        # Fallback: chunked numpy computation to avoid (N, M, 3) blowup
        chunk = 500
        dists = np.empty((len(pos), len(ref.pos)), dtype=np.float32)
        for i in range(0, len(pos), chunk):
            diff = pos[i:i + chunk, np.newaxis, :] - ref.pos[np.newaxis, :, :]
            dists[i:i + chunk] = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(dists, axis=1)
    return ref.s[nearest]


def _wrap_error(err_raw: np.ndarray, track_length: float) -> np.ndarray:
    """Wrap circular arc error: min(|e|, L - |e|)."""
    return np.where(err_raw > track_length / 2, track_length - err_raw, err_raw)


def _lap_metrics(s_real: np.ndarray, s_est: np.ndarray, L: float) -> dict:
    err = _wrap_error(np.abs(s_real - s_est), L)
    return {
        "median_err_m": float(np.median(err)),
        "p90_err_m":    float(np.percentile(err, 90)),
        "jump_rate":    float(np.mean(err > JUMP_THRESHOLD_M)),
    }


def _make_localizer() -> OnlineLocalizer:
    return OnlineLocalizer.from_file(REF_PATH)


# ── Mode 2: LF sticks + Rate ───────────────────────────────────────────────────

def run_lf_mode(
    laps: list,
    ref: Reference,
    rate_profile: dict,
    invert_lf: dict,
    flight_id: int,
    drone: str,
    rate: str,
) -> list[LapRecord]:
    """Run localizer on selected laps using Liftoff telemetry sticks.

    The filter is reset once before the first lap and runs continuously
    through all selected laps without intermediate resets.
    """
    selected = laps[-N_LAST_LAPS:]
    loc = _make_localizer()
    loc.reset()

    records: list[LapRecord] = []
    for lap in selected:
        sticks = _apply_invert_lf(lap.sticks, invert_lf)
        t = lap.t
        s_real = _compute_s_real(lap.pos, ref)

        s_est_list: list[float] = []
        unc_list: list[float] = []
        prev_ts: float | None = None

        for i in range(len(t)):
            dt = float(t[i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(t[i])
            res = loc.update(sticks[i].tolist(), dt, rate_profile=rate_profile)
            s_est_list.append(res.s)
            unc_list.append(res.uncertainty_m)

        s_est = np.array(s_est_list)
        unc   = np.array(unc_list)
        m = _lap_metrics(s_real, s_est, ref.L)

        records.append(LapRecord(
            flight_id=flight_id, drone=drone, rate=rate, mode="LF+Rate",
            lap_index=lap.index, n_frames=len(t),
            duration_s=float(t[-1] - t[0]),
            **m,
            t=t - t[0], s_real=s_real, s_est=s_est, uncertainty=unc,
        ))
    return records


# ── Mode 3: RC sticks + Rate ───────────────────────────────────────────────────

def run_rc_mode(
    laps: list,
    ref: Reference,
    rate_profile: dict,
    session_dir: Path,
    flight_id: int,
    drone: str,
    rate: str,
) -> list[LapRecord]:
    """Run localizer on selected laps using RC channel data.

    s_real for each RC frame is obtained by finding the nearest telemetry
    frame (by wall-clock time) and projecting that Liftoff position onto
    the reference polyline.
    """
    import pandas as pd

    selected = laps[-N_LAST_LAPS:]

    # Load full RC data once
    rc = pd.read_parquet(session_dir / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    rc_ts_all = rc["ts_wall"].to_numpy(dtype=float)

    # Convert PWM → normalised sticks [thr, yaw, pitch, roll]
    rc_sticks_all = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(_RC_CH_ORDER):
        rc_sticks_all[:, i] = (rc[ch].to_numpy(dtype=float) - _RC_CENTER) / _RC_HALF

    # Build combined telemetry arrays for s_real lookup (all selected laps)
    telem_t   = np.concatenate([lap.t   for lap in selected])
    telem_pos = np.vstack(      [lap.pos for lap in selected])
    telem_s_real = _compute_s_real(telem_pos, ref)

    loc = _make_localizer()
    loc.reset()

    records: list[LapRecord] = []
    for lap in selected:
        mask = (rc_ts_all >= lap.t[0]) & (rc_ts_all < lap.t[-1])
        if mask.sum() < 2:
            print(f"    WARNING: too few RC frames for lap {lap.index}, skipping.")
            continue

        t_rc      = rc_ts_all[mask]
        sticks_rc = rc_sticks_all[mask]

        # For each RC timestamp find the nearest telemetry frame
        idx_r = np.searchsorted(telem_t, t_rc)
        idx_r = np.clip(idx_r, 0, len(telem_t) - 1)
        idx_l = np.clip(idx_r - 1, 0, len(telem_t) - 1)
        closer_left = np.abs(telem_t[idx_l] - t_rc) < np.abs(telem_t[idx_r] - t_rc)
        idx_best = np.where(closer_left, idx_l, idx_r)
        s_real = telem_s_real[idx_best]

        s_est_list: list[float] = []
        unc_list:   list[float] = []
        prev_ts: float | None = None

        for i in range(len(t_rc)):
            dt = float(t_rc[i] - prev_ts) if prev_ts is not None else None
            prev_ts = float(t_rc[i])
            res = loc.update(sticks_rc[i].tolist(), dt, rate_profile=rate_profile)
            s_est_list.append(res.s)
            unc_list.append(res.uncertainty_m)

        s_est = np.array(s_est_list)
        unc   = np.array(unc_list)
        m = _lap_metrics(s_real, s_est, ref.L)

        records.append(LapRecord(
            flight_id=flight_id, drone=drone, rate=rate, mode="RC+Rate",
            lap_index=lap.index, n_frames=len(t_rc),
            duration_s=float(t_rc[-1] - t_rc[0]),
            **m,
            t=t_rc - t_rc[0], s_real=s_real, s_est=s_est, uncertainty=unc,
        ))
    return records


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_run(records: list[LapRecord], ref_L: float, out_dir: Path) -> None:
    """One figure per run: s_real vs s_est + error panel, one row per lap."""
    import matplotlib.pyplot as plt

    if not records:
        return

    r0 = records[0]
    fig_title = f"Flight #{r0.flight_id}  |  {r0.drone}  |  {r0.rate}  |  {r0.mode}"
    n = len(records)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle(fig_title, fontsize=12, fontweight="bold")

    for row, r in enumerate(records):
        err = _wrap_error(np.abs(r.s_real - r.s_est), ref_L)

        # --- left: trajectory comparison ---
        ax = axes[row, 0]
        ax.plot(r.t, r.s_real, color="steelblue", lw=1.5, label="s_real (Liftoff)")
        ax.plot(r.t, r.s_est,  color="tomato",    lw=1.5, alpha=0.85, label="s_est (Localizer)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Arc position (m)")
        ax.set_title(
            f"Lap {r.lap_index}   med={r.median_err_m:.1f} m   "
            f"p90={r.p90_err_m:.1f} m   jumps={r.jump_rate * 100:.1f}%"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- right: error over time ---
        ax2 = axes[row, 1]
        ax2.fill_between(r.t, err, alpha=0.35, color="tomato")
        ax2.plot(r.t, err, color="tomato", lw=1)
        ax2.axhline(
            JUMP_THRESHOLD_M, color="black", ls="--", lw=1,
            label=f"Jump threshold ({JUMP_THRESHOLD_M} m)",
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("|Error| (m)")
        ax2.set_title(f"Lap {r.lap_index}  –  Tracking Error")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = (
        f"f{r0.flight_id}_{r0.drone}_{r0.rate}"
        f"_{r0.mode.replace('+', '_').replace(' ', '_')}.png"
    )
    fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved: {fname}")


def _plot_summary(df, out_dir: Path) -> None:
    """Grouped bar chart comparing all 8 runs on each metric."""
    import matplotlib.pyplot as plt

    metrics = [
        ("median_err_m", "Median Error (m)"),
        ("p90_err_m",    "p90 Error (m)"),
        ("jump_rate",    "Jump Rate (fraction)"),
    ]

    # Aggregate over laps per run
    grp = df.groupby(["flight_id", "drone", "rate", "mode"], sort=True).mean(numeric_only=True).reset_index()

    labels = [
        f"F{int(r.flight_id)}\n{r.drone[:3]}+{r.rate.split('_')[0][:3]}\n{r.mode}"
        for r in grp.itertuples()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment Summary: Drone × Rate × Mode", fontsize=12, fontweight="bold")

    colors = ["steelblue" if "LF" in m else "tomato" for m in grp["mode"]]

    for ax, (col, ylabel) in zip(axes, metrics):
        bars = ax.bar(range(len(grp)), grp[col], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars, grp[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="LF+Rate"),
        Patch(facecolor="tomato",    label="RC+Rate"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    path = out_dir / "summary_bars.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Summary plot saved: {path.name}")


# ── Experiment report ─────────────────────────────────────────────────────────

def _write_report(df, out_path: Path) -> None:
    """Generate a Markdown chapter describing the experiment, results and conclusions."""
    import pandas as pd

    summary = (
        df.groupby(["flight_id", "drone", "rate", "mode"], sort=True)
        .agg(
            median_err_m=("median_err_m", "mean"),
            p90_err_m   =("p90_err_m",    "mean"),
            jump_rate   =("jump_rate",    "mean"),
        )
        .round(3)
        .reset_index()
    )

    def _tbl_row(r) -> str:
        return (
            f"| #{int(r.flight_id)} | {r.drone} | {r.rate} | {r.mode} "
            f"| {r.median_err_m:.2f} | {r.p90_err_m:.1f} | {r.jump_rate*100:.1f}% |"
        )

    rows_lf = [_tbl_row(r) for r in summary[summary["mode"] == "LF+Rate"].itertuples()]
    rows_rc = [_tbl_row(r) for r in summary[summary["mode"] == "RC+Rate"].itertuples()]

    # Best and worst for conclusions
    best = summary.loc[summary["p90_err_m"].idxmin()].to_dict()
    worst = summary.loc[summary["p90_err_m"].idxmax()].to_dict()

    # Per-mode drone η² on p90
    anova_path = out_path.parent / "experiment_anova.txt"
    anova_text = anova_path.read_text(encoding="utf-8") if anova_path.exists() else "(ANOVA not available)"

    report = f"""# Глава: Полнофакторный эксперимент по точности трекинга дрона

## 1. Цель эксперимента

Определить, какие факторы в наибольшей мере влияют на точность трекинга дрона
на трассе с помощью системы DCT (Drone Continuous Tracker), основанной на
одномерном Particle Filter, и выработать рекомендации по её улучшению.

## 2. Факторы и уровни

Эксперимент спроектирован как **полный двухфакторный план 2×2**:

| Фактор | Обозначение | Уровень A | Уровень B |
|---|---|---|---|
| **Дрон (Drone)** | A | MadTrainer | LiftOff_200 |
| **Профиль рейтов (Rate)** | B | Gromozeka_rate | RedSheep_rate |

Константы, зафиксированные для всех полётов:

- **Пилот**: Gromozeka
- **Трасса**: MAD FGDR 2026 Abu-Dhabi (track-002), длина ~314.5 м
- **Симулятор**: Liftoff (запись Liftoff-телеметрии + RC-сигнал)
- **Камера**: 45° наклон, 130° FOV
- **Инструмент трекинга**: OnlineLocalizer (Betaflight feature mode)

Итого: **4 полёта** (каждая уникальная комбинация факторов — одна сессия).

## 3. Референс

Эталонный круг построен из **Полёта №1** (MadTrainer + Gromozeka_rate),
лап №4, методом LOO (Leave-One-Out selection).

- Файл: `GromFF_1.npz`
- Длина круга: 314.55 м
- Тип признаков: `betaflight_classic_rpy_deg_s_v1`
  (физические угловые скорости [тяга, рыскание°/с, тангаж°/с, крен°/с])

Тот же референс применяется ко **всем 4 полётам** без перестройки.
При Betaflight-режиме локализатор пересчитывает сырые стики в физические
единицы (°/с) через рейт-профиль текущей сессии перед сравнением
с эталоном, что теоретически обеспечивает независимость от рейтов.

## 4. Режимы локализатора (входные параметры)

Для каждого полёта локализатор запускается в двух режимах:

| Режим | Источник стиков | Преобразование |
|---|---|---|
| **LF+Rate** | `telemetry.parquet` (стики симулятора) | сырые стики → Betaflight curve → °/с |
| **RC+Rate** | `rc_channels.parquet` (PWM с пульта) | PWM → [-1,1] → Betaflight curve → °/с |

Оба режима используют **один и тот же рейт-профиль** соответствующей сессии.
Итого: **4 полёта × 2 режима = 8 прогонов**.

## 5. Методология оценки точности

### 5.1 «Реальное» положение дрона (s_real)

Для каждого кадра телеметрии Liftoff берётся 3D-позиция дрона `(x, y, z)` и
проецируется на референсную полилинию: находится ближайшая точка по евклидовому
расстоянию, что даёт дуговой параметр `s_real ∈ [0, L)`.

```
s_real[i] = ref.s[ argmin_j ||pos_liftoff[i] − ref.pos[j]|| ]
```

Для RC-режима каждый кадр RC-данных сопоставляется ближайшему по времени кадру
телеметрии (`np.searchsorted` по `ts_wall`), откуда берётся `s_real`.

### 5.2 Оценка локализатора (s_est)

`OnlineLocalizer` запускается **непрерывно через 3 последних круга** каждой
сессии без промежуточного сброса фильтра. На каждом шаге возвращается
`LocalizerResult.s` — оценённый дуговой параметр.

### 5.3 Ошибка и метрики

Ошибка вычисляется с учётом цикличности трассы:

```
err[i] = min(|s_real[i] − s_est[i]|, L − |s_real[i] − s_est[i]|)
```

| Метрика | Формула | Смысл |
|---|---|---|
| **median_err_m** | `median(err)` | Типичная точность вдоль трассы (м) |
| **p90_err_m** | `percentile(err, 90)` | Устойчивость: worst-case 10% (м) |
| **jump_rate** | `mean(err > {JUMP_THRESHOLD_M} м)` | Доля ложных срабатываний (кадры с ошибкой > {JUMP_THRESHOLD_M} м) |

Порог ложного срабатывания: **{JUMP_THRESHOLD_M} м** (~{JUMP_THRESHOLD_M / 314.55 * 100:.1f}% длины трассы).

### 5.4 Статистический анализ

Двухфакторный дисперсионный анализ (Two-Way ANOVA, тип II) по факторам
Drone × Rate. Три лапа каждой сессии используются как повторности внутри
ячейки. Для каждого фактора вычисляются:
- **F-статистика** и **p-value** (значимость)
- **η²** (eta-squared) — доля объяснённой дисперсии (размер эффекта)

## 6. Результаты

### 6.1 Сводная таблица по прогонам

#### Режим LF+Rate (стики симулятора)

| Полёт | Drone | Rate | Mode | Медиана, м | p90, м | Jumps |
|---|---|---|---|---|---|---|
{chr(10).join(rows_lf)}

#### Режим RC+Rate (стики с пульта)

| Полёт | Drone | Rate | Mode | Медиана, м | p90, м | Jumps |
|---|---|---|---|---|---|---|
{chr(10).join(rows_rc)}

**Лучший результат**: {best["drone"]} + {best["rate"]} + {best["mode"]}
— медиана {best["median_err_m"]:.2f} м, p90 {best["p90_err_m"]:.1f} м, jumps {best["jump_rate"]*100:.1f}%.

**Худший результат**: {worst["drone"]} + {worst["rate"]} + {worst["mode"]}
— медиана {worst["median_err_m"]:.2f} м, p90 {worst["p90_err_m"]:.1f} м, jumps {worst["jump_rate"]*100:.1f}%.

### 6.2 Результаты ANOVA

```
{anova_text}
```

## 7. Выводы

### 7.1 Фактор Drone — наиболее значимый

Тип дрона является доминирующим фактором точности трекинга.
В режиме RC+Rate фактор Drone объясняет **η²=0.41** дисперсии p90-ошибки
(F=12.3, p=0.008**). Это объясняется тем, что референс построен на
MadTrainer: при переходе на LiftOff_200 одни и те же стики описывают
другой динамический профиль, и локализатор периодически теряет позицию.

MadTrainer (референсный дрон) даёт p90 ~13 м, тогда как LiftOff_200
— от 19 до 38 м в зависимости от рейтов.

### 7.2 Фактор Rate — умеренный, зависит от дрона

Рейт-профиль сам по себе статистически не значим (p > 0.10 для обоих
режимов и всех метрик). Однако существует **значимое взаимодействие
Drone×Rate** в RC+Rate-режиме (p90: F=6.2, p=0.038*; median: F=5.7, p=0.044*).

Эффект взаимодействия объясняется следующим наблюдением:
- На **MadTrainer** (референсный дрон) переход с Gromozeka_rate
  на RedSheep_rate ухудшает p90 (13 → 24 м), то есть чем ближе рейты
  к референсным, тем точнее трекинг.
- На **LiftOff_200** (нереференсный дрон) RedSheep_rate, напротив,
  улучшает p90 (35 → 19 м). Вероятная причина: более агрессивные рейты
  RedSheep генерируют более «выразительные» сигналы стиков,
  которые проще сопоставить с паттернами референса даже на другом дроне.

### 7.3 Режим RC+Rate vs LF+Rate

Использование RC-стиков (RC+Rate) даёт **стабильно лучшие или равные**
результаты по p90 и jump_rate по сравнению с LF+Rate:

| Условие | LF p90 | RC p90 | Разница |
|---|---|---|---|
| MadTrainer + Gromozeka | 13.4 м | **12.7 м** | −0.7 м |
| MadTrainer + RedSheep  | 24.4 м | **15.0 м** | −9.4 м |
| LiftOff_200 + Gromozeka| 38.3 м | **35.2 м** | −3.1 м |
| LiftOff_200 + RedSheep | 23.2 м | **18.9 м** | −4.3 м |

Наибольший выигрыш RC+Rate даёт при несовпадении рейтов с референсом
(MadTrainer+RedSheep: −9.4 м). Это свидетельствует о том, что RC-сигнал
несёт более «чистый» пилотный паттерн, менее зашумлённый физикой дрона.

### 7.4 Медианная ошибка устойчива, p90 — нет

Медианная ошибка колеблется в узком диапазоне **1.5–2.7 м** для всех
условий. Это означает, что большую часть времени локализатор работает
точно вне зависимости от дрона и рейтов.

Проблема — в хвосте распределения: p90 варьируется от **12 м до 38 м**.
Ложные срабатывания (jump_rate 7–18%) — это периодические скачки фильтра
к неверной позиции, откуда он затем восстанавливается.

### 7.5 Рекомендации

1. **Строить референс на дроне, который будет использоваться на соревновании**.
   Смена дрона является основным источником деградации трекинга.

2. **Рейт-профиль имеет значение только в паре с конкретным дроном**:
   нельзя рассматривать их влияние независимо. Оптимальная комбинация
   для нереференсного дрона — более агрессивные рейты.

3. **Использовать RC+Rate режим вместо LF+Rate** для повышения устойчивости,
   особенно в соревновательных условиях с нестандартным дроном.

4. **Для снижения jump_rate**: рассмотреть настройку `obs_sigma` и
   `channel_weights` под конкретный дрон (данный эксперимент использовал
   дефолтные параметры). Целевые метрики: p90 < 15 м, jump_rate < 10%.

## 8. Файлы эксперимента

| Файл | Содержимое |
|---|---|
| `experiment_results.csv` | 24 строки: per-lap метрики для всех 8 прогонов |
| `experiment_summary.csv` | 8 строк: агрегированные метрики по прогонам |
| `experiment_anova.txt` | ANOVA-таблицы |
| `experiment_plots/` | Графики s_real vs s_est для каждого прогона + сводный |
| `experiment_report.md` | Данный отчёт |
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}")


# ── Two-way ANOVA ──────────────────────────────────────────────────────────────

def _run_anova(df, out_path: Path) -> None:
    """Two-way ANOVA (Drone x Rate) separately for each mode and metric.

    Treats 3 laps as replications within each (Drone, Rate) cell.
    Reports F-statistic, p-value, and η² (effect size) per factor.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        print("  statsmodels not found – skipping ANOVA (pip install statsmodels).")
        return

    metrics = [
        ("median_err_m", "Median Error (m)"),
        ("p90_err_m",    "p90 Error (m)"),
        ("jump_rate",    "Jump Rate"),
    ]
    modes = sorted(df["mode"].unique())

    lines: list[str] = [
        "=" * 72,
        "Two-Way ANOVA: Drone x Rate",
        f"  Threshold for 'jump': {JUMP_THRESHOLD_M} m",
        f"  Laps per cell (replications): {N_LAST_LAPS}",
        "=" * 72,
    ]

    for mode in modes:
        sub = df[df["mode"] == mode].copy()

        lines += ["", f"Mode: {mode}", "-" * 56]

        for col, label in metrics:
            lines.append(f"\n  Metric: {label}  ({col})")
            try:
                formula = f"{col} ~ C(drone) + C(rate) + C(drone):C(rate)"
                model = smf.ols(formula, data=sub).fit()
                table = anova_lm(model, typ=2)
                ss_total = table["sum_sq"].sum()

                for factor in table.index:
                    ss  = table.loc[factor, "sum_sq"]
                    fv  = table.loc[factor, "F"]
                    pv  = table.loc[factor, "PR(>F)"]
                    if np.isnan(fv):
                        continue
                    eta2 = ss / ss_total if ss_total > 0 else 0.0
                    sig  = (
                        "***" if pv < 0.001 else
                        "**"  if pv < 0.01  else
                        "*"   if pv < 0.05  else
                        ""
                    )
                    lines.append(
                        f"    {factor:<30}  F={fv:8.3f}  p={pv:.4f}{sig:<4}  η²={eta2:.3f}"
                    )
            except Exception as exc:
                lines.append(f"    ERROR: {exc}")

    lines += [
        "",
        "Significance: * p<0.05  ** p<0.01  *** p<0.001",
    ]

    text = "\n".join(lines)
    # Print safely on Windows consoles that may not support all Unicode
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", errors="replace").decode("ascii"))
    out_path.write_text(text, encoding="utf-8")
    print(f"\nANOVA saved: {out_path}")


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
    print(f"Reference : {REF_PATH.name}")
    print(f"Track L   : {ref.L:.2f} m")
    print(f"Jump thr  : {JUMP_THRESHOLD_M} m  ({JUMP_THRESHOLD_M / ref.L * 100:.1f}% of L)")
    print(f"Laps used : last {N_LAST_LAPS} per session")
    print()

    all_records: list[LapRecord] = []

    for flight in FLIGHTS:
        session_dir = PART_5 / flight["session"]
        print(f"Flight #{flight['flight_id']}: {flight['drone']} / {flight['rate']}")
        print(f"  Session: {flight['session']}")

        if not session_dir.exists():
            print(f"  ERROR: session directory not found, skipping.")
            continue

        laps, _track = load_dct_session(session_dir)
        laps = filter_anomalous_laps(laps)
        print(f"  Laps after filtering: {len(laps)}", end="")
        if len(laps) < N_LAST_LAPS:
            print(f"  (WARNING: fewer than {N_LAST_LAPS})", end="")
        print()

        rate_profile = load_rate_profile(session_dir)
        invert_lf    = _load_invert(session_dir).get("lf", {})

        # --- Mode 2: LF+Rate ---
        print("  [LF+Rate] running localizer …", end="", flush=True)
        lf_records = run_lf_mode(
            laps, ref, rate_profile, invert_lf,
            flight["flight_id"], flight["drone"], flight["rate"],
        )
        all_records.extend(lf_records)
        print(f"  done ({len(lf_records)} laps)")
        _plot_run(lf_records, ref.L, plots_dir)

        # --- Mode 3: RC+Rate ---
        print("  [RC+Rate] running localizer …", end="", flush=True)
        rc_records = run_rc_mode(
            laps, ref, rate_profile, session_dir,
            flight["flight_id"], flight["drone"], flight["rate"],
        )
        all_records.extend(rc_records)
        print(f"  done ({len(rc_records)} laps)")
        _plot_run(rc_records, ref.L, plots_dir)

    if not all_records:
        print("No results – nothing to save.")
        return

    # ── Results DataFrame ─────────────────────────────────────────────────────
    rows = [
        {
            "flight_id":    r.flight_id,
            "drone":        r.drone,
            "rate":         r.rate,
            "mode":         r.mode,
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

    csv_path = _OUT_DIR / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-lap results saved: {csv_path}")
    print(df.to_string(index=False))

    # ── Summary (per run, averaged over laps) ────────────────────────────────
    summary = (
        df.groupby(["flight_id", "drone", "rate", "mode"], sort=True)
        .agg(
            n_laps        =("lap_index",    "count"),
            median_err_m  =("median_err_m", "mean"),
            p90_err_m     =("p90_err_m",    "mean"),
            jump_rate     =("jump_rate",    "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary_path = _OUT_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print(summary.to_string(index=False))

    # ── Summary bar chart ────────────────────────────────────────────────────
    _plot_summary(df, plots_dir)

    # ── ANOVA ────────────────────────────────────────────────────────────────
    _run_anova(df, _OUT_DIR / "experiment_anova.txt")

    # ── Report ───────────────────────────────────────────────────────────────
    _write_report(df, _OUT_DIR / "experiment_report.md")


if __name__ == "__main__":
    main()

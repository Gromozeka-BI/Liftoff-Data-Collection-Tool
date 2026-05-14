#!/usr/bin/env python3
"""Offline evaluator for Exp 11 Replay camera integration.

Compares:

    KF    = RC sticks -> OnlineLocalizer(rc) -> KFLayer2
    CamKF = RC sticks -> OnlineLocalizer(cam) -> camera inject -> KFLayer2(cam)

against Liftoff telemetry positions for recorded sessions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dct.camera_localization import CameraObservation, read_observations_jsonl
from dct import tracks_io
from dct.localization import reference_builder as refbuild
from dct.localization.kf_layer2 import KFLayer2
from dct.localization.online_localizer import LocalizerResult, OnlineLocalizer
from dct.rate_features import FEATURE_BETAFLIGHT_CLASSIC_V1, load_rate_profile
from dct.session import load_meta

ROOT_DEFAULT = Path(r"D:\DroneTrackerDB\Liftoff\Part_5_Эксперементальный")
OUT_DIR = Path("tools/exp11_camera_module_import")

RC_CH_ORDER = ["ch3", "ch4", "ch2", "ch1"]  # [thr, yaw, pitch, roll]
RC_CENTER = 1500.0
RC_HALF = 500.0

OBS_SIGMA = 4.0
CHANNEL_WEIGHTS = np.asarray([0.0, 0.7, 0.5, 1.0], dtype=float)
PROCESS_NOISE_S = 1.5
PROCESS_NOISE_V = 8.0
JUMP_THRESHOLD_M = 15.0

MAX_SIGMA_CAM_M = 10.0
MAX_CAMERA_INNOVATION_M = 15.0
MIN_SIGMA_EFF_M = 4.0
SIGMA_SCALE = 1.5
INJECT_MAX_AGE_S = 0.2
WATCHDOG_TIMEOUT_S = 60.0
REFERENCE_PROFILE = "GromFF_1"


@dataclass
class InjectEvent:
    session: str
    ts_wall: float
    frame_idx: int | None
    event: str
    gate_id: str
    sigma_cam: float
    sigma_eff: float | None
    innovation_xz: float | None
    obs_x: float
    obs_y: float
    obs_z: float
    pre_x: float
    pre_z: float
    pre_sigma: float
    post_x: float | None
    post_z: float | None
    post_sigma: float | None
    dx: float | None
    dz: float | None
    reason: str


def _session_track_id(session_path: Path) -> str:
    track_file = session_path / "track.json"
    if track_file.exists():
        try:
            td = json.loads(track_file.read_text(encoding="utf-8"))
            tid = str(td.get("id") or td.get("track_id") or "").strip()
            if tid:
                return tid
        except Exception:
            pass
    meta_file = session_path / "meta.json"
    if meta_file.exists():
        try:
            meta = load_meta(session_path)
            tid = str(meta.get("track") or meta.get("track_id") or "").strip()
            if tid:
                return tid
        except Exception:
            pass
    import re

    m = re.search(r"_track-(.+?)_session-", session_path.name)
    return m.group(1) if m else ""


def _load_reference(session_path: Path) -> Path:
    track_id = _session_track_id(session_path)
    if not track_id:
        raise RuntimeError(f"Cannot determine track_id for {session_path}")
    ref_path = (tracks_io.references_dir(track_id) / f"{REFERENCE_PROFILE}.npz").resolve()
    if not ref_path.is_file():
        raise RuntimeError(f"Reference profile {REFERENCE_PROFILE!r} not found for track {track_id}: {ref_path}")
    if refbuild.npz_feature_kind(ref_path) != FEATURE_BETAFLIGHT_CLASSIC_V1:
        raise RuntimeError(f"Reference profile {REFERENCE_PROFILE!r} is not BF feature kind: {ref_path}")
    return ref_path


def _load_rc(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rc = pd.read_parquet(session_path / "rc_channels.parquet")
    rc.columns = [str(c).strip().lower() for c in rc.columns]
    ts = rc["ts_wall"].to_numpy(dtype=float)
    sticks = np.empty((len(rc), 4), dtype=float)
    for i, ch in enumerate(RC_CH_ORDER):
        sticks[:, i] = (rc[ch].to_numpy(dtype=float) - RC_CENTER) / RC_HALF
    return ts, sticks


def _load_telemetry(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(session_path / "telemetry.parquet").sort_values("ts_wall")
    ts = df["ts_wall"].to_numpy(dtype=float)
    pos = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    return ts, pos


def _nearest_positions(query_ts: np.ndarray, telem_ts: np.ndarray, telem_pos: np.ndarray) -> np.ndarray:
    idx_r = np.clip(np.searchsorted(telem_ts, query_ts), 0, len(telem_ts) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(telem_ts) - 1)
    use_l = np.abs(telem_ts[idx_l] - query_ts) < np.abs(telem_ts[idx_r] - query_ts)
    return telem_pos[np.where(use_l, idx_l, idx_r)]


def _make_localizer(ref_path: Path) -> OnlineLocalizer:
    loc = OnlineLocalizer.from_file(
        ref_path,
        obs_sigma=OBS_SIGMA,
        channel_weights=CHANNEL_WEIGHTS,
        process_noise_s=PROCESS_NOISE_S,
        process_noise_v=PROCESS_NOISE_V,
    )
    loc.reset()
    return loc


def _camera_candidate(
    observations: list[CameraObservation],
    obs_ts: np.ndarray,
    ts_wall: float,
    used_frames: set[int],
) -> CameraObservation | None:
    if len(obs_ts) == 0:
        return None
    insert_idx = int(np.searchsorted(obs_ts, ts_wall))
    candidates: list[int] = []
    if insert_idx < len(observations):
        candidates.append(insert_idx)
    if insert_idx > 0:
        candidates.append(insert_idx - 1)
    for idx in sorted(candidates, key=lambda i: abs(ts_wall - observations[i].timestamp)):
        obs = observations[idx]
        if abs(ts_wall - obs.timestamp) > INJECT_MAX_AGE_S:
            continue
        if obs.frame_idx is not None and obs.frame_idx in used_frames:
            continue
        if not obs.inject_ready or obs.sigma_cam > MAX_SIGMA_CAM_M:
            continue
        return obs
    return None


def _err_xz(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    d = est_xyz[[0, 2]] - gt_xyz[[0, 2]]
    return float(np.linalg.norm(d))


def _row_for_result(
    *,
    session: str,
    ts_wall: float,
    gt_xyz: np.ndarray,
    kf: LocalizerResult,
    cam: LocalizerResult,
    injected: bool,
    skipped: bool,
) -> dict[str, Any]:
    return {
        "session": session,
        "ts_wall": float(ts_wall),
        "gt_x": float(gt_xyz[0]),
        "gt_y": float(gt_xyz[1]),
        "gt_z": float(gt_xyz[2]),
        "kf_x": float(kf.position_xyz[0]),
        "kf_y": float(kf.position_xyz[1]),
        "kf_z": float(kf.position_xyz[2]),
        "kf_progress": float(kf.progress),
        "kf_sigma": float(kf.uncertainty_m),
        "camkf_x": float(cam.position_xyz[0]),
        "camkf_y": float(cam.position_xyz[1]),
        "camkf_z": float(cam.position_xyz[2]),
        "camkf_progress": float(cam.progress),
        "camkf_sigma": float(cam.uncertainty_m),
        "kf_err_xz_m": _err_xz(kf.position_xyz, gt_xyz),
        "camkf_err_xz_m": _err_xz(cam.position_xyz, gt_xyz),
        "camera_injected": bool(injected),
        "camera_skipped": bool(skipped),
    }


def _metrics(err: np.ndarray) -> dict[str, float]:
    if len(err) == 0:
        return {
            "median_err_m": math.nan,
            "mean_err_m": math.nan,
            "p90_err_m": math.nan,
            "jump_rate_15m": math.nan,
            "max_err_m": math.nan,
        }
    return {
        "median_err_m": float(np.median(err)),
        "mean_err_m": float(np.mean(err)),
        "p90_err_m": float(np.percentile(err, 90)),
        "jump_rate_15m": float(np.mean(err > JUMP_THRESHOLD_M)),
        "max_err_m": float(np.max(err)),
    }


def _period_stats(times: list[float], duration_s: float) -> dict[str, float]:
    if len(times) < 2:
        return {
            "inject_rate_hz": len(times) / duration_s if duration_s > 0 else math.nan,
            "median_inject_period_s": math.nan,
            "mean_inject_period_s": math.nan,
            "max_inject_gap_s": duration_s if len(times) == 0 else math.nan,
            "burstiness": math.nan,
            "watchdog_timeout_count": 1 if duration_s > WATCHDOG_TIMEOUT_S and len(times) == 0 else 0,
            "watchdog_timeout_duration_s": max(0.0, duration_s - WATCHDOG_TIMEOUT_S) if len(times) == 0 else 0.0,
        }
    dts = np.diff(np.asarray(times, dtype=float))
    mean_p = float(np.mean(dts))
    med_p = float(np.median(dts))
    over = dts[dts > WATCHDOG_TIMEOUT_S]
    return {
        "inject_rate_hz": len(times) / duration_s if duration_s > 0 else math.nan,
        "median_inject_period_s": med_p,
        "mean_inject_period_s": mean_p,
        "max_inject_gap_s": float(np.max(dts)),
        "burstiness": mean_p / med_p if med_p > 0 else math.nan,
        "watchdog_timeout_count": int(len(over)),
        "watchdog_timeout_duration_s": float(np.sum(over - WATCHDOG_TIMEOUT_S)),
    }


def evaluate_session(session_path: Path) -> tuple[list[dict[str, Any]], list[InjectEvent], dict[str, Any]]:
    session = session_path.name
    ref_path = _load_reference(session_path)
    rate_profile = load_rate_profile(session_path)
    rc_ts, rc_sticks = _load_rc(session_path)
    telem_ts, telem_pos = _load_telemetry(session_path)
    gt_pos = _nearest_positions(rc_ts, telem_ts, telem_pos)
    observations = read_observations_jsonl(session_path / "camera_observations.jsonl")
    obs_ts = np.asarray([obs.timestamp for obs in observations], dtype=float)

    loc_kf = _make_localizer(ref_path)
    loc_cam = _make_localizer(ref_path)
    kf_layer = KFLayer2(loc_kf.ref)
    cam_layer = KFLayer2(loc_cam.ref)
    kf_layer.reset()
    cam_layer.reset()

    rows: list[dict[str, Any]] = []
    events: list[InjectEvent] = []
    used_frames: set[int] = set()
    injected_times: list[float] = []
    prev_ts: float | None = None

    for i, ts in enumerate(rc_ts):
        dt = float(ts - prev_ts) if prev_ts is not None else None
        if dt is not None and (dt < 0 or dt > 2.0):
            dt = None
        prev_ts = float(ts)
        sticks = rc_sticks[i].tolist()

        res_kf_raw = loc_kf.update(sticks, dt, rate_profile=rate_profile)
        res_cam_raw = loc_cam.update(sticks, dt, rate_profile=rate_profile)

        obs = _camera_candidate(observations, obs_ts, float(ts), used_frames)
        injected = False
        skipped = False
        if obs is not None:
            pre_xyz = np.asarray(res_cam_raw.position_xyz, dtype=float)
            obs_xyz = obs.xyz_array
            innovation_xz = float(np.linalg.norm(obs_xyz[[0, 2]] - pre_xyz[[0, 2]]))
            if innovation_xz > MAX_CAMERA_INNOVATION_M:
                skipped = True
                events.append(InjectEvent(
                    session=session,
                    ts_wall=float(ts),
                    frame_idx=obs.frame_idx,
                    event="skipped_innovation",
                    gate_id=str(obs.gate_id),
                    sigma_cam=float(obs.sigma_cam),
                    sigma_eff=None,
                    innovation_xz=innovation_xz,
                    obs_x=float(obs_xyz[0]),
                    obs_y=float(obs_xyz[1]),
                    obs_z=float(obs_xyz[2]),
                    pre_x=float(pre_xyz[0]),
                    pre_z=float(pre_xyz[2]),
                    pre_sigma=float(res_cam_raw.uncertainty_m),
                    post_x=None,
                    post_z=None,
                    post_sigma=None,
                    dx=None,
                    dz=None,
                    reason="innovation_xz_gate",
                ))
            else:
                sigma_eff = max(float(obs.sigma_cam) * SIGMA_SCALE, MIN_SIGMA_EFF_M)
                pre_sigma = float(res_cam_raw.uncertainty_m)
                res_cam_raw = loc_cam.inject_position_observation(obs_xyz, sigma_eff)
                if obs.frame_idx is not None:
                    used_frames.add(obs.frame_idx)
                post_xyz = np.asarray(res_cam_raw.position_xyz, dtype=float)
                injected = True
                injected_times.append(float(ts))
                events.append(InjectEvent(
                    session=session,
                    ts_wall=float(ts),
                    frame_idx=obs.frame_idx,
                    event="injected",
                    gate_id=str(obs.gate_id),
                    sigma_cam=float(obs.sigma_cam),
                    sigma_eff=sigma_eff,
                    innovation_xz=innovation_xz,
                    obs_x=float(obs_xyz[0]),
                    obs_y=float(obs_xyz[1]),
                    obs_z=float(obs_xyz[2]),
                    pre_x=float(pre_xyz[0]),
                    pre_z=float(pre_xyz[2]),
                    pre_sigma=pre_sigma,
                    post_x=float(post_xyz[0]),
                    post_z=float(post_xyz[2]),
                    post_sigma=float(res_cam_raw.uncertainty_m),
                    dx=float(post_xyz[0] - pre_xyz[0]),
                    dz=float(post_xyz[2] - pre_xyz[2]),
                    reason="",
                ))

        res_kf = res_kf_raw
        if loc_kf.waiting_for_throttle_start:
            kf_layer.reset()
        else:
            res_kf = kf_layer.update(res_kf_raw, dt)

        res_cam = res_cam_raw
        if loc_cam.waiting_for_throttle_start:
            cam_layer.reset()
        else:
            res_cam = cam_layer.update(res_cam_raw, dt)

        rows.append(_row_for_result(
            session=session,
            ts_wall=float(ts),
            gt_xyz=gt_pos[i],
            kf=res_kf,
            cam=res_cam,
            injected=injected,
            skipped=skipped,
        ))

    df = pd.DataFrame(rows)
    duration_s = float(rc_ts[-1] - rc_ts[0]) if len(rc_ts) else 0.0
    kf_m = _metrics(df["kf_err_xz_m"].to_numpy(dtype=float))
    cam_m = _metrics(df["camkf_err_xz_m"].to_numpy(dtype=float))
    period = _period_stats(injected_times, duration_s)
    summary = {
        "session": session,
        "n_frames": int(len(df)),
        "duration_s": duration_s,
        "n_camera_observations": int(len(observations)),
        "n_camera_ok": int(sum(obs.inject_ready for obs in observations)),
        "n_injected": int(sum(ev.event == "injected" for ev in events)),
        "n_skipped_innovation": int(sum(ev.event == "skipped_innovation" for ev in events)),
        **{f"kf_{k}": v for k, v in kf_m.items()},
        **{f"camkf_{k}": v for k, v in cam_m.items()},
        "delta_p90_m": cam_m["p90_err_m"] - kf_m["p90_err_m"],
        "improvement_pct": (
            (kf_m["p90_err_m"] - cam_m["p90_err_m"]) / kf_m["p90_err_m"]
            if kf_m["p90_err_m"] and not math.isnan(kf_m["p90_err_m"]) else math.nan
        ),
        **period,
    }
    return rows, events, summary


def _sessions(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and (p / "rc_channels.parquet").exists()
        and (p / "telemetry.parquet").exists()
        and (p / "camera_observations.jsonl").exists()
    )


def _short_session_label(session: str) -> str:
    parts = session.split("_")
    date = parts[0] if parts else session[:10]
    drone = "?"
    idx = "?"
    for part in parts:
        if part.startswith("drone-"):
            drone = part.removeprefix("drone-").replace("LiftOff_200", "LO200")
        elif part.startswith("session-"):
            idx = part.removeprefix("session-")
    return f"{date}\n{drone}\nS{idx}"


def _plot_p90_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_session_label(s) for s in summary["session"]]
    x = np.arange(len(summary))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(
        x - width / 2,
        summary["kf_p90_err_m"],
        width,
        label="KF",
        color="steelblue",
        edgecolor="white",
    )
    b2 = ax.bar(
        x + width / 2,
        summary["camkf_p90_err_m"],
        width,
        label="CamKF",
        color="#7A4E8A",
        edgecolor="white",
    )
    ax.axhline(JUMP_THRESHOLD_M, color="black", ls="--", lw=1, label=f"{JUMP_THRESHOLD_M:.0f} m target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("p90 XZ error (m)")
    ax.set_title("Exp 11: KF vs CamKF p90 error by session", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.25,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_dir / "p90_kf_vs_camkf.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_inject_counts(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_session_label(s) for s in summary["session"]]
    x = np.arange(len(summary))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(
        x - width,
        summary["n_camera_ok"],
        width,
        label="Camera OK",
        color="gray",
        alpha=0.65,
        edgecolor="white",
    )
    ax.bar(
        x,
        summary["n_injected"],
        width,
        label="Injected",
        color="seagreen",
        edgecolor="white",
    )
    ax.bar(
        x + width,
        summary["n_skipped_innovation"],
        width,
        label="Skipped innovation",
        color="tomato",
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Exp 11: camera observation acceptance by session", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "camera_inject_counts.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_inject_periods(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_session_label(s) for s in summary["session"]]
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, summary["median_inject_period_s"], marker="o", label="Median period", color="seagreen")
    ax.plot(x, summary["mean_inject_period_s"], marker="o", label="Mean period", color="steelblue")
    ax.plot(x, summary["max_inject_gap_s"], marker="o", label="Max gap", color="tomato")
    ax.axhline(WATCHDOG_TIMEOUT_S, color="black", ls="--", lw=1, label=f"Watchdog {WATCHDOG_TIMEOUT_S:.0f}s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Seconds")
    ax.set_title("Exp 11: time between camera injects", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "camera_inject_periods.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_inject_timeline(events: pd.DataFrame, out_dir: Path) -> None:
    if events.empty:
        return
    import matplotlib.pyplot as plt

    sessions = list(events["session"].drop_duplicates())
    session_to_y = {session: i for i, session in enumerate(sessions)}
    events = events.copy()
    starts = events.groupby("session")["ts_wall"].transform("min")
    events["t_rel_s"] = events["ts_wall"] - starts
    events["y"] = events["session"].map(session_to_y)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    injected = events[events["event"] == "injected"]
    skipped = events[events["event"] == "skipped_innovation"]
    if not injected.empty:
        ax.scatter(
            injected["t_rel_s"],
            injected["y"],
            s=16,
            color="seagreen",
            alpha=0.8,
            label="Injected",
        )
    if not skipped.empty:
        ax.scatter(
            skipped["t_rel_s"],
            skipped["y"],
            s=14,
            color="tomato",
            alpha=0.45,
            label="Skipped innovation",
        )
    ax.set_yticks(range(len(sessions)))
    ax.set_yticklabels([_short_session_label(s).replace("\n", " ") for s in sessions], fontsize=8)
    ax.set_xlabel("Seconds from first camera event in session")
    ax.set_title("Exp 11: camera inject event timeline", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "camera_inject_timeline.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_vs_rate(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["seagreen" if v < 0 else "tomato" for v in summary["delta_p90_m"]]
    ax.scatter(summary["inject_rate_hz"], summary["delta_p90_m"], s=80, c=colors, edgecolor="white")
    for _, row in summary.iterrows():
        ax.annotate(
            _short_session_label(row["session"]).replace("\n", " "),
            (row["inject_rate_hz"], row["delta_p90_m"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Inject rate (Hz)")
    ax.set_ylabel("CamKF p90 - KF p90 (m)")
    ax.set_title("Exp 11: p90 change vs inject frequency", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_p90_vs_inject_rate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_overall(summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    total_duration_s = float(summary["duration_s"].sum())
    total_injected = int(summary["n_injected"].sum())
    injected = events[events["event"] == "injected"].copy() if not events.empty else pd.DataFrame()

    periods: list[float] = []
    if not injected.empty:
        for _session, grp in injected.sort_values(["session", "ts_wall"]).groupby("session"):
            dts = np.diff(grp["ts_wall"].to_numpy(dtype=float))
            periods.extend(float(v) for v in dts)

    overall = {
        "n_sessions": int(len(summary)),
        "mean_kf_p90": float(summary["kf_p90_err_m"].mean()),
        "mean_camkf_p90": float(summary["camkf_p90_err_m"].mean()),
        "mean_improvement_pct": float(summary["improvement_pct"].mean()),
        "total_duration_s": total_duration_s,
        "total_injected": total_injected,
        "total_skipped_innovation": int(summary["n_skipped_innovation"].sum()),
        "overall_inject_rate_hz": total_injected / total_duration_s if total_duration_s > 0 else math.nan,
        "overall_seconds_per_inject": total_duration_s / total_injected if total_injected > 0 else math.nan,
        "overall_median_inject_period_s": float(np.median(periods)) if periods else math.nan,
        "overall_mean_inject_period_s": float(np.mean(periods)) if periods else math.nan,
        "overall_max_inject_gap_s": float(np.max(periods)) if periods else math.nan,
    }
    (out_dir / "summary_overall.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overall


def _write_plots(summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> None:
    _plot_p90_comparison(summary, out_dir)
    _plot_inject_counts(summary, out_dir)
    _plot_inject_periods(summary, out_dir)
    _plot_inject_timeline(events, out_dir)
    _plot_delta_vs_rate(summary, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 11 offline KF vs CamKF evaluation")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--session", action="append", default=None, help="Session name filter")
    parser.add_argument("--plots-only", action="store_true", help="Build plots from existing CSV files")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        summary_df = pd.read_csv(out_dir / "summary.csv")
        events_df = pd.read_csv(out_dir / "inject_events.csv")
        overall = _write_overall(summary_df, events_df, out_dir)
        _write_plots(summary_df, events_df, out_dir)
        print("=== OVERALL ===")
        print(json.dumps(overall, ensure_ascii=False, indent=2))
        return

    sessions = _sessions(args.root)
    if args.session:
        wanted = set(args.session)
        sessions = [p for p in sessions if p.name in wanted]
    if not sessions:
        raise SystemExit("No sessions found")

    all_rows: list[dict[str, Any]] = []
    all_events: list[InjectEvent] = []
    summaries: list[dict[str, Any]] = []

    for session_path in sessions:
        print(f"=== EVAL {session_path.name} ===", flush=True)
        rows, events, summary = evaluate_session(session_path)
        all_rows.extend(rows)
        all_events.extend(events)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    results_df = pd.DataFrame(all_rows)
    events_df = pd.DataFrame([asdict(ev) for ev in all_events])
    summary_df = pd.DataFrame(summaries)
    results_df.to_csv(out_dir / "results.csv", index=False)
    events_df.to_csv(out_dir / "inject_events.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    overall = _write_overall(summary_df, events_df, out_dir)
    _write_plots(summary_df, events_df, out_dir)
    print("=== OVERALL ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

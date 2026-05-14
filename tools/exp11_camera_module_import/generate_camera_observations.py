#!/usr/bin/env python3
"""Generate camera observations from a recorded DCT session video.

The first integration mode is offline: read `video.mp4`, run the imported YOLO
gate-pose model, refine gate IDs/PnP with the current FK/PF prior, and write
`camera_observations.jsonl` next to the session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from dct.camera_localization import CameraObservation, write_observations_jsonl
from dct.camera_localization.gate_localization import CoarseRefineLocalizer
from dct.camera_localization.yolo_gate_adapter import YoloAdapterConfig
from dct.camera_localization.yolo_gate_adapter.adapter import YoloGateDetection


DEFAULT_WEIGHTS = Path("models/yolo_gate_pose/testgate/weights/best.pt")
DEFAULT_OUTPUT_NAME = "camera_observations.jsonl"


def _load_yolo_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for video inference. Install it or run "
            "the label/calibration smoke tests instead.",
        ) from exc
    return YOLO(str(weights))


def _load_frame_timestamps(session_dir: Path, video_path: Path) -> tuple[np.ndarray, str]:
    ts_path = session_dir / "video_timestamps.parquet"
    if ts_path.exists():
        df = pd.read_parquet(ts_path).sort_values("frame_idx")
        if "ts_wall" not in df.columns:
            raise ValueError(f"{ts_path} has no ts_wall column")
        return df["ts_wall"].to_numpy(dtype=float), "video_timestamps.parquet"

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if n_frames <= 0:
        raise ValueError(f"Cannot determine frame count for {video_path}")

    start_ts = 0.0
    timeline_path = session_dir / "timeline.parquet"
    if timeline_path.exists():
        timeline = pd.read_parquet(timeline_path)
        if len(timeline) and "ts_wall" in timeline.columns:
            start_ts = float(timeline["ts_wall"].iloc[0])
    return start_ts + np.arange(n_frames, dtype=float) / fps, "fps_fallback"


def _iter_sampled_frames(
    video_path: Path,
    timestamps: np.ndarray,
    every_n_frames: int,
    max_frames: int | None,
) -> Iterable[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = -1
        yielded = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % every_n_frames != 0:
                continue
            if frame_idx >= len(timestamps):
                break
            yield frame_idx, float(timestamps[frame_idx]), frame_bgr
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        cap.release()


def _result_to_yolo_detections(result, config: YoloAdapterConfig) -> list[YoloGateDetection]:
    if not result:
        return []
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    box_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    kxy = keypoints.xy.cpu().numpy()
    if getattr(keypoints, "conf", None) is not None:
        kconf = keypoints.conf.cpu().numpy()
    else:
        kconf = np.ones(kxy.shape[:2], dtype=float)

    detections: list[YoloGateDetection] = []
    for i in range(len(xyxy)):
        if kxy[i].shape[0] < 4:
            continue
        keypoint_confidences = tuple(float(v) for v in kconf[i][:4])
        detection = YoloGateDetection(
            keypoints=np.asarray(kxy[i][:4], dtype=np.float64),
            bbox_xyxy=tuple(float(v) for v in xyxy[i]),
            class_id=int(cls[i]),
            bbox_confidence=float(box_conf[i]),
            keypoint_confidences=keypoint_confidences,  # type: ignore[arg-type]
            source_line="ultralytics",
        )
        if detection.min_keypoint_confidence < config.min_keypoint_confidence:
            continue
        if detection.area_px < config.min_box_area_px:
            continue
        detections.append(detection)

    detections.sort(
        key=lambda det: (
            det.min_keypoint_confidence,
            det.mean_keypoint_confidence,
            det.area_px,
        ),
        reverse=True,
    )
    return detections[: config.max_detections]


def _coarse_prior_from_telemetry(session_dir: Path) -> pd.DataFrame | None:
    telemetry_path = session_dir / "telemetry.parquet"
    if not telemetry_path.exists():
        return None
    df = pd.read_parquet(telemetry_path)
    cols = {"ts_wall", "pos_x", "pos_y", "pos_z"}
    if not cols.issubset(df.columns):
        return None
    return df.sort_values("ts_wall")[list(cols)].reset_index(drop=True)


def _nearest_coarse_xyz(telemetry: pd.DataFrame | None, timestamp: float) -> np.ndarray | None:
    if telemetry is None or telemetry.empty:
        return None
    ts = telemetry["ts_wall"].to_numpy(dtype=float)
    idx = int(np.searchsorted(ts, timestamp, side="left"))
    idx = max(0, min(idx, len(ts) - 1))
    row = telemetry.iloc[idx]
    return np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=np.float64)


def _observation_from_result(
    *,
    frame_idx: int,
    timestamp: float,
    detections: list[YoloGateDetection],
    refine_result,
) -> CameraObservation:
    primary_detection = detections[0] if detections else None
    selected = refine_result.selected
    if selected is not None and selected.pose.success:
        pose = selected.pose
        xyz = tuple(float(v) for v in pose.position_world)
        sigma = float(refine_result.q_out_m)
        confidence = float(selected.confidence_to_next)
        gate_id: list[int] | int | None = list(selected.gate_ids)
        reproj = float(pose.reprojection_rmse_px)
    else:
        xyz = (0.0, 0.0, 0.0)
        sigma = float(refine_result.q_out_m)
        confidence = 0.0
        gate_id = None
        reproj = None

    return CameraObservation(
        timestamp=timestamp,
        xyz_obs=xyz,
        sigma_cam=sigma,
        confidence=confidence,
        gate_id=gate_id,
        status="ok" if refine_result.success else "rejected",
        source="fpv_gate_pnp_yolo_video",
        reprojection_error_px=reproj,
        reason=refine_result.reason,
        frame_idx=frame_idx,
        image_timestamp=timestamp,
        bbox_xyxy=(
            None
            if primary_detection is None
            else tuple(float(v) for v in primary_detection.bbox_xyxy)
        ),
        keypoints=(
            None
            if primary_detection is None
            else tuple((float(x), float(y)) for x, y in primary_detection.keypoints)
        ),
    )


def generate(args: argparse.Namespace) -> list[CameraObservation]:
    session_dir = args.session_dir
    video_path = args.video or (session_dir / "video.mp4")
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    timestamps, timestamp_source = _load_frame_timestamps(session_dir, video_path)
    telemetry = _coarse_prior_from_telemetry(session_dir)
    model = _load_yolo_model(args.weights)
    localizer = CoarseRefineLocalizer()
    adapter_config = YoloAdapterConfig(
        image_width_px=args.image_width,
        image_height_px=args.image_height,
        min_keypoint_confidence=args.min_keypoint_confidence,
        max_detections=args.max_detections,
    )

    observations: list[CameraObservation] = []
    summary = {
        "timestamp_source": timestamp_source,
        "frames_processed": 0,
        "frames_with_detections": 0,
        "accepted": 0,
        "rejected": 0,
        "no_prior": 0,
    }

    for frame_idx, timestamp, frame_bgr in _iter_sampled_frames(
        video_path,
        timestamps,
        args.every_n_frames,
        args.max_frames,
    ):
        summary["frames_processed"] += 1
        coarse_xyz = _nearest_coarse_xyz(telemetry, timestamp)
        if coarse_xyz is None:
            summary["no_prior"] += 1
            continue

        result = model.predict(frame_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        yolo_dets = _result_to_yolo_detections(result, adapter_config)
        if yolo_dets:
            summary["frames_with_detections"] += 1
        gate_dets = [det.as_gate_detection() for det in yolo_dets]
        refine = localizer.refine(gate_dets, coarse_xyz, q_m=args.prior_q_m)
        obs = _observation_from_result(
            frame_idx=frame_idx,
            timestamp=timestamp,
            detections=yolo_dets,
            refine_result=refine,
        )
        observations.append(obs)
        if obs.status == "ok":
            summary["accepted"] += 1
        else:
            summary["rejected"] += 1

    output = args.output or (session_dir / DEFAULT_OUTPUT_NAME)
    n_written = write_observations_jsonl(output, observations)
    summary["observations_written"] = n_written
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate camera observations from recorded video")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--every-n-frames", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--prior-q-m", type=float, default=5.0)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()

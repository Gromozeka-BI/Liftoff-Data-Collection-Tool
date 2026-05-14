#!/usr/bin/env python3
"""Smoke test for the imported experimental camera localization modules.

The script uses one annotated frame from the imported calibration JSON, writes a
temporary YOLO-pose label, loads it through `yolo_gate_adapter`, then runs
`CoarseRefineLocalizer` with a nearby coarse prior.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dct.camera_localization.gate_localization import CoarseRefineLocalizer
from dct.camera_localization.pnp_solver.pnp_solver import DEFAULT_CALIB_JSON
from dct.camera_localization.yolo_gate_adapter import (
    YoloAdapterConfig,
    load_gate_detections_from_yolo,
    load_yolo_gate_detections,
)


def _frame_to_yolo_label(frame: dict, image_size: tuple[int, int]) -> str:
    """Convert calibration-frame keypoints to the YOLO pose label format."""

    width_px, height_px = image_size
    lines = []
    for ann in frame["annotations"]:
        points = np.array(
            [[kp["x_px"], kp["y_px"]] for kp in ann["keypoints"]],
            dtype=np.float64,
        )
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        cx = ((x_min + x_max) / 2.0) / width_px
        cy = ((y_min + y_max) / 2.0) / height_px
        box_w = (x_max - x_min) / width_px
        box_h = (y_max - y_min) / height_px

        values: list[float | int] = [0, cx, cy, box_w, box_h]
        for x_px, y_px in points:
            values.extend([x_px / width_px, y_px / height_px, 1.0])
        lines.append(" ".join(str(v) for v in values))
    return "\n".join(lines) + "\n"


def _load_frame(calib_json_path: Path, section: str, frame_idx: int) -> tuple[dict, tuple[int, int]]:
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get(section, [])
    if not frames:
        raise ValueError(f"Section {section!r} is empty in {calib_json_path}")
    if frame_idx < 0 or frame_idx >= len(frames):
        raise ValueError(f"frame_idx must be in [0, {len(frames) - 1}], got {frame_idx}")
    image_size_raw = data["image_size"]
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    return frames[frame_idx], image_size


def _ground_truth_xyz(frame: dict) -> np.ndarray:
    pos = frame["ground_truth"]["position_world"]
    return np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)


def run_smoke(args: argparse.Namespace) -> None:
    frame, image_size = _load_frame(args.calib_json, args.section, args.frame_idx)
    label_text = _frame_to_yolo_label(frame, image_size)
    gt_xyz = _ground_truth_xyz(frame)
    coarse_xyz = gt_xyz + np.array([args.coarse_offset_x, 0.0, args.coarse_offset_z])

    config = YoloAdapterConfig(
        image_width_px=image_size[0],
        image_height_px=image_size[1],
        min_keypoint_confidence=args.min_keypoint_confidence,
        max_detections=args.max_detections,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as tmp:
        tmp.write(label_text)
        tmp_path = Path(tmp.name)

    try:
        yolo_detections = load_yolo_gate_detections(tmp_path, config)
        gate_detections = load_gate_detections_from_yolo(tmp_path, config)
    finally:
        tmp_path.unlink(missing_ok=True)

    localizer = CoarseRefineLocalizer()
    result = localizer.refine(gate_detections, coarse_xyz, q_m=args.q_m)

    print("Camera import smoke test")
    print(f"  frame:       {frame['image_filename']}")
    print(f"  section:     {args.section}[{args.frame_idx}]")
    print(f"  image size:  {image_size[0]}x{image_size[1]}")
    print(f"  detections:  {len(yolo_detections)} after adapter")
    print(f"  GT xyz:      {gt_xyz.round(3).tolist()}")
    print(f"  coarse xyz:  {coarse_xyz.round(3).tolist()}  q={args.q_m:.2f} m")
    print(f"  success:     {result.success}")
    print(f"  reason:      {result.reason}")
    print(f"  q_out:       {result.q_out_m:.3f} m")
    print(f"  runtime:     {result.runtime_ms:.2f} ms")
    if result.selected is not None:
        pose = result.selected.pose
        print(f"  gate ids:    {list(result.selected.gate_ids)}")
        print(f"  pose xyz:    {pose.position_world.round(3).tolist()}")
        print(f"  pose rmse:   {pose.reprojection_rmse_px:.3f} px")
        print(f"  topK conf:   {result.selected.confidence_to_next:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test imported camera localization modules")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB_JSON)
    parser.add_argument("--section", choices=["fov_frames", "multi_frames"], default="fov_frames")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--q-m", type=float, default=1.0)
    parser.add_argument("--coarse-offset-x", type=float, default=2.0)
    parser.add_argument("--coarse-offset-z", type=float, default=0.0)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=6)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()

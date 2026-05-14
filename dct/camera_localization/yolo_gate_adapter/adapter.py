#!/usr/bin/env python3
"""Adapter from YOLO keypoint labels to gate-localization detections."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from dct.camera_localization.gate_localization.global_search import GateDetection


class YoloGateDetection(NamedTuple):
    keypoints: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    bbox_confidence: float | None
    keypoint_confidences: tuple[float, float, float, float]
    source_line: str

    @property
    def min_keypoint_confidence(self) -> float:
        return min(self.keypoint_confidences)

    @property
    def mean_keypoint_confidence(self) -> float:
        return float(np.mean(self.keypoint_confidences))

    @property
    def area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def as_gate_detection(self) -> GateDetection:
        return GateDetection(self.keypoints)


class YoloAdapterConfig(NamedTuple):
    image_width_px: int = 1920
    image_height_px: int = 1080
    min_keypoint_confidence: float = 0.7
    min_box_area_px: float = 0.0
    max_detections: int = 6
    deduplicate_iou_threshold: float = 0.75
    deduplicate_center_distance_px: float = 12.0


def load_yolo_gate_detections(
    label_path: str | Path,
    config: YoloAdapterConfig = YoloAdapterConfig(),
) -> list[YoloGateDetection]:
    """Load and filter one YOLO keypoint label file.

    Expected format per line:
    class cx cy w h x1 y1 c1 x2 y2 c2 x3 y3 c3 x4 y4 c4
    with normalized coordinates.
    """

    label_path = Path(label_path)
    if not label_path.exists():
        return []

    detections: list[YoloGateDetection] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = _parse_yolo_keypoint_line(line, config)
        if parsed is None:
            continue
        if parsed.min_keypoint_confidence < config.min_keypoint_confidence:
            continue
        if parsed.area_px < config.min_box_area_px:
            continue
        detections.append(parsed)

    detections.sort(
        key=lambda det: (
            det.min_keypoint_confidence,
            det.mean_keypoint_confidence,
            det.area_px,
        ),
        reverse=True,
    )
    detections = _deduplicate_detections(detections, config)
    return detections[: config.max_detections]


def load_gate_detections_from_yolo(
    label_path: str | Path,
    config: YoloAdapterConfig = YoloAdapterConfig(),
) -> list[GateDetection]:
    """Return only the GateDetection objects expected by gate_localization."""

    return [
        detection.as_gate_detection()
        for detection in load_yolo_gate_detections(label_path, config)
    ]


def _parse_yolo_keypoint_line(
    line: str,
    config: YoloAdapterConfig,
) -> YoloGateDetection | None:
    parts = line.split()
    if len(parts) != 17:
        return None

    try:
        class_id = int(float(parts[0]))
        values = [float(part) for part in parts[1:]]
    except ValueError:
        return None

    cx, cy, width, height = values[:4]
    image_width = float(config.image_width_px)
    image_height = float(config.image_height_px)
    bbox_xyxy = (
        (cx - width / 2.0) * image_width,
        (cy - height / 2.0) * image_height,
        (cx + width / 2.0) * image_width,
        (cy + height / 2.0) * image_height,
    )

    keypoints = []
    confidences = []
    for idx in range(4):
        x_norm = values[4 + idx * 3]
        y_norm = values[5 + idx * 3]
        confidence = values[6 + idx * 3]
        keypoints.append((x_norm * image_width, y_norm * image_height))
        confidences.append(confidence)

    return YoloGateDetection(
        keypoints=np.asarray(keypoints, dtype=np.float64),
        bbox_xyxy=bbox_xyxy,
        class_id=class_id,
        bbox_confidence=None,
        keypoint_confidences=tuple(confidences),  # type: ignore[arg-type]
        source_line=line,
    )


def _deduplicate_detections(
    detections: list[YoloGateDetection],
    config: YoloAdapterConfig,
) -> list[YoloGateDetection]:
    unique: list[YoloGateDetection] = []
    for detection in detections:
        if any(_is_duplicate(detection, existing, config) for existing in unique):
            continue
        unique.append(detection)
    return unique


def _is_duplicate(
    left: YoloGateDetection,
    right: YoloGateDetection,
    config: YoloAdapterConfig,
) -> bool:
    left_center = np.mean(left.keypoints, axis=0)
    right_center = np.mean(right.keypoints, axis=0)
    center_distance = float(np.linalg.norm(left_center - right_center))
    if center_distance <= config.deduplicate_center_distance_px:
        return True
    return _bbox_iou(left.bbox_xyxy, right.bbox_xyxy) >= config.deduplicate_iou_threshold


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union_area = left_area + right_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

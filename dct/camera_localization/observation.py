"""Camera observation schema and JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CameraObservation:
    """One camera-derived position observation in DCT track coordinates."""

    timestamp: float
    xyz_obs: tuple[float, float, float]
    sigma_cam: float
    confidence: float
    gate_id: int | str | list[int] | None
    status: str
    source: str = "fpv_gate_pnp"
    reprojection_error_px: float | None = None
    reason: str = ""
    frame_idx: int | None = None
    image_timestamp: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    keypoints: tuple[tuple[float, float], ...] | None = None

    @property
    def xyz_array(self) -> np.ndarray:
        return np.asarray(self.xyz_obs, dtype=np.float64)

    @property
    def inject_ready(self) -> bool:
        return self.status == "ok" and self.sigma_cam > 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CameraObservation":
        xyz = data.get("xyz_obs", data.get("xyz"))
        if xyz is None:
            raise ValueError("CameraObservation requires xyz_obs")
        return cls(
            timestamp=float(data["timestamp"]),
            xyz_obs=tuple(float(v) for v in xyz),
            sigma_cam=float(data["sigma_cam"]),
            confidence=float(data.get("confidence", 0.0)),
            gate_id=data.get("gate_id"),
            status=str(data.get("status", "ok")),
            source=str(data.get("source", "fpv_gate_pnp")),
            reprojection_error_px=(
                None
                if data.get("reprojection_error_px") is None
                else float(data["reprojection_error_px"])
            ),
            reason=str(data.get("reason", "")),
            frame_idx=None if data.get("frame_idx") is None else int(data["frame_idx"]),
            image_timestamp=(
                None if data.get("image_timestamp") is None else float(data["image_timestamp"])
            ),
            bbox_xyxy=(
                None
                if data.get("bbox_xyxy") is None
                else tuple(float(v) for v in data["bbox_xyxy"])  # type: ignore[arg-type]
            ),
            keypoints=(
                None
                if data.get("keypoints") is None
                else tuple(
                    (float(pt[0]), float(pt[1]))
                    for pt in data["keypoints"]
                )
            ),
        )

    def to_overlay_dict(self) -> dict[str, Any] | None:
        if self.bbox_xyxy is None and self.keypoints is None:
            return None
        return {
            "bbox_xyxy": self.bbox_xyxy,
            "keypoints": self.keypoints,
            "gate_id": self.gate_id,
            "confidence": self.confidence,
            "status": self.status,
        }


def write_observations_jsonl(path: str | Path, observations: Iterable[CameraObservation]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for obs in observations:
            f.write(json.dumps(obs.to_json_dict(), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def read_observations_jsonl(path: str | Path) -> list[CameraObservation]:
    path = Path(path)
    observations: list[CameraObservation] = []
    if not path.exists():
        return observations
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            observations.append(CameraObservation.from_json_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid camera observation at {path}:{line_no}: {exc}") from exc
    observations.sort(key=lambda obs: obs.timestamp)
    return observations

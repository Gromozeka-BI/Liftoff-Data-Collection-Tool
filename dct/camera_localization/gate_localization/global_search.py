#!/usr/bin/env python3
"""Global gate-ID search over the known track map.

The detector provides gate corners but no gate_id.  This module enumerates
possible IDs, asks the existing PnPSolver to score each hypothesis, and returns
the best assignment together with the pose.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from dct.camera_localization.pnp_solver.pnp_solver import (
    DEFAULT_CALIB_JSON,
    PnPResult,
    PnPSolver,
    _ippe_candidates_single,
)


class GateDetection(NamedTuple):
    """One detected gate without ID."""

    keypoints: np.ndarray  # shape (4, 2), TL -> TR -> BR -> BL


class GateAssignment(NamedTuple):
    """Assignment of one detection to one gate from the map."""

    detection_index: int
    gate_id: int
    confidence: float


class GlobalSearchResult(NamedTuple):
    """Output of global gate-ID search."""

    success: bool
    assignments: list[GateAssignment]
    pose: PnPResult
    score: float
    reason: str
    second_best_score: float = float("inf")
    n_hypotheses: int = 0


class _Hypothesis(NamedTuple):
    gate_ids: tuple[int, ...]
    pose: PnPResult
    score: float
    physical_penalty: float
    spread_m: float = float("inf")
    mean_rmse_px: float = float("inf")
    candidate_labels: str = ""


class GlobalSearchLocalizer:
    """Assign gate IDs by exhaustive map search and existing PnP verification."""

    def __init__(
        self,
        solver: Optional[PnPSolver] = None,
        *,
        max_detections_for_full_search: int = 3,
        max_hypotheses: int = 3000,
        max_rmse_px: float = 50.0,
        min_confidence: float = 0.05,
        min_y_m: float = -0.5,
        max_y_m: float = 20.0,
        max_track_margin_m: float = 80.0,
    ):
        self.solver = solver or PnPSolver()
        self.max_detections_for_full_search = max_detections_for_full_search
        self.max_hypotheses = max_hypotheses
        self.max_rmse_px = max_rmse_px
        self.min_confidence = min_confidence
        self.min_y_m = min_y_m
        self.max_y_m = max_y_m
        self.max_track_margin_m = max_track_margin_m

    def assign_and_solve(
        self,
        detections: list[GateDetection | np.ndarray],
        *,
        candidate_gate_ids: Optional[list[int]] = None,
        use_ransac: bool = False,
    ) -> GlobalSearchResult:
        valid = self._normalize_detections(detections)
        if not valid:
            return self._failed("no valid 4-corner detections")

        gate_ids = self._candidate_ids(candidate_gate_ids)
        if not gate_ids:
            return self._failed("empty candidate gate list")

        if len(valid) > self.max_detections_for_full_search:
            return self._failed(
                f"too many detections for full search: {len(valid)} > "
                f"{self.max_detections_for_full_search}"
            )

        total_hypotheses = math.perm(len(gate_ids), len(valid))
        if total_hypotheses > self.max_hypotheses:
            return self._failed(
                f"too many hypotheses: {total_hypotheses} > {self.max_hypotheses}"
            )

        ranked: list[_Hypothesis] = []
        for gate_tuple in itertools.permutations(gate_ids, len(valid)):
            gate_detections = [
                (gate_id, valid[det_i])
                for det_i, gate_id in enumerate(gate_tuple)
            ]
            pose = self.solver.solve(gate_detections, use_ransac=use_ransac)
            physical_penalty = self._physical_penalty(pose)
            if len(valid) >= 2:
                score, spread, mean_rmse, labels = self._score_gate_tuple_by_consensus(
                    gate_tuple,
                    valid,
                    physical_penalty,
                )
            else:
                score = self._score_pose(pose, physical_penalty)
                spread = 0.0
                mean_rmse = float(pose.reprojection_rmse_px)
                labels = pose.perm_used
            ranked.append(
                _Hypothesis(
                    gate_tuple,
                    pose,
                    score,
                    physical_penalty,
                    spread,
                    mean_rmse,
                    labels,
                )
            )

        ranked.sort(key=lambda h: h.score)
        best = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else float("inf")
        if not best.pose.success:
            return GlobalSearchResult(
                success=False,
                assignments=[],
                pose=best.pose,
                score=best.score,
                reason="best hypothesis failed PnP",
                second_best_score=second_score,
                n_hypotheses=len(ranked),
            )
        if best.pose.reprojection_rmse_px > self.max_rmse_px:
            reason = (
                f"best hypothesis RMSE too high: "
                f"{best.pose.reprojection_rmse_px:.2f}px > {self.max_rmse_px:.2f}px"
            )
            return GlobalSearchResult(
                success=False,
                assignments=[],
                pose=best.pose,
                score=best.score,
                reason=reason,
                second_best_score=second_score,
                n_hypotheses=len(ranked),
            )

        confidence = self._confidence(best.score, second_score)
        assignments = [
            GateAssignment(det_i, int(gate_id), confidence)
            for det_i, gate_id in enumerate(best.gate_ids)
        ]
        if confidence < self.min_confidence:
            return GlobalSearchResult(
                success=False,
                assignments=assignments,
                pose=best.pose,
                score=best.score,
                reason=(
                    f"ambiguous assignment: confidence={confidence:.3f} "
                    f"< {self.min_confidence:.3f}"
                ),
                second_best_score=second_score,
                n_hypotheses=len(ranked),
            )
        reason = (
            f"ok: margin={second_score - best.score:.3f}, "
            f"spread={best.spread_m:.3f}m, "
            f"candidate_rmse={best.mean_rmse_px:.3f}px, "
            f"candidates={best.candidate_labels}, "
            f"physical_penalty={best.physical_penalty:.3f}"
        )
        return GlobalSearchResult(
            success=True,
            assignments=assignments,
            pose=best.pose,
            score=best.score,
            reason=reason,
            second_best_score=second_score,
            n_hypotheses=len(ranked),
        )

    def _normalize_detections(
        self,
        detections: list[GateDetection | np.ndarray],
    ) -> list[np.ndarray]:
        valid = []
        for det in detections:
            pts = det.keypoints if isinstance(det, GateDetection) else det
            arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            if arr.shape == (4, 2):
                valid.append(arr)
        return valid

    def _candidate_ids(self, candidate_gate_ids: Optional[list[int]]) -> list[int]:
        if candidate_gate_ids is None:
            return self.solver.gate_map.get_all_ids()
        return [
            int(gid)
            for gid in candidate_gate_ids
            if int(gid) in self.solver.gate_map.gates
        ]

    def _score_pose(self, pose: PnPResult, physical_penalty: float) -> float:
        if not pose.success:
            return 1_000_000.0
        rmse = float(pose.reprojection_rmse_px)
        if not np.isfinite(rmse):
            rmse = 100_000.0
        return rmse + physical_penalty

    def _score_gate_tuple_by_consensus(
        self,
        gate_tuple: tuple[int, ...],
        detections: list[np.ndarray],
        physical_penalty: float,
    ) -> tuple[float, float, float, str]:
        per_detection: list[list[tuple[str, int, np.ndarray, float]]] = []
        for det_i, gate_id in enumerate(gate_tuple):
            pts3 = self.solver.gate_map.get_corners(gate_id).astype(np.float64)
            raw = _ippe_candidates_single(pts3, detections[det_i], self.solver.K, self.solver.dist)
            cands = [
                (perm_name, cand_idx, pos_world, rmse)
                for perm_name, cand_idx, _, pos_world, _, rmse in raw
                if pos_world[1] >= self.min_y_m
            ]
            if not cands:
                return 1_000_000.0 + physical_penalty, float("inf"), float("inf"), ""
            per_detection.append(cands)

        best_key: tuple[float, float, float] | None = None
        best_spread = float("inf")
        best_rmse = float("inf")
        best_labels = ""
        for combo in itertools.product(*per_detection):
            positions = np.array([c[2] for c in combo], dtype=np.float64)
            center = np.median(positions, axis=0)
            spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
            mean_rmse = float(np.mean([c[3] for c in combo]))
            max_rmse = float(np.max([c[3] for c in combo]))
            key = (spread, mean_rmse, max_rmse)
            if best_key is None or key < best_key:
                best_key = key
                best_spread = spread
                best_rmse = mean_rmse
                best_labels = "+".join(f"{c[0]}:c{c[1]}" for c in combo)

        # Position consensus is the primary signal; RMSE only breaks ties.
        score = best_spread * 100.0 + best_rmse + physical_penalty
        return score, best_spread, best_rmse, best_labels

    def _physical_penalty(self, pose: PnPResult) -> float:
        if not pose.success:
            return 0.0
        pos = pose.position_world.astype(np.float64)
        penalty = 0.0
        if pos[1] < self.min_y_m:
            penalty += 1000.0 + 100.0 * (self.min_y_m - pos[1])
        if pos[1] > self.max_y_m:
            penalty += 1000.0 + 20.0 * (pos[1] - self.max_y_m)

        centers = np.array(
            [gate.position.astype(np.float64) for gate in self.solver.gate_map],
            dtype=np.float64,
        )
        if len(centers):
            min_xz_dist = float(np.min(np.linalg.norm(centers[:, [0, 2]] - pos[[0, 2]], axis=1)))
            if min_xz_dist > self.max_track_margin_m:
                penalty += min_xz_dist - self.max_track_margin_m
        return penalty

    @staticmethod
    def _confidence(best_score: float, second_score: float) -> float:
        if not np.isfinite(second_score):
            return 1.0
        margin = max(0.0, second_score - best_score)
        return float(1.0 - math.exp(-margin / 5.0))

    def _failed(self, reason: str) -> GlobalSearchResult:
        failed_pose = self.solver.solve([])
        return GlobalSearchResult(False, [], failed_pose, float("inf"), reason)


def _detections_from_frame(frame: dict) -> tuple[list[GateDetection], list[int], np.ndarray]:
    detections: list[GateDetection] = []
    gt_ids: list[int] = []
    for ann in frame["annotations"]:
        pts = np.array([[p["x_px"], p["y_px"]] for p in ann["keypoints"]], dtype=np.float64)
        detections.append(GateDetection(pts))
        gt_ids.append(int(ann["gate_id"]))

    gt = frame["ground_truth"]["position_world"]
    gt_pos = np.array([gt["x"], gt["y"], gt["z"]], dtype=np.float64)
    return detections, gt_ids, gt_pos


def validate_on_calibration_json(
    calib_json_path: Path = DEFAULT_CALIB_JSON,
    *,
    sections: tuple[str, ...] = ("fov_frames", "multi_frames"),
    max_detections_for_full_search: int = 3,
    max_hypotheses: int = 3000,
) -> None:
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)

    localizer = GlobalSearchLocalizer(
        max_detections_for_full_search=max_detections_for_full_search,
        max_hypotheses=max_hypotheses,
    )
    total_frames = 0
    ok_frames = 0
    total_assignments = 0
    ok_assignments = 0
    pos_errors: list[float] = []

    print("Global search validation:")
    print(
        f"  {'Frame':35s} {'GT ids':16s} {'Pred ids':16s} "
        f"{'ID OK':5s} {'pos_err':>8s} {'RMSE':>8s} {'hyp':>5s}  reason"
    )
    print("  " + "-" * 118)

    for section in sections:
        for frame in data.get(section, []):
            detections, gt_ids, gt_pos = _detections_from_frame(frame)
            result = localizer.assign_and_solve(detections)
            pred_ids = [a.gate_id for a in result.assignments]
            ids_ok = result.success and pred_ids == gt_ids
            pos_err = float("inf")
            if result.pose.success:
                pos_err = float(np.linalg.norm(result.pose.position_world.astype(np.float64) - gt_pos))
                pos_errors.append(pos_err)

            total_frames += 1
            ok_frames += int(ids_ok)
            total_assignments += len(gt_ids)
            ok_assignments += sum(int(a == b) for a, b in zip(pred_ids, gt_ids))

            print(
                f"  {frame['image_filename']:35s} "
                f"{str(gt_ids):16s} {str(pred_ids):16s} "
                f"{str(ids_ok):5s} {pos_err:8.3f} "
                f"{result.pose.reprojection_rmse_px:8.3f} "
                f"{result.n_hypotheses:5d}  {result.reason}"
            )

    mean_pos = float(np.mean(pos_errors)) if pos_errors else float("inf")
    print("\nSummary:")
    print(f"  frame ID accuracy:      {ok_frames}/{total_frames}")
    print(f"  assignment ID accuracy: {ok_assignments}/{total_assignments}")
    print(f"  mean pose error:        {mean_pos:.3f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Global gate-ID search validation")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB_JSON)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--max-hypotheses", type=int, default=3000)
    args = parser.parse_args()

    if args.validate:
        validate_on_calibration_json(
            args.calib_json,
            max_detections_for_full_search=args.max_detections,
            max_hypotheses=args.max_hypotheses,
        )
        return
    parser.error("Use --validate")


if __name__ == "__main__":
    main()

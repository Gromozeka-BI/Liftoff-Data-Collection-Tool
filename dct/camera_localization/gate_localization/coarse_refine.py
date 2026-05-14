#!/usr/bin/env python3
"""Coarse-pose prior for gate-ID localization.

V1 builds per-detection candidate gate lists from radial gate rings around a
coarse camera position, then asks TopKHypothesisGenerator for ranked ID
hypotheses and chooses the one closest to the coarse position.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from dct.camera_localization.gate_localization.global_search import GateDetection
from dct.camera_localization.gate_localization.topk_hypotheses import (
    TopKHypothesis,
    TopKHypothesisGenerator,
    TopKResult,
)
from dct.camera_localization.pnp_solver.pnp_solver import (
    DEFAULT_CALIB,
    DEFAULT_CALIB_JSON,
    DEFAULT_TRACK,
    PnPSolver,
)


class CoarsePriorConfig(NamedTuple):
    min_near_radius_m: float = 10.0
    min_mid_radius_m: float = 25.0
    min_far_radius_m: float = 40.0
    mid_radius_factor: float = 2.0
    far_radius_factor: float = 4.0
    max_candidates_per_detection: int = 14
    far_tail_candidates: int = 4
    extra_tail_candidates: int = 4
    group_first_min_detections: int = 4
    group_extra_candidates: int = 1
    fallback_max_spread_m: float = 3.0
    fallback_max_coarse_distance_m: float = 8.0
    coarse_distance_weight: float = 10.0
    q_base_m: float = 0.25
    q_spread_weight: float = 1.0
    q_ambiguity_penalty_m: float = 2.0
    q_coarse_distance_weight: float = 0.15
    q_timeout_penalty_m: float = 1.0
    q_min_m: float = 0.2
    q_max_m: float = 20.0
    q_injection_max_m: float = 3.0
    reject_distance_factor: float = 3.0
    reject_min_distance_m: float = 5.0
    single_gate_reject_factor: float = 2.0
    single_gate_reject_min_m: float = 3.0
    min_prior_uncertainty_for_injection_m: float = 1.0
    useful_distance_factor: float = 1.0


class CoarseCandidateSet(NamedTuple):
    candidate_gate_ids_by_detection: list[list[int]]
    gate_distance_by_id: dict[int, float]
    ring_by_gate_id: dict[int, str]


class CoarseRefineResult(NamedTuple):
    success: bool
    selected: Optional[TopKHypothesis]
    topk: TopKResult
    candidate_set: CoarseCandidateSet
    q_out_m: float
    runtime_ms: float
    reason: str


class CoarseRefineLocalizer:
    """Use approximate position to shortlist gates and refine top-K IDs."""

    def __init__(
        self,
        solver: Optional[PnPSolver] = None,
        *,
        calib_path: Optional[Path] = None,
        track_path: Optional[Path] = None,
        config: CoarsePriorConfig = CoarsePriorConfig(),
        topk_generator: Optional[TopKHypothesisGenerator] = None,
    ):
        self.solver = solver or PnPSolver(
            calib_path=calib_path or DEFAULT_CALIB,
            track_path=track_path or DEFAULT_TRACK,
        )
        self.config = config
        self.topk = topk_generator or TopKHypothesisGenerator(
            solver=self.solver,
            top_k=10,
            per_detection_top_n=config.max_candidates_per_detection,
            beam_width=64,
            partial_spread_prune_m=20.0,
            pairwise_compatibility_m=8.0,
            max_pose_solves=6,
            max_pose_solves_many_detections=3,
            time_budget_ms=200.0,
        )

    def refine(
        self,
        detections: list[GateDetection | np.ndarray],
        coarse_position_world: np.ndarray,
        q_m: float,
    ) -> CoarseRefineResult:
        t0 = time.perf_counter()
        valid = self._normalize_detections(detections)
        candidate_set = self.build_candidate_set(valid, coarse_position_world, q_m)
        active_candidate_set = candidate_set
        if len(valid) >= self.config.group_first_min_detections:
            active_candidate_set = self.build_group_candidate_set(
                valid,
                coarse_position_world,
                q_m,
            )
        topk = self.topk.generate(
            valid,
            candidate_gate_ids_by_detection=active_candidate_set.candidate_gate_ids_by_detection,
        )
        if not topk.hypotheses:
            topk = self._generate_exhaustive_fallback(
                valid,
                active_candidate_set.candidate_gate_ids_by_detection,
            )
        selected = self._select_by_coarse_position(topk, coarse_position_world)
        if (
            active_candidate_set is not candidate_set
            and self._needs_fallback(topk, selected, coarse_position_world)
        ):
            topk = self.topk.generate(
                valid,
                candidate_gate_ids_by_detection=candidate_set.candidate_gate_ids_by_detection,
            )
            if not topk.hypotheses:
                topk = self._generate_exhaustive_fallback(
                    valid,
                    candidate_set.candidate_gate_ids_by_detection,
                )
            selected = self._select_by_coarse_position(topk, coarse_position_world)
            active_candidate_set = candidate_set
        success = selected is not None
        reason = "ok" if success else "no top-K hypotheses"
        q_out_m = self._estimate_q_out_m(
            topk,
            selected,
            coarse_position_world,
            len(valid),
            q_m,
        )
        if selected is not None and self._should_reject_by_coarse_distance(
            selected,
            coarse_position_world,
            q_m,
            len(valid),
        ):
            success = False
            reason = "visual pose rejected: too far from coarse prior"
            q_out_m = self.config.q_max_m
        elif selected is not None and self._should_reject_as_not_useful(
            selected,
            coarse_position_world,
            q_m,
        ):
            success = False
            reason = "visual pose rejected: not useful versus coarse prior"
            q_out_m = self.config.q_max_m
        elif q_out_m > self.config.q_injection_max_m:
            success = False
            reason = "visual pose rejected: q_out too high for injection"
            q_out_m = self.config.q_max_m
        return CoarseRefineResult(
            success=success,
            selected=selected,
            topk=topk,
            candidate_set=active_candidate_set,
            q_out_m=q_out_m,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            reason=reason,
        )

    def build_candidate_set(
        self,
        detections: list[np.ndarray],
        coarse_position_world: np.ndarray,
        q_m: float,
    ) -> CoarseCandidateSet:
        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)
        q = max(float(q_m), 1e-6)
        near_radius = max(q, self.config.min_near_radius_m)
        mid_radius = max(q * self.config.mid_radius_factor, self.config.min_mid_radius_m)
        far_radius = max(q * self.config.far_radius_factor, self.config.min_far_radius_m)
        gate_distance_by_id: dict[int, float] = {}
        ring_by_gate_id: dict[int, str] = {}
        rings = {"near": [], "mid": [], "far": [], "extra": []}

        for gate in self.solver.gate_map:
            pos = gate.position.astype(np.float64)
            dist = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))
            gate_distance_by_id[gate.gate_id] = dist
            if dist <= near_radius:
                ring = "near"
            elif dist <= mid_radius:
                ring = "mid"
            elif dist <= far_radius:
                ring = "far"
            else:
                ring = "extra"
            ring_by_gate_id[gate.gate_id] = ring
            rings[ring].append(gate.gate_id)

        for ids in rings.values():
            ids.sort(key=lambda gid: gate_distance_by_id[gid])

        areas = np.array([self._quad_area(det) for det in detections], dtype=np.float64)
        order = np.argsort(-areas) if len(areas) else np.array([], dtype=int)
        rank_by_det = {int(det_i): rank for rank, det_i in enumerate(order)}

        per_detection: list[list[int]] = []
        for det_i in range(len(detections)):
            rank = rank_by_det.get(det_i, det_i)
            if rank == 0:
                ids = rings["near"] + rings["mid"]
            elif rank == len(detections) - 1:
                ids = rings["mid"] + rings["far"] + rings["near"]
            else:
                ids = rings["near"] + rings["mid"] + rings["far"]

            if not ids:
                ids = rings["near"] + rings["mid"] + rings["far"] + rings["extra"]
            # Keep a small distance-sorted tail from farther rings. Single-gate
            # and symmetric-gate frames can otherwise lose the true ID before
            # top-K has a chance to use image geometry.
            ids = (
                ids
                + rings["far"][: self.config.far_tail_candidates]
                + rings["extra"][: self.config.extra_tail_candidates]
            )
            ids = self._unique_sorted_by_distance(ids, gate_distance_by_id)
            per_detection.append(ids[: self.config.max_candidates_per_detection])

        return CoarseCandidateSet(per_detection, gate_distance_by_id, ring_by_gate_id)

    def build_group_candidate_set(
        self,
        detections: list[np.ndarray],
        coarse_position_world: np.ndarray,
        q_m: float,
    ) -> CoarseCandidateSet:
        wide = self.build_candidate_set(detections, coarse_position_world, q_m)
        extra = self.config.group_extra_candidates if len(detections) <= 4 else 0
        n_ids = max(1, len(detections) + extra)
        group_ids = sorted(
            wide.gate_distance_by_id,
            key=lambda gid: wide.gate_distance_by_id[gid],
        )[:n_ids]
        per_detection = [group_ids for _ in detections]
        return CoarseCandidateSet(
            per_detection,
            wide.gate_distance_by_id,
            wide.ring_by_gate_id,
        )

    def _generate_exhaustive_fallback(
        self,
        detections: list[np.ndarray],
        candidate_gate_ids_by_detection: list[list[int]],
    ) -> TopKResult:
        fallback = TopKHypothesisGenerator(
            solver=self.solver,
            top_k=self.topk.top_k,
            search_mode="exhaustive",
            max_exhaustive_hypotheses=100000,
            max_candidate_ids=self.topk.max_candidate_ids,
            max_pose_solves=self.topk.max_pose_solves,
            max_pose_solves_many_detections=self.topk.max_pose_solves_many_detections,
            time_budget_ms=self.topk.time_budget_ms,
            pairwise_compatibility_m=self.topk.pairwise_compatibility_m,
            partial_spread_prune_m=self.topk.partial_spread_prune_m,
            beam_width=self.topk.beam_width,
            per_detection_top_n=self.topk.per_detection_top_n,
        )
        return fallback.generate(
            detections,
            candidate_gate_ids_by_detection=candidate_gate_ids_by_detection,
        )

    def _select_by_coarse_position(
        self,
        topk: TopKResult,
        coarse_position_world: np.ndarray,
    ) -> Optional[TopKHypothesis]:
        if not topk.hypotheses:
            return None
        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)

        def final_score(hyp: TopKHypothesis) -> float:
            if not hyp.pose.success:
                return float("inf")
            pos = hyp.pose.position_world.astype(np.float64)
            dist = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))
            return hyp.score + dist * self.config.coarse_distance_weight

        return min(topk.hypotheses, key=final_score)

    def _needs_fallback(
        self,
        topk: TopKResult,
        selected: Optional[TopKHypothesis],
        coarse_position_world: np.ndarray,
    ) -> bool:
        if not topk.hypotheses:
            return True
        if selected is None or not selected.pose.success:
            return True
        if selected.spread_m > self.config.fallback_max_spread_m:
            return True
        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)
        pos = selected.pose.position_world.astype(np.float64)
        dist = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))
        return dist > self.config.fallback_max_coarse_distance_m

    def _estimate_q_out_m(
        self,
        topk: TopKResult,
        selected: Optional[TopKHypothesis],
        coarse_position_world: np.ndarray,
        n_detections: int,
        q_m: float,
    ) -> float:
        if selected is None or not selected.pose.success:
            return self.config.q_max_m

        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)
        pos = selected.pose.position_world.astype(np.float64)
        coarse_dist = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))

        # More visible gates usually constrain position better, but avoid
        # claiming extreme precision from this heuristic confidence estimate.
        visibility_factor = 1.0 / max(1.0, np.sqrt(float(n_detections)))
        q = self.config.q_base_m + selected.spread_m * self.config.q_spread_weight
        q *= visibility_factor
        q += coarse_dist * self.config.q_coarse_distance_weight

        if selected.confidence_to_next < 0.5:
            q += self.config.q_ambiguity_penalty_m * (1.0 - selected.confidence_to_next)
        if topk.timed_out:
            q += self.config.q_timeout_penalty_m
        if self._should_reject_by_coarse_distance(
            selected,
            coarse_position_world,
            q_m,
            n_detections,
        ):
            q = self.config.q_max_m
        elif self._should_reject_as_not_useful(
            selected,
            coarse_position_world,
            q_m,
        ):
            q = self.config.q_max_m

        return float(np.clip(q, self.config.q_min_m, self.config.q_max_m))

    def _should_reject_by_coarse_distance(
        self,
        selected: TopKHypothesis,
        coarse_position_world: np.ndarray,
        q_m: float,
        n_detections: int,
    ) -> bool:
        if not selected.pose.success:
            return True
        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)
        pos = selected.pose.position_world.astype(np.float64)
        dist_xz = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))
        if n_detections <= 1:
            threshold = max(
                self.config.single_gate_reject_factor * float(q_m),
                self.config.single_gate_reject_min_m,
            )
        else:
            threshold = max(
                self.config.reject_distance_factor * float(q_m),
                self.config.reject_min_distance_m,
            )
        return dist_xz > threshold

    def _should_reject_as_not_useful(
        self,
        selected: TopKHypothesis,
        coarse_position_world: np.ndarray,
        q_m: float,
    ) -> bool:
        if not selected.pose.success:
            return True
        coarse = np.asarray(coarse_position_world, dtype=np.float64).reshape(3)
        pos = selected.pose.position_world.astype(np.float64)
        dist_xz = float(np.linalg.norm(pos[[0, 2]] - coarse[[0, 2]]))
        useful_distance_m = max(
            self.config.useful_distance_factor * float(q_m),
            self.config.min_prior_uncertainty_for_injection_m,
        )
        return dist_xz < useful_distance_m

    @staticmethod
    def _normalize_detections(detections: list[GateDetection | np.ndarray]) -> list[np.ndarray]:
        valid = []
        for det in detections:
            pts = det.keypoints if isinstance(det, GateDetection) else det
            arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            if arr.shape == (4, 2):
                valid.append(arr)
        return valid

    @staticmethod
    def _quad_area(pts2: np.ndarray) -> float:
        pts = np.asarray(pts2, dtype=np.float64).reshape(4, 2)
        x, y = pts[:, 0], pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _unique_sorted_by_distance(ids: list[int], distances: dict[int, float]) -> list[int]:
        return sorted(dict.fromkeys(ids), key=lambda gid: distances[gid])


def _detections_from_frame(frame: dict) -> tuple[list[GateDetection], list[int], np.ndarray]:
    detections = []
    gt_ids = []
    for ann in frame["annotations"]:
        pts = np.array([[p["x_px"], p["y_px"]] for p in ann["keypoints"]], dtype=np.float64)
        detections.append(GateDetection(pts))
        gt_ids.append(int(ann["gate_id"]))
    gt = frame["ground_truth"]["position_world"]
    return detections, gt_ids, np.array([gt["x"], gt["y"], gt["z"]], dtype=np.float64)


def validate_on_calibration_json(
    calib_json_path: Path = DEFAULT_CALIB_JSON,
    *,
    sections: tuple[str, ...] = ("multi_frames",),
) -> None:
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)
    localizer = CoarseRefineLocalizer()
    print("Coarse refine validation:")
    print(
        f"  {'Frame':30s} {'GT ids':20s} {'selected':20s} "
        f"{'GT rank':>7s} {'q_out':>7s} {'topK ms':>8s} {'total ms':>8s} reason"
    )
    print("  " + "-" * 112)
    for section in sections:
        for frame in data.get(section, []):
            detections, gt_ids, gt_pos = _detections_from_frame(frame)
            result = localizer.refine(detections, gt_pos, q_m=10.0)
            selected_ids = result.selected.gate_ids if result.selected else ()
            gt_rank = "-"
            for hyp in result.topk.hypotheses:
                if hyp.gate_ids == tuple(gt_ids):
                    gt_rank = str(hyp.rank)
                    break
            print(
                f"  {frame['image_filename']:30s} {str(gt_ids):20s} "
                f"{str(selected_ids):20s} {gt_rank:>7s} "
                f"{result.q_out_m:7.2f} {result.topk.runtime_ms:8.2f} "
                f"{result.runtime_ms:8.2f} {result.reason}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Coarse prior + top-K refinement")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB_JSON)
    args = parser.parse_args()
    if args.validate:
        validate_on_calibration_json(args.calib_json)
        return
    parser.error("Use --validate")


if __name__ == "__main__":
    main()

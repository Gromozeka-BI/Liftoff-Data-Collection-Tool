#!/usr/bin/env python3
"""Real-time oriented top-K gate-ID hypothesis generation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np

from dct.camera_localization.gate_localization.global_search import GateAssignment, GateDetection
from dct.camera_localization.pnp_solver.pnp_solver import (
    DEFAULT_CALIB_JSON,
    PnPResult,
    PnPSolver,
    _FAILED,
    _ippe_candidates_single,
)

SearchMode = Literal["auto", "exhaustive", "beam"]


class TopKHypothesis(NamedTuple):
    rank: int
    gate_ids: tuple[int, ...]
    assignments: list[GateAssignment]
    score: float
    spread_m: float
    mean_candidate_rmse_px: float
    max_candidate_rmse_px: float
    candidate_labels: str
    pose: PnPResult
    confidence_to_next: float


class TopKResult(NamedTuple):
    success: bool
    hypotheses: list[TopKHypothesis]
    runtime_ms: float
    timed_out: bool
    search_mode_used: str
    n_hypotheses_total_estimated: int
    n_hypotheses_evaluated: int
    n_pose_solves: int
    reason: str


class _SingleGateCandidate(NamedTuple):
    gate_id: int
    label: str
    pos_world: np.ndarray
    rmse_px: float


class _FastHypothesis(NamedTuple):
    gate_ids: tuple[int, ...]
    score: float
    spread_m: float
    mean_rmse_px: float
    max_rmse_px: float
    candidate_labels: str


class TopKHypothesisGenerator:
    """Generate top-K ID assignments within a real-time budget."""

    def __init__(
        self,
        solver: Optional[PnPSolver] = None,
        *,
        top_k: int = 10,
        search_mode: SearchMode = "auto",
        per_detection_top_n: int = 5,
        beam_width: int = 16,
        force_beam_for_detections: int = 4,
        max_exhaustive_hypotheses: int = 1000,
        max_candidate_ids: int = 13,
        max_ippe_candidates_per_pair: int = 3,
        partial_spread_prune_m: float = 8.0,
        pairwise_compatibility_m: float = 3.0,
        max_pose_solves: int = 6,
        max_pose_solves_many_detections: int = 1,
        many_detections_threshold: int = 4,
        time_budget_ms: float = 200.0,
        min_y_m: float = -0.5,
        spread_weight: float = 100.0,
    ):
        self.solver = solver or PnPSolver()
        self.top_k = top_k
        self.search_mode = search_mode
        self.per_detection_top_n = per_detection_top_n
        self.beam_width = beam_width
        self.force_beam_for_detections = force_beam_for_detections
        self.max_exhaustive_hypotheses = max_exhaustive_hypotheses
        self.max_candidate_ids = max_candidate_ids
        self.max_ippe_candidates_per_pair = max_ippe_candidates_per_pair
        self.partial_spread_prune_m = partial_spread_prune_m
        self.pairwise_compatibility_m = pairwise_compatibility_m
        self.max_pose_solves = max_pose_solves
        self.max_pose_solves_many_detections = max_pose_solves_many_detections
        self.many_detections_threshold = many_detections_threshold
        self.time_budget_ms = time_budget_ms
        self.min_y_m = min_y_m
        self.spread_weight = spread_weight

    def generate(
        self,
        detections: list[GateDetection | np.ndarray],
        *,
        candidate_gate_ids: Optional[list[int]] = None,
        candidate_gate_ids_by_detection: Optional[list[list[int]]] = None,
    ) -> TopKResult:
        t0 = time.perf_counter()
        deadline = t0 + self.time_budget_ms / 1000.0
        timed_out = False

        valid = self._normalize_detections(detections)
        if not valid:
            return self._result([], t0, False, "none", 0, 0, 0, "no valid detections")

        per_det_gate_ids = self._candidate_ids_by_detection(
            len(valid),
            candidate_gate_ids,
            candidate_gate_ids_by_detection,
        )
        if not per_det_gate_ids or any(not ids for ids in per_det_gate_ids):
            return self._result([], t0, False, "none", 0, 0, 0, "empty candidate gate list")
        all_gate_ids = sorted({gid for ids in per_det_gate_ids for gid in ids})

        total_est = self._estimate_hypotheses(per_det_gate_ids)
        if total_est == 0:
            return self._result([], t0, False, "none", total_est, 0, 0, "not enough candidate IDs")

        candidate_table = self._precompute_candidates(valid, per_det_gate_ids)
        if time.perf_counter() >= deadline:
            timed_out = True

        mode = self._choose_mode(total_est, len(valid))
        if mode == "exhaustive":
            fast, evaluated, timed_out = self._run_exhaustive(
                valid, per_det_gate_ids, candidate_table, deadline, timed_out
            )
        else:
            fast, evaluated, timed_out = self._run_beam(
                valid, per_det_gate_ids, candidate_table, deadline, timed_out
            )

        fast = self._dedupe_fast(fast)
        fast.sort(key=lambda h: h.score)
        fast = fast[: self.top_k]

        hypotheses, n_pose_solves, timed_out = self._attach_poses(
            fast, valid, deadline, timed_out
        )
        reason = "ok" if hypotheses else "no hypotheses generated"
        return self._result(
            hypotheses,
            t0,
            timed_out,
            mode,
            total_est,
            evaluated,
            n_pose_solves,
            reason,
        )

    def _choose_mode(self, total_est: int, n_detections: int) -> str:
        if self.search_mode == "exhaustive":
            return "exhaustive"
        if self.search_mode == "beam":
            return "beam"
        if n_detections >= self.force_beam_for_detections:
            return "beam"
        return "exhaustive" if total_est <= self.max_exhaustive_hypotheses else "beam"

    def _normalize_detections(self, detections: list[GateDetection | np.ndarray]) -> list[np.ndarray]:
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
        return [int(gid) for gid in candidate_gate_ids if int(gid) in self.solver.gate_map.gates]

    def _candidate_ids_by_detection(
        self,
        n_detections: int,
        candidate_gate_ids: Optional[list[int]],
        candidate_gate_ids_by_detection: Optional[list[list[int]]],
    ) -> list[list[int]]:
        if candidate_gate_ids_by_detection is not None:
            out = []
            for ids in candidate_gate_ids_by_detection[:n_detections]:
                clean = [
                    int(gid)
                    for gid in ids
                    if int(gid) in self.solver.gate_map.gates
                ]
                out.append(clean[: self.max_candidate_ids])
            return out

        ids = self._candidate_ids(candidate_gate_ids)
        if len(ids) > self.max_candidate_ids and candidate_gate_ids is None:
            ids = ids[: self.max_candidate_ids]
        return [ids for _ in range(n_detections)]

    @staticmethod
    def _estimate_hypotheses(per_det_gate_ids: list[list[int]]) -> int:
        count = 0
        for gate_tuple in itertools.product(*per_det_gate_ids):
            if len(set(gate_tuple)) == len(gate_tuple):
                count += 1
        return count

    def _precompute_candidates(
        self,
        detections: list[np.ndarray],
        per_det_gate_ids: list[list[int]],
    ) -> list[dict[int, list[_SingleGateCandidate]]]:
        table: list[dict[int, list[_SingleGateCandidate]]] = []
        for det_i, det in enumerate(detections):
            per_gate: dict[int, list[_SingleGateCandidate]] = {}
            for gate_id in per_det_gate_ids[det_i]:
                pts3 = self.solver.gate_map.get_corners(gate_id).astype(np.float64)
                raw = _ippe_candidates_single(pts3, det, self.solver.K, self.solver.dist)
                cands = [
                    _SingleGateCandidate(gate_id, f"{perm}:c{idx}", pos, float(rmse))
                    for perm, idx, _, pos, _, rmse in raw
                    if pos[1] >= self.min_y_m
                ]
                if cands:
                    per_gate[gate_id] = cands[: self.max_ippe_candidates_per_pair]
            table.append(per_gate)
        return table

    def _run_exhaustive(
        self,
        detections: list[np.ndarray],
        per_det_gate_ids: list[list[int]],
        table: list[dict[int, list[_SingleGateCandidate]]],
        deadline: float,
        timed_out: bool,
    ) -> tuple[list[_FastHypothesis], int, bool]:
        out = []
        evaluated = 0
        for gate_tuple in itertools.product(*per_det_gate_ids):
            if len(set(gate_tuple)) != len(gate_tuple):
                continue
            if time.perf_counter() >= deadline:
                return out, evaluated, True
            hyp = self._score_gate_tuple(gate_tuple, table)
            evaluated += 1
            if hyp is not None:
                out.append(hyp)
        return out, evaluated, timed_out

    def _run_beam(
        self,
        detections: list[np.ndarray],
        per_det_gate_ids: list[list[int]],
        table: list[dict[int, list[_SingleGateCandidate]]],
        deadline: float,
        timed_out: bool,
    ) -> tuple[list[_FastHypothesis], int, bool]:
        per_det_ids = []
        for det_i in range(len(detections)):
            ids_with_candidates = []
            for gate_id in per_det_gate_ids[det_i]:
                cands = table[det_i].get(gate_id, []) if det_i < len(table) else []
                if cands:
                    ids_with_candidates.append(gate_id)
            per_det_ids.append(ids_with_candidates[: self.per_detection_top_n])
        compatibility = self._build_pairwise_compatibility(per_det_ids, table)

        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        evaluated = 0
        for det_i, ids_for_det in enumerate(per_det_ids):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for partial, _ in beams:
                used = set(partial)
                for gate_id in ids_for_det:
                    if gate_id in used:
                        continue
                    if not self._is_pairwise_compatible(partial, det_i, gate_id, compatibility):
                        continue
                    if time.perf_counter() >= deadline:
                        complete = [
                            self._score_gate_tuple(b[0], table)
                            for b in beams
                            if len(b[0]) == len(detections)
                        ]
                        return [h for h in complete if h is not None], evaluated, True
                    new_tuple = partial + (gate_id,)
                    hyp = self._score_partial_tuple(new_tuple, table)
                    evaluated += 1
                    if hyp is not None:
                        if (
                            len(new_tuple) >= 2
                            and hyp.spread_m > self.partial_spread_prune_m
                        ):
                            continue
                        expanded.append((new_tuple, hyp.score))
            expanded.sort(key=lambda x: x[1])
            beams = expanded[: self.beam_width]
            if not beams:
                break

        out = []
        for gate_tuple, _ in beams:
            hyp = self._score_gate_tuple(gate_tuple, table)
            if hyp is not None:
                out.append(hyp)
        return out, evaluated, timed_out

    def _build_pairwise_compatibility(
        self,
        per_det_ids: list[list[int]],
        table: list[dict[int, list[_SingleGateCandidate]]],
    ) -> dict[tuple[int, int, int, int], bool]:
        compatibility: dict[tuple[int, int, int, int], bool] = {}
        for det_a, ids_a in enumerate(per_det_ids):
            for det_b in range(det_a + 1, len(per_det_ids)):
                for gate_a in ids_a:
                    cands_a = table[det_a].get(gate_a, [])
                    if not cands_a:
                        continue
                    pos_a = np.array([c.pos_world for c in cands_a], dtype=np.float64)
                    for gate_b in per_det_ids[det_b]:
                        cands_b = table[det_b].get(gate_b, [])
                        if not cands_b:
                            continue
                        pos_b = np.array([c.pos_world for c in cands_b], dtype=np.float64)
                        dists = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)
                        ok = bool(float(np.min(dists)) <= self.pairwise_compatibility_m)
                        compatibility[(det_a, gate_a, det_b, gate_b)] = ok
                        compatibility[(det_b, gate_b, det_a, gate_a)] = ok
        return compatibility

    @staticmethod
    def _is_pairwise_compatible(
        partial: tuple[int, ...],
        det_i: int,
        gate_id: int,
        compatibility: dict[tuple[int, int, int, int], bool],
    ) -> bool:
        for prev_det_i, prev_gate_id in enumerate(partial):
            if not compatibility.get((prev_det_i, prev_gate_id, det_i, gate_id), False):
                return False
        return True

    def _score_partial_tuple(
        self,
        gate_tuple: tuple[int, ...],
        table: list[dict[int, list[_SingleGateCandidate]]],
    ) -> Optional[_FastHypothesis]:
        return self._score_gate_tuple(gate_tuple, table)

    def _score_gate_tuple(
        self,
        gate_tuple: tuple[int, ...],
        table: list[dict[int, list[_SingleGateCandidate]]],
    ) -> Optional[_FastHypothesis]:
        per_detection = []
        for det_i, gate_id in enumerate(gate_tuple):
            if det_i >= len(table):
                return None
            cands = table[det_i].get(gate_id)
            if not cands:
                return None
            per_detection.append(cands)

        best_spread = float("inf")
        best_mean = float("inf")
        best_max = float("inf")
        best_labels = ""

        pos_arrays = [
            np.array([c.pos_world for c in cands], dtype=np.float64)
            for cands in per_detection
        ]
        seeds = np.vstack(pos_arrays)
        for seed in seeds:
            nearest_idx = [
                int(np.argmin(np.linalg.norm(pos - seed, axis=1)))
                for pos in pos_arrays
            ]
            positions = np.array(
                [pos_arrays[i][idx] for i, idx in enumerate(nearest_idx)],
                dtype=np.float64,
            )
            center = np.median(positions, axis=0)

            # Refine once around the robust center found from the seed assignment.
            refined_idx = [
                int(np.argmin(np.linalg.norm(pos - center, axis=1)))
                for pos in pos_arrays
            ]
            positions = np.array(
                [pos_arrays[i][idx] for i, idx in enumerate(refined_idx)],
                dtype=np.float64,
            )
            center = np.median(positions, axis=0)

            spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
            if spread < best_spread:
                refined = [
                    per_detection[i][idx]
                    for i, idx in enumerate(refined_idx)
                ]
                best_spread = spread
                best_mean = float(np.mean([c.rmse_px for c in refined]))
                best_max = float(np.max([c.rmse_px for c in refined]))
                best_labels = "+".join(c.label for c in refined)

        score = best_spread * self.spread_weight
        return _FastHypothesis(gate_tuple, score, best_spread, best_mean, best_max, best_labels)

    def _dedupe_fast(self, fast: list[_FastHypothesis]) -> list[_FastHypothesis]:
        best_by_ids: dict[tuple[int, ...], _FastHypothesis] = {}
        for hyp in fast:
            prev = best_by_ids.get(hyp.gate_ids)
            if prev is None or hyp.score < prev.score:
                best_by_ids[hyp.gate_ids] = hyp
        return list(best_by_ids.values())

    def _attach_poses(
        self,
        fast: list[_FastHypothesis],
        detections: list[np.ndarray],
        deadline: float,
        timed_out: bool,
    ) -> tuple[list[TopKHypothesis], int, bool]:
        out = []
        n_pose = 0
        pose_limit = self.max_pose_solves
        if len(detections) >= self.many_detections_threshold:
            pose_limit = min(pose_limit, self.max_pose_solves_many_detections)
        limit = min(len(fast), self.top_k, pose_limit)
        for rank, hyp in enumerate(fast[:limit], start=1):
            if rank > 1 and time.perf_counter() >= deadline:
                timed_out = True
                pose = _FAILED
            else:
                if time.perf_counter() >= deadline:
                    timed_out = True
                gate_detections = [
                    (gate_id, detections[det_i])
                    for det_i, gate_id in enumerate(hyp.gate_ids)
                ]
                pose = self.solver.solve(
                    gate_detections,
                    refine=len(detections) < self.many_detections_threshold,
                )
                n_pose += 1
            next_score = fast[rank].score if rank < len(fast) else float("inf")
            confidence = self._confidence(hyp.score, next_score)
            assignments = [
                GateAssignment(det_i, int(gate_id), confidence)
                for det_i, gate_id in enumerate(hyp.gate_ids)
            ]
            out.append(
                TopKHypothesis(
                    rank=rank,
                    gate_ids=hyp.gate_ids,
                    assignments=assignments,
                    score=hyp.score,
                    spread_m=hyp.spread_m,
                    mean_candidate_rmse_px=hyp.mean_rmse_px,
                    max_candidate_rmse_px=hyp.max_rmse_px,
                    candidate_labels=hyp.candidate_labels,
                    pose=pose,
                    confidence_to_next=confidence,
                )
            )
        return out, n_pose, timed_out

    @staticmethod
    def _confidence(best_score: float, next_score: float) -> float:
        if not np.isfinite(next_score):
            return 1.0
        margin = max(0.0, next_score - best_score)
        return float(1.0 - math.exp(-margin / 5.0))

    def _result(
        self,
        hypotheses: list[TopKHypothesis],
        t0: float,
        timed_out: bool,
        mode: str,
        total_est: int,
        evaluated: int,
        n_pose_solves: int,
        reason: str,
    ) -> TopKResult:
        return TopKResult(
            success=bool(hypotheses),
            hypotheses=hypotheses,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            timed_out=timed_out,
            search_mode_used=mode,
            n_hypotheses_total_estimated=total_est,
            n_hypotheses_evaluated=evaluated,
            n_pose_solves=n_pose_solves,
            reason=reason,
        )


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
    candidate_from_gt: bool = False,
) -> None:
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)
    gen = TopKHypothesisGenerator()
    print("Top-K hypothesis validation:")
    print(
        f"  {'Frame':30s} {'GT ids':20s} {'rank':>5s} {'top ids':20s} "
        f"{'mode':>10s} {'eval':>5s} {'poses':>5s} {'ms':>8s} {'timeout':>7s}"
    )
    print("  " + "-" * 115)
    for section in sections:
        for frame in data.get(section, []):
            detections, gt_ids, _ = _detections_from_frame(frame)
            candidates = sorted(gt_ids) if candidate_from_gt else None
            result = gen.generate(detections, candidate_gate_ids=candidates)
            top_ids = result.hypotheses[0].gate_ids if result.hypotheses else ()
            rank = "-"
            for hyp in result.hypotheses:
                if hyp.gate_ids == tuple(gt_ids):
                    rank = str(hyp.rank)
                    break
            print(
                f"  {frame['image_filename']:30s} {str(gt_ids):20s} "
                f"{rank:>5s} {str(top_ids):20s} {result.search_mode_used:>10s} "
                f"{result.n_hypotheses_evaluated:5d} {result.n_pose_solves:5d} "
                f"{result.runtime_ms:8.2f} {str(result.timed_out):>7s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time top-K gate-ID hypotheses")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB_JSON)
    parser.add_argument("--candidate-from-gt", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_on_calibration_json(args.calib_json, candidate_from_gt=args.candidate_from_gt)
        return
    parser.error("Use --validate")


if __name__ == "__main__":
    main()

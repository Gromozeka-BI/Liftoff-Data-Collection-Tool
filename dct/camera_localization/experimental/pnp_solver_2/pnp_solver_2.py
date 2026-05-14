#!/usr/bin/env python3
"""Experimental AP3P/LM PnP solver inspired by the CTU reference project."""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from dct.camera_localization.gate_model.gate_model import GateMap, rotation_y

_CAMERA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB = _CAMERA_ROOT / "config" / "camera_calibration.json"
DEFAULT_TRACK = _CAMERA_ROOT / "config" / "track.json"
DEFAULT_CALIB_JSON = _CAMERA_ROOT / "calibration" / "calibration_frames_card.json"

KP_PERMS: dict[str, tuple[int, int, int, int]] = {
    "IDENT": (0, 1, 2, 3),
    "HFLIP": (1, 0, 3, 2),
}


class PnPResult(NamedTuple):
    success: bool
    position_world: np.ndarray
    yaw_deg: float
    quaternion: np.ndarray
    R_world_cam: np.ndarray
    reprojection_rmse_px: float
    n_points: int
    gate_ids: list[int]
    perm_used: str
    details: str = ""


_FAILED = PnPResult(
    False,
    np.zeros(3, dtype=np.float32),
    0.0,
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    np.eye(3, dtype=np.float64),
    float("inf"),
    0,
    [],
    "",
    "failed",
)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = np.array([
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
            0.25 / s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        q = np.array([
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[2, 1] - R[1, 2]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        q = np.array([
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
            (R[0, 2] - R[2, 0]) / s,
        ])
    else:
        s = 2.0 * math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        q = np.array([
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    q = q.astype(np.float32)
    return q / np.linalg.norm(q)


def _camera_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_w2c, _ = cv2.Rodrigues(rvec)
    R_c2w = R_w2c.T
    pos = (-R_c2w @ tvec.reshape(3, 1)).ravel()
    return R_c2w, pos


def _pose_from_camera(R_c2w: np.ndarray, pos_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_w2c = R_c2w.T
    tvec = (-R_w2c @ pos_world.reshape(3, 1)).astype(np.float64)
    rvec, _ = cv2.Rodrigues(R_w2c)
    return rvec, tvec


def _reprojection_rmse(
    pts3: np.ndarray,
    pts2: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        pts3.reshape(-1, 1, 3).astype(np.float64),
        rvec,
        tvec,
        K,
        dist,
    )
    diff = proj.reshape(-1, 2) - pts2.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _quad_area(pts2: np.ndarray) -> float:
    pts = np.asarray(pts2, dtype=np.float64).reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _gate_face_normal(gate_map: GateMap, gate_id: int) -> np.ndarray:
    return (rotation_y(gate_map[gate_id].yaw_deg) @ np.array([0.0, 0.0, 1.0])).astype(np.float64)


def _load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.array(data["camera_matrix"], dtype=np.float64),
        np.array(data["dist_coeffs"], dtype=np.float64),
    )


class PnPSolver2:
    """Prototype solver using AP3P seed hypotheses plus joint LM refinement."""

    def __init__(self, calib_path: Path = DEFAULT_CALIB, track_path: Path = DEFAULT_TRACK):
        self.K, self.dist = _load_calibration(calib_path)
        self.gate_map = GateMap(track_path)

    def solve(self, gate_detections: list[tuple[int, np.ndarray]]) -> PnPResult:
        valid = self._valid_detections(gate_detections)
        if not valid:
            return _FAILED
        if len(valid) == 1:
            return self._solve_single(valid[0])
        return self._solve_multi_known_ids(valid)

    def _valid_detections(
        self,
        gate_detections: list[tuple[int, np.ndarray]],
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        valid = []
        for gate_id, kpts in gate_detections:
            if gate_id not in self.gate_map.gates:
                continue
            pts2 = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
            if pts2.shape != (4, 2):
                continue
            pts3 = self.gate_map.get_corners(gate_id).astype(np.float64)
            valid.append((gate_id, pts3, pts2))
        return valid

    def _solve_single(self, det: tuple[int, np.ndarray, np.ndarray]) -> PnPResult:
        gate_id, pts3, pts2 = det
        candidates = self._ap3p_candidates(gate_id, pts3, pts2)
        if not candidates:
            return _FAILED._replace(details="no AP3P candidates")
        best = min(candidates, key=lambda c: (not c["front"], c["rmse"]))
        return self._result_from_pose(
            best["rvec"],
            best["tvec"],
            [gate_id],
            best["perm"],
            4,
            f"single AP3P {best['details']}",
            best["rmse"],
        )

    def _solve_multi_known_ids(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray]],
    ) -> PnPResult:
        best = None
        best_key = None
        perm_names = list(KP_PERMS)
        seed_order = sorted(
            range(len(valid)),
            key=lambda i: _quad_area(valid[i][2]),
            reverse=True,
        )
        for seed_idx in seed_order:
            seed_gate_id, seed_pts3, seed_pts2 = valid[seed_idx]
            seed_candidates = self._ap3p_candidates(seed_gate_id, seed_pts3, seed_pts2)
            for cand in seed_candidates:
                seed_front_penalty = self._front_penalty(valid, cand["pos"])
                for perm_choice_tuple in product(perm_names, repeat=len(valid)):
                    perm_choice = list(perm_choice_tuple)
                    pts3_all, pts2_all, gate_ids = self._stack_points(valid, perm_choice)
                    coarse_rmse = _reprojection_rmse(
                        pts3_all, pts2_all, cand["rvec"], cand["tvec"], self.K, self.dist
                    )
                    rvec, tvec = self._refine_pose(pts3_all, pts2_all, cand["rvec"], cand["tvec"])
                    refined_rmse = _reprojection_rmse(pts3_all, pts2_all, rvec, tvec, self.K, self.dist)
                    _, refined_pos = _camera_from_pose(rvec, tvec)
                    front_penalty = self._front_penalty(valid, refined_pos)
                    height_penalty = 1 if refined_pos[1] < -0.2 else 0
                    seed_drift = float(np.linalg.norm(refined_pos - cand["pos"]))
                    drift_penalty = 0.0 if seed_drift <= 5.0 else 1.0 + seed_drift
                    key = (
                        front_penalty,
                        height_penalty,
                        drift_penalty,
                        refined_rmse,
                        seed_front_penalty,
                        min(coarse_rmse, 9999.0),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (
                            seed_gate_id,
                            cand,
                            perm_choice,
                            pts3_all,
                            pts2_all,
                            gate_ids,
                            coarse_rmse,
                            rvec,
                            tvec,
                            refined_rmse,
                            seed_drift,
                        )

        if best is None:
            return _FAILED._replace(details="no multi hypothesis")

        seed_gate_id, cand, _, _, _, gate_ids, coarse_rmse, rvec, tvec, refined_rmse, seed_drift = best
        details = (
            f"seed gate {seed_gate_id}, coarse_rmse={coarse_rmse:.2f}px, "
            f"seed_drift={seed_drift:.2f}m"
        )
        return self._result_from_pose(
            rvec,
            tvec,
            gate_ids,
            "+".join(perm_choice),
            len(pts3_all),
            details,
            refined_rmse,
        )

    def _ap3p_candidates(
        self,
        gate_id: int,
        pts3: np.ndarray,
        pts2: np.ndarray,
        max_error_px: float = 25.0,
    ) -> list[dict]:
        candidates = []
        centroid = pts3.mean(axis=0)
        normal = _gate_face_normal(self.gate_map, gate_id)
        for perm_name, perm in KP_PERMS.items():
            img = pts2[list(perm)]
            retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                pts3.reshape(-1, 1, 3),
                img.reshape(-1, 1, 2),
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_AP3P,
            )
            if retval == 0:
                continue
            for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                R_c2w, pos = _camera_from_pose(rvec, tvec)
                rmse = _reprojection_rmse(pts3, img, rvec, tvec, self.K, self.dist)
                front = float(np.dot(pos - centroid, normal)) <= 0.5
                if rmse > max_error_px and not front:
                    continue
                candidates.append({
                    "perm": f"{perm_name}:c{idx}",
                    "perm_name": perm_name,
                    "rvec": rvec,
                    "tvec": tvec,
                    "R_c2w": R_c2w,
                    "pos": pos,
                    "rmse": rmse,
                    "front": front,
                    "details": f"rmse={rmse:.2f}px front={front}",
                })
        if not candidates:
            rvec, tvec = self._solve_iterative(pts3, pts2)
            if rvec is not None:
                R_c2w, pos = _camera_from_pose(rvec, tvec)
                candidates.append({
                    "perm": "ITERATIVE",
                    "perm_name": "IDENT",
                    "rvec": rvec,
                    "tvec": tvec,
                    "R_c2w": R_c2w,
                    "pos": pos,
                    "rmse": _reprojection_rmse(pts3, pts2, rvec, tvec, self.K, self.dist),
                    "front": True,
                    "details": "fallback",
                })
        return candidates

    def _best_perms_for_pose(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray]],
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> list[str]:
        choice = []
        for _, pts3, pts2 in valid:
            scored = []
            for name, perm in KP_PERMS.items():
                rmse = _reprojection_rmse(pts3, pts2[list(perm)], rvec, tvec, self.K, self.dist)
                scored.append((rmse, name))
            choice.append(min(scored)[1])
        return choice

    def _stack_points(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray]],
        perm_choice: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        pts3_all, pts2_all, gate_ids = [], [], []
        for (gate_id, pts3, pts2), perm_name in zip(valid, perm_choice):
            pts3_all.append(pts3)
            pts2_all.append(pts2[list(KP_PERMS[perm_name])])
            gate_ids.append(gate_id)
        return (
            np.concatenate(pts3_all).astype(np.float64),
            np.concatenate(pts2_all).astype(np.float64),
            gate_ids,
        )

    def _front_penalty(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray]],
        pos: np.ndarray,
    ) -> int:
        penalty = 0
        for gate_id, pts3, _ in valid:
            if float(np.dot(pos - pts3.mean(axis=0), _gate_face_normal(self.gate_map, gate_id))) > 0.5:
                penalty += 1
        return penalty

    def _solve_iterative(
        self,
        pts3: np.ndarray,
        pts2: np.ndarray,
        initial: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        kwargs = {}
        if initial is not None:
            kwargs = {"rvec": initial[0], "tvec": initial[1], "useExtrinsicGuess": True}
        ok, rvec, tvec = cv2.solvePnP(
            pts3.reshape(-1, 1, 3),
            pts2.reshape(-1, 1, 2),
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            **kwargs,
        )
        if not ok:
            return None, None
        return rvec, tvec

    def _refine_pose(
        self,
        pts3: np.ndarray,
        pts2: np.ndarray,
        rvec0: np.ndarray,
        tvec0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rvec, tvec = self._solve_iterative(pts3, pts2, (rvec0, tvec0))
        if rvec is None:
            return rvec0, tvec0
        try:
            return cv2.solvePnPRefineLM(
                pts3.reshape(-1, 1, 3),
                pts2.reshape(-1, 1, 2),
                self.K,
                self.dist,
                rvec,
                tvec,
            )
        except cv2.error:
            return rvec, tvec

    def _result_from_pose(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        gate_ids: list[int],
        perm_used: str,
        n_points: int,
        details: str,
        rmse: float | None = None,
    ) -> PnPResult:
        R_c2w, pos = _camera_from_pose(rvec, tvec)
        cam_fwd = R_c2w[:, 2]
        yaw_deg = math.degrees(math.atan2(float(cam_fwd[0]), float(cam_fwd[2])))
        if rmse is None:
            rmse = float("nan")
        return PnPResult(
            True,
            pos.astype(np.float32),
            yaw_deg,
            _rotation_matrix_to_quaternion(R_c2w),
            R_c2w,
            rmse,
            n_points,
            gate_ids,
            perm_used,
            details,
        )


def _detections_from_calibration_frame(frame: dict) -> list[tuple[int, np.ndarray]]:
    detections = []
    for ann in frame["annotations"]:
        pts = np.array([[p["x_px"], p["y_px"]] for p in ann["keypoints"]], dtype=np.float64)
        detections.append((int(ann["gate_id"]), pts))
    return detections


def validate(calib_json_path: Path = DEFAULT_CALIB_JSON) -> None:
    solver = PnPSolver2()
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("fov_frames", []) + data.get("multi_frames", [])
    for frame in frames:
        result = solver.solve(_detections_from_calibration_frame(frame))
        gt = frame["ground_truth"]["position_world"]
        gt_pos = np.array([gt["x"], gt["y"], gt["z"]], dtype=np.float64)
        err = float(np.linalg.norm(result.position_world.astype(np.float64) - gt_pos))
        print(
            f"{frame['image_filename']}: success={result.success} "
            f"pos={np.round(result.position_world, 3).tolist()} "
            f"err={err:.3f}m yaw={result.yaw_deg:.1f} "
            f"rmse={result.reprojection_rmse_px:.2f}px perm={result.perm_used} "
            f"{result.details}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental AP3P/LM PnP solver")
    parser.add_argument("--validate", action="store_true", help="Run on calibration_frames_card.json")
    parser.add_argument("--gate-id", type=int, default=None)
    parser.add_argument("--keypoints", type=float, nargs=8, metavar="V")
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    if args.gate_id is None or args.keypoints is None:
        parser.error("Use --validate or provide --gate-id and 8 --keypoints values")

    pts = np.array(args.keypoints, dtype=np.float64).reshape(4, 2)
    result = PnPSolver2().solve([(args.gate_id, pts)])
    print(result)


if __name__ == "__main__":
    main()

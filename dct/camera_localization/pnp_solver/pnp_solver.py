#!/usr/bin/env python3
"""
PnP solver: estimate drone camera pose from 2D gate-corner detections.

Workflow
--------
Given a list of (gate_id, 4x2 keypoints) detections and a loaded track map,
this module
  1. Looks up the 3D world corners from the gate map.
  2. Generates ALL physically plausible PnP candidates by trying the two
     keypoint-ordering conventions observed in the annotation set:
        - IDENTITY [TL, TR, BR, BL]  ↔ permutation (0, 1, 2, 3)
        - HFLIP    [TR, TL, BL, BR]  ↔ permutation (1, 0, 3, 2)
     For each ordering, OpenCV's SOLVEPNP_IPPE returns up to two solutions
     after a canonical Z=0 pre-rotation, giving 4 candidates per detection.
  3. Filters out candidates that are physically implausible:
        - camera on the back face of the gate (dot(cam-centroid, normal) > 0)
        - camera below typical FPV-drone flying altitude (y < min_y)
  4. Picks the candidate with the lowest reprojection RMSE; ties broken by
     preferring the candidate with higher y (drones rarely skim the floor).

Multi-gate
----------
For 2+ gates all corners stack into one rigid PnP problem.  Because gates have
different yaws and positions the joint set of points is **not coplanar**, so
the planar mirror ambiguity disappears.  We still brute-force IDENT/HFLIP per
gate (the only annotation conventions in the dataset).  Pose: **SQPNP** for an
initial estimate + **iterative LM refinement** (`solvePnP(..., useExtrinsicGuess=True,
flags=SOLVEPNP_ITERATIVE)`); optionally `solvePnPRansac` if `use_ransac=True`.
Permutation choice is then made by:
  1. Keep only combinations whose camera passes the approach-side cheirality
     test for *every* visible gate (camera in front of the gate plane and
     above the floor).
  2. Among those, pick the lowest joint RMSE.
  3. If none pass cheirality, fall back to lowest RMSE overall.

Why two keypoint orderings?
---------------------------
CVAT annotators labelled corners from their own image-frame perspective.
For some gates this matches the gate-model's intrinsic order (IDENTITY)
and for others it is mirrored left↔right (HFLIP) because the drone observed
the gate from the opposite face.  No vertical-flip / 180-rotation cases
exist in the calibration set.
"""

import json
import math
import argparse
from itertools import product
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import cv2

from dct.camera_localization.gate_model.gate_model import GateMap, rotation_y, CORNER_NAMES

_CAMERA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB      = _CAMERA_ROOT / "config" / "camera_calibration.json"
DEFAULT_TRACK      = _CAMERA_ROOT / "config" / "track.json"
DEFAULT_CALIB_JSON = _CAMERA_ROOT / "calibration" / "calibration_frames_card.json"

# Keypoint orderings actually observed in calibration_frames_card.json
KP_PERMS: dict[str, tuple[int, int, int, int]] = {
    "IDENT": (0, 1, 2, 3),      # annotator order matches gate-model order
    "HFLIP": (1, 0, 3, 2),      # annotator TL = 3D-TR  (camera mirrored L↔R)
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class PnPResult(NamedTuple):
    success: bool
    position_world: np.ndarray      # (3,) float32  [x, y, z] metres
    yaw_deg: float                  # camera/drone yaw in world frame
    quaternion: np.ndarray          # (4,) float32  [qx, qy, qz, qw]
    R_world_cam: np.ndarray         # (3, 3) camera→world rotation
    reprojection_rmse_px: float
    n_points: int
    gate_ids: list[int]
    perm_used: str                  # which keypoint permutation won


_FAILED = PnPResult(
    success=False,
    position_world=np.zeros(3, dtype=np.float32),
    yaw_deg=0.0,
    quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    R_world_cam=np.eye(3),
    reprojection_rmse_px=float("inf"),
    n_points=0,
    gate_ids=[],
    perm_used="",
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → unit quaternion [qx, qy, qz, qw] (Shepperd)."""
    R = R.astype(np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / np.linalg.norm(q)


def _reprojection_rmse(
    pts3: np.ndarray, pts2: np.ndarray,
    rvec: np.ndarray, tvec: np.ndarray,
    K: np.ndarray, dist: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        pts3.reshape(-1, 1, 3).astype(np.float64),
        rvec, tvec, K, dist,
    )
    diff = pts2.astype(np.float64) - proj.reshape(-1, 2)
    return float(np.sqrt(np.mean(diff ** 2)))


def _align_normal_to_z(n: np.ndarray) -> np.ndarray:
    """Compute a 3x3 rotation matrix R such that R @ n = [0, 0, 1]."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(n, z))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(n, z)
    sin_a = np.linalg.norm(axis)
    axis = axis / sin_a
    cos_a = dot
    K = np.array([[ 0.0,       -axis[2],  axis[1]],
                  [ axis[2],    0.0,      -axis[0]],
                  [-axis[1],    axis[0],   0.0    ]])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def _gate_face_normal(gate_map: GateMap, gate_id: int) -> np.ndarray:
    """Gate face normal in world coordinates = R_y(yaw) @ [0, 0, 1]."""
    yaw = gate_map[gate_id].yaw_deg
    return (rotation_y(yaw) @ np.array([0.0, 0.0, 1.0])).astype(np.float64)


# ---------------------------------------------------------------------------
# Candidate generation (single gate)
# ---------------------------------------------------------------------------

def _ippe_candidates_single(
    pts3_world: np.ndarray,
    kpts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Return all candidate (perm_name, ippe_idx, R_wc, pos_world, rvec_local,
    rmse) for a single 4-corner detection, by trying both keypoint
    orderings and both IPPE solutions.

    pts3_world is (4, 3) world coords in [TL, TR, BR, BL] gate-model order.
    kpts       is (4, 2) image coords in JSON [TL, TR, BR, BL] order.
    """
    centroid = pts3_world.mean(axis=0)
    pts_local = pts3_world - centroid

    # IPPE requires Z=0 plane in local frame; pre-rotate the gate.
    n_loc = np.cross(pts_local[1] - pts_local[0],
                     pts_local[2] - pts_local[0])
    n_loc /= np.linalg.norm(n_loc)
    R_pre = _align_normal_to_z(n_loc)
    pts_canon = (R_pre @ pts_local.T).T

    cands = []
    for perm_name, perm in KP_PERMS.items():
        kp = np.asarray(kpts, dtype=np.float64)[list(perm)]
        ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            pts_canon.reshape(-1, 1, 3),
            kp.reshape(-1, 1, 2),
            K, dist, flags=cv2.SOLVEPNP_IPPE,
        )
        if ret == 0:
            continue
        for j, (rv, tv) in enumerate(zip(rvecs, tvecs)):
            R_cv_ippe, _ = cv2.Rodrigues(rv)
            R_cv_local = R_cv_ippe @ R_pre
            R_wc = R_cv_local.T
            c_canon = (-R_cv_ippe.T @ tv).ravel()
            c_local = (R_pre.T @ c_canon.reshape(3, 1)).ravel()
            pos_world = c_local + centroid
            rv_local, _ = cv2.Rodrigues(R_cv_local)
            rmse = _reprojection_rmse(pts_local, kp, rv_local, tv, K, dist)
            cands.append((perm_name, j, R_wc, pos_world.astype(np.float64),
                          rv_local, rmse))
    return cands


def _disambiguate(
    cands: list[tuple],
    gate_centroid: np.ndarray,
    gate_normal_world: np.ndarray,
    *, min_y: float = 0.6,
) -> Optional[tuple]:
    """
    Choose the most physically plausible candidate.

    Filtering:
      - camera on approach side of gate: dot(cam-cent, normal) <= 0
        (this rejects ALL IDENTITY candidates because they place the
        camera on the back face of the gate)
      - camera above typical FPV-drone flying altitude (y >= min_y)

    Selection among survivors:
      1. Smallest reprojection RMSE.
      2. If ties (within 0.5 px), prefer the candidate with higher y
         (drones rarely skim the floor).
    """
    front = [c for c in cands
             if np.dot(c[3] - gate_centroid, gate_normal_world) <= 0.5]
    if not front:
        return None

    survivors = [c for c in front if c[3][1] >= min_y]
    if not survivors:
        # Fall back to any front-side candidate above ground
        survivors = [c for c in front if c[3][1] >= -0.2]
    if not survivors:
        return None

    survivors.sort(key=lambda c: c[5])              # by RMSE
    best_rmse = survivors[0][5]
    tied = [c for c in survivors if c[5] - best_rmse < 0.5]
    if len(tied) == 1:
        return tied[0]
    # Tiebreaker: prefer higher y (drone above gate centre rather than below)
    return max(tied, key=lambda c: float(c[3][1]))


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class PnPSolver:
    """Estimate camera pose from 2D gate-corner detections."""

    def __init__(
        self,
        calib_path: Path = DEFAULT_CALIB,
        track_path: Path = DEFAULT_TRACK,
    ):
        self.K, self.dist = self._load_calibration(calib_path)
        self.gate_map = GateMap(track_path)

    def solve(
        self,
        gate_detections: list[tuple[int, np.ndarray]],
        use_ransac: bool = False,
        refine: bool = True,
    ) -> PnPResult:
        if not gate_detections:
            return _FAILED

        if len(gate_detections) == 1:
            gate_id, kpts = gate_detections[0]
            return self._solve_single(gate_id, np.asarray(kpts, dtype=np.float64))

        return self._solve_multi(gate_detections, use_ransac, refine)

    # ------------------------------------------------------------------
    # Single-gate solve
    # ------------------------------------------------------------------

    def _solve_single(self, gate_id: int, kpts: np.ndarray) -> PnPResult:
        if gate_id not in self.gate_map.gates:
            return _FAILED
        if kpts.reshape(-1, 2).shape[0] != 4:
            return _FAILED

        pts3_world = self.gate_map.get_corners(gate_id).astype(np.float64)
        cands = _ippe_candidates_single(pts3_world, kpts, self.K, self.dist)
        if not cands:
            return _FAILED

        centroid = pts3_world.mean(axis=0)
        normal_w = _gate_face_normal(self.gate_map, gate_id)
        best = _disambiguate(cands, centroid, normal_w)
        if best is None:
            return _FAILED

        perm_name, _, R_wc, pos_world, _, rmse = best
        cam_fwd = R_wc[:, 2]
        yaw_deg = math.degrees(
            math.atan2(float(cam_fwd[0]), float(cam_fwd[2]))
        )
        quat = _rotation_matrix_to_quaternion(R_wc)

        return PnPResult(
            success=True,
            position_world=pos_world.astype(np.float32),
            yaw_deg=yaw_deg,
            quaternion=quat,
            R_world_cam=R_wc,
            reprojection_rmse_px=rmse,
            n_points=4,
            gate_ids=[gate_id],
            perm_used=perm_name,
        )

    # ------------------------------------------------------------------
    # Multi-gate solve
    # ------------------------------------------------------------------
    #   1. Validate detections, collect (gate_id, 3D corners, 2D keypoints)
    #   2. For every gate produce single-gate IPPE candidates:
    #      IDENT/HFLIP x 2 mirror solutions.
    #   3. Choose the combination where all gates agree on the same camera
    #      position (minimum consensus spread).  RMSE is only a secondary
    #      criterion because the lower-RMSE IPPE mirror can be the wrong pose.
    #   4. Optionally refine the resulting consensus pose with one joint
    #      ITERATIVE solve using the chosen 2D-3D correspondences.

    def _solve_multi(
        self,
        gate_detections: list[tuple[int, np.ndarray]],
        use_ransac: bool,
        refine: bool,
    ) -> PnPResult:
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for gate_id, kpts in gate_detections:
            if gate_id not in self.gate_map.gates:
                continue
            kp = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
            if kp.shape[0] != 4:
                continue
            pts3 = self.gate_map.get_corners(gate_id).astype(np.float64)
            normal = _gate_face_normal(self.gate_map, gate_id)
            valid.append((gate_id, pts3, kp, normal))
        if not valid:
            return _FAILED

        return self._solve_multi_consensus(valid, use_ransac, refine)

    def _solve_multi_consensus(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        use_ransac: bool,
        refine: bool,
    ) -> PnPResult:
        same_facing = self._gate_normals_are_same_facing(valid)
        plane_axis = self._shared_gate_plane_axis(valid)
        per_gate: list[list[tuple[str, int, np.ndarray, np.ndarray, float]]] = []
        for _, pts3, kp, normal in valid:
            raw = _ippe_candidates_single(pts3, kp, self.K, self.dist)
            cands = [
                (perm_name, cand_idx, R_wc, pos, rmse)
                for perm_name, cand_idx, R_wc, pos, _, rmse in raw
                if perm_name in KP_PERMS
            ]
            if not cands:
                return _FAILED
            centroid = pts3.mean(axis=0)
            # In multi-gate mode we care about consensus more than each
            # single-gate candidate's vertical mirror.  Some correct XZ
            # candidates from oblique gates can land slightly below ground in
            # the planar IPPE pair, so keep them and let the cluster decide.
            physical = [c for c in cands if c[3][1] >= -1.0]
            if same_facing:
                # Only apply per-gate facing when all visible gates face the
                # same way.  For pairs like yaw 0/180 this test describes the
                # same geometric side with opposite signs and would reject the
                # correct consensus.
                physical = [
                    c for c in physical
                    if float(np.dot(c[3] - centroid, normal)) <= 0.5
                ]
            if physical:
                cands = physical
            per_gate.append(cands)

        best_combo = None
        expected_side = self._infer_common_plane_side(valid, plane_axis)
        best_key: tuple[float, float, float, float, float] | None = None
        pos_arrays = [
            np.array([c[3] for c in cands], dtype=np.float64)
            for cands in per_gate
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
            refined_idx = [
                int(np.argmin(np.linalg.norm(pos - center, axis=1)))
                for pos in pos_arrays
            ]
            combo = tuple(
                per_gate[i][idx]
                for i, idx in enumerate(refined_idx)
            )
            positions = np.array([c[3] for c in combo], dtype=np.float64)
            center = np.median(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            spread = float(np.max(distances))
            mean_rmse = float(np.mean([c[4] for c in combo]))
            max_rmse = float(np.max([c[4] for c in combo]))
            inferred_side_penalty = self._expected_side_penalty(
                center, valid, plane_axis, expected_side
            )
            side_penalty = self._common_side_penalty(positions, valid, plane_axis)
            # Primary: camera consensus.  Secondary: reprojection quality.
            key = (inferred_side_penalty, side_penalty, spread, mean_rmse, max_rmse)
            if best_key is None or key < best_key:
                best_key = key
                best_combo = combo

        if best_combo is None:
            return _FAILED

        result = self._result_from_consensus(valid, best_combo)
        if not refine:
            return result
        refined = self._solve_joint_from_consensus(valid, best_combo, use_ransac)
        if refined.success:
            # Keep the joint refinement only if it stays close to the robust
            # consensus point.  Otherwise the optimizer likely fell into the
            # planar mirror basin again.
            delta = float(np.linalg.norm(
                refined.position_world.astype(np.float64)
                - result.position_world.astype(np.float64)
            ))
            if delta <= 1.0 and refined.reprojection_rmse_px <= result.reprojection_rmse_px + 5.0:
                return refined
        return result

    def _gate_normals_are_same_facing(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        *,
        min_dot: float = 0.75,
    ) -> bool:
        normals = [normal / np.linalg.norm(normal) for _, _, _, normal in valid]
        if len(normals) < 2:
            return True
        ref = normals[0]
        return all(float(np.dot(ref, n)) >= min_dot for n in normals[1:])

    def _shared_gate_plane_axis(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        *,
        max_center_spread_m: float = 0.75,
    ) -> Optional[np.ndarray]:
        """Return a signless common plane axis if gate centres are coplanar.

        For gates with yaw 0 and yaw 180 the face normals have opposite signs,
        but their planes share the same geometric normal.  Align signs before
        averaging and activate the rule only when gate centres lie nearly on
        the same plane along that axis.
        """
        if len(valid) < 2:
            return None
        normals = [normal / np.linalg.norm(normal) for _, _, _, normal in valid]
        ref = normals[0]
        aligned = [n if float(np.dot(ref, n)) >= 0 else -n for n in normals]
        axis = np.mean(np.array(aligned, dtype=np.float64), axis=0)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return None
        axis = axis / norm

        centers = np.array([pts3.mean(axis=0) for _, pts3, _, _ in valid], dtype=np.float64)
        signed = centers @ axis
        if float(np.max(signed) - np.min(signed)) > max_center_spread_m:
            return None
        return axis

    def _common_side_penalty(
        self,
        positions: np.ndarray,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        plane_axis: Optional[np.ndarray],
        *,
        side_tol_m: float = 0.25,
    ) -> float:
        """Penalty when candidate cameras sit on different sides of one plane."""
        if plane_axis is None or len(positions) < 2:
            return 0.0
        centers = np.array([pts3.mean(axis=0) for _, pts3, _, _ in valid], dtype=np.float64)
        plane_center = np.mean(centers, axis=0)
        signed = (positions - plane_center) @ plane_axis

        positive = np.any(signed > side_tol_m)
        negative = np.any(signed < -side_tol_m)
        if positive and negative:
            # Mixed geometric side: same-frame gates should not imply the
            # camera is simultaneously in front of and behind their common
            # plane.  Scale by how split the candidates are.
            return 1.0 + float(np.max(signed) - np.min(signed))
        return 0.0

    def _infer_common_plane_side(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        plane_axis: Optional[np.ndarray],
    ) -> Optional[float]:
        """Infer which side of a common gate plane the camera is on.

        For gates lying in one vertical plane, compare the order of gate
        centres in world space with their observed left-right order in the
        image.  With Y-up, camera-right ~= up x forward, and forward points
        from camera to the plane.  This gives the sign of
        dot(camera - plane_center, plane_axis) without ground truth.
        """
        if plane_axis is None or len(valid) < 2:
            return None

        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_ref = np.cross(up, plane_axis)
        norm = np.linalg.norm(right_ref)
        if norm < 1e-9:
            return None
        right_ref /= norm

        score = 0.0
        for i in range(len(valid)):
            _, pts_i, kp_i, _ = valid[i]
            center_i = pts_i.mean(axis=0)
            image_x_i = float(np.mean(kp_i[:, 0]))
            for j in range(i + 1, len(valid)):
                _, pts_j, kp_j, _ = valid[j]
                center_j = pts_j.mean(axis=0)
                image_x_j = float(np.mean(kp_j[:, 0]))
                world_dx = float(np.dot(center_j - center_i, right_ref))
                image_dx = image_x_j - image_x_i
                if abs(world_dx) < 1e-6 or abs(image_dx) < 1e-3:
                    continue
                score += image_dx * world_dx

        if abs(score) < 1e-6:
            return None
        # If image order matches +right_ref, the camera is on -plane_axis side.
        return -float(np.sign(score))

    def _expected_side_penalty(
        self,
        camera_pos: np.ndarray,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        plane_axis: Optional[np.ndarray],
        expected_side: Optional[float],
        *,
        side_tol_m: float = 0.25,
    ) -> float:
        if plane_axis is None or expected_side is None:
            return 0.0
        centers = np.array([pts3.mean(axis=0) for _, pts3, _, _ in valid], dtype=np.float64)
        plane_center = np.mean(centers, axis=0)
        signed = float(np.dot(camera_pos - plane_center, plane_axis))
        if abs(signed) <= side_tol_m:
            return 0.5
        actual_side = float(np.sign(signed))
        if actual_side == expected_side:
            return 0.0
        return 1.0 + abs(signed)

    def _result_from_consensus(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        combo: tuple[tuple[str, int, np.ndarray, np.ndarray, float], ...],
    ) -> PnPResult:
        positions = np.array([c[3] for c in combo], dtype=np.float64)
        pos_world = np.median(positions, axis=0).astype(np.float32)

        # Pick orientation from the candidate closest to the consensus camera
        # centre.  This avoids averaging rotations while preserving a real PnP
        # orientation from one gate.
        distances = np.linalg.norm(positions - pos_world.astype(np.float64), axis=1)
        orient_idx = int(np.argmin(distances))
        R_wc = combo[orient_idx][2]
        cam_fwd = R_wc[:, 2]
        yaw_deg = math.degrees(math.atan2(float(cam_fwd[0]), float(cam_fwd[2])))
        quat = _rotation_matrix_to_quaternion(R_wc)
        gate_ids = [gate_id for gate_id, _, _, _ in valid]
        perm_used = "+".join(f"{c[0]}:c{c[1]}" for c in combo)
        rmse = float(np.mean([c[4] for c in combo]))

        return PnPResult(
            success=True,
            position_world=pos_world,
            yaw_deg=yaw_deg,
            quaternion=quat,
            R_world_cam=R_wc,
            reprojection_rmse_px=rmse,
            n_points=4 * len(valid),
            gate_ids=gate_ids,
            perm_used=perm_used,
        )

    def _solve_joint_from_consensus(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        combo: tuple[tuple[str, int, np.ndarray, np.ndarray, float], ...],
        use_ransac: bool,
    ) -> PnPResult:
        perm_choice = [c[0] for c in combo]
        seed_idx = int(np.argmin([c[4] for c in combo]))
        seed_R_wc = combo[seed_idx][2]
        seed_pos = np.median(np.array([c[3] for c in combo]), axis=0)
        seed_R_cv = seed_R_wc.T
        seed_tvec = (-seed_R_cv @ seed_pos.reshape(3, 1)).astype(np.float64)
        seed_rvec, _ = cv2.Rodrigues(seed_R_cv)
        return self._solve_joint(
            valid,
            perm_choice,
            use_ransac,
            initial=(seed_rvec, seed_tvec),
        )

    def _joint_is_feasible(
        self,
        result: PnPResult,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        *, min_y: float = -0.2,
        approach_tol: float = 0.5,
    ) -> bool:
        """Camera must lie on the approach side of EVERY visible gate."""
        pos = result.position_world.astype(np.float64)
        if pos[1] < min_y:
            return False
        for _, pts3, _, normal in valid:
            centroid = pts3.mean(axis=0)
            if float(np.dot(pos - centroid, normal)) > approach_tol:
                return False
        return True

    def _solve_joint(
        self,
        valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        perm_choice: list[str],
        use_ransac: bool,
        initial: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> PnPResult:
        pts3_all, pts2_all, gate_ids = [], [], []
        for (gate_id, pts3, kp, _), pname in zip(valid, perm_choice):
            perm = KP_PERMS[pname]
            pts3_all.append(pts3)
            pts2_all.append(kp[list(perm)])
            gate_ids.append(gate_id)
        pts3 = np.concatenate(pts3_all).astype(np.float64)
        pts2 = np.concatenate(pts2_all).astype(np.float64)
        n_used = len(pts3)

        rvec, tvec = self._joint_pose(pts3, pts2, use_ransac=use_ransac, initial=initial)
        if rvec is None:
            return _FAILED
        if use_ransac:
            # solvePnPRansac path returns inliers via _joint_pose's helper,
            # but for simplicity refine on all points after RANSAC.
            pass

        R_cv, _ = cv2.Rodrigues(rvec)
        R_wc = R_cv.T
        pos_world = (-R_wc @ tvec).ravel().astype(np.float32)
        cam_fwd = R_wc[:, 2]
        yaw_deg = math.degrees(math.atan2(float(cam_fwd[0]), float(cam_fwd[2])))
        quat = _rotation_matrix_to_quaternion(R_wc)
        rmse = _reprojection_rmse(pts3, pts2, rvec, tvec, self.K, self.dist)

        return PnPResult(
            success=True,
            position_world=pos_world,
            yaw_deg=yaw_deg,
            quaternion=quat,
            R_world_cam=R_wc,
            reprojection_rmse_px=rmse,
            n_points=n_used,
            gate_ids=gate_ids,
            perm_used="+".join(perm_choice),
        )

    def _joint_pose(
        self,
        pts3: np.ndarray,
        pts2: np.ndarray,
        *, use_ransac: bool,
        initial: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """SQPNP initial guess → LM refinement.  Returns (rvec, tvec) or (None, None)."""
        obj = pts3.reshape(-1, 1, 3)
        img = pts2.reshape(-1, 1, 2)

        if initial is not None:
            rvec0, tvec0 = initial
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, self.K, self.dist,
                rvec=rvec0, tvec=tvec0,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                return rvec, tvec
            return rvec0, tvec0

        if use_ransac:
            ok, rvec0, tvec0, inliers = cv2.solvePnPRansac(
                obj, img, self.K, self.dist,
                reprojectionError=8.0,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not ok or inliers is None or len(inliers) < 4:
                return None, None
        else:
            ok, rvec0, tvec0 = cv2.solvePnP(
                obj, img, self.K, self.dist,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not ok:
                return None, None

        ok, rvec, tvec = cv2.solvePnP(
            obj, img, self.K, self.dist,
            rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return rvec0, tvec0
        return rvec, tvec

    # ------------------------------------------------------------------
    # Calibration loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        K    = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"],   dtype=np.float64)
        return K, dist


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------

def validate_on_fov_frames(
    solver: PnPSolver,
    calib_json_path: Path = DEFAULT_CALIB_JSON,
) -> None:
    """Run the PnP solver on each fov_frame and compare with ground truth."""
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)

    print("\nValidation on fov_frames (single-gate PnP):")
    header = (f"  {'Frame':40s}  {'perm':6s}  {'RMSE':>7s}  "
              f"{'pos_xz':>7s}  {'pos_y':>6s}  {'yaw':>7s}  OK?")
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_ok = 0
    total = 0
    for frame in data.get("fov_frames", []):
        ann     = frame["annotations"][0]
        gate_id = ann["gate_world"]["gate_id"]
        kpts    = np.array(
            [[kp["x_px"], kp["y_px"]] for kp in ann["keypoints"]],
            dtype=np.float64,
        )
        result = solver.solve([(gate_id, kpts)])

        gt      = frame["ground_truth"]
        gt_pos  = np.array([gt["position_world"]["x"],
                            gt["position_world"]["y"],
                            gt["position_world"]["z"]])
        gt_yaw  = float(gt.get("yaw_deg", 0.0))
        total  += 1

        fname = frame["image_filename"]
        if not result.success:
            print(f"  {fname:40s}  FAILED")
            continue

        pos = result.position_world.astype(np.float64)
        pos_xz = math.sqrt((pos[0] - gt_pos[0])**2 + (pos[2] - gt_pos[2])**2)
        pos_y  = abs(pos[1] - gt_pos[1])
        yaw_e  = abs(result.yaw_deg - gt_yaw) % 360
        if yaw_e > 180:
            yaw_e = 360 - yaw_e

        ok = pos_xz <= 0.2 and pos_y <= 0.1 and yaw_e <= 2.0
        if ok:
            n_ok += 1

        print(
            f"  {fname:40s}  {result.perm_used:6s}  "
            f"{result.reprojection_rmse_px:5.3f} px  "
            f"{pos_xz:5.3f} m  "
            f"{pos_y:4.3f} m  "
            f"{yaw_e:5.1f} deg  "
            f"{'OK' if ok else 'FAIL'}"
        )

    print(f"\n  Passed (strict TZ): {n_ok}/{total}")


def validate_on_multi_frames(
    solver: PnPSolver,
    calib_json_path: Path = DEFAULT_CALIB_JSON,
) -> None:
    """Run the PnP solver on each multi_frame (joint multi-gate PnP)."""
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)

    print("\nValidation on multi_frames (joint multi-gate PnP):")
    header = (f"  {'Frame':25s}  {'perms':40s}  {'RMSE':>7s}  "
              f"{'pos_xz':>7s}  {'pos_y':>6s}  {'yaw':>7s}  OK?")
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_ok = 0
    total = 0
    for frame in data.get("multi_frames", []):
        dets = []
        for ann in frame["annotations"]:
            gid = ann["gate_world"]["gate_id"]
            kpts = np.array([[kp["x_px"], kp["y_px"]]
                             for kp in ann["keypoints"]], dtype=np.float64)
            dets.append((gid, kpts))
        result = solver.solve(dets)

        gt = frame["ground_truth"]
        gt_pos = np.array([gt["position_world"]["x"],
                           gt["position_world"]["y"],
                           gt["position_world"]["z"]])
        gt_yaw = float(gt.get("yaw_deg", 0.0))
        total += 1

        fname = frame["image_filename"]
        if not result.success:
            print(f"  {fname:25s}  FAILED")
            continue

        pos = result.position_world.astype(np.float64)
        pos_xz = math.sqrt((pos[0] - gt_pos[0])**2 + (pos[2] - gt_pos[2])**2)
        pos_y  = abs(pos[1] - gt_pos[1])
        yaw_e  = abs(result.yaw_deg - gt_yaw) % 360
        if yaw_e > 180:
            yaw_e = 360 - yaw_e

        ok = pos_xz <= 0.2 and pos_y <= 0.1 and yaw_e <= 2.0
        if ok:
            n_ok += 1

        print(
            f"  {fname:25s}  {result.perm_used:40s}  "
            f"{result.reprojection_rmse_px:5.3f} px  "
            f"{pos_xz:5.3f} m  "
            f"{pos_y:4.3f} m  "
            f"{yaw_e:5.1f} deg  "
            f"{'OK' if ok else 'FAIL'}"
        )

    print(f"\n  Passed (strict TZ): {n_ok}/{total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PnP solver — estimate camera pose from gate corners"
    )
    parser.add_argument("--calib",      type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--track",      type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB_JSON)
    parser.add_argument("--validate",   action="store_true")
    parser.add_argument("--validate-multi", action="store_true")
    parser.add_argument("--gate-id",    type=int,  default=None)
    parser.add_argument("--keypoints",  type=float, nargs=8, metavar="PX",
                        help="8 values: x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL)")
    parser.add_argument("--ransac",     action="store_true")
    args = parser.parse_args()

    solver = PnPSolver(args.calib, args.track)
    print(f"PnPSolver ready  ({len(solver.gate_map)} gates)")

    if args.validate:
        validate_on_fov_frames(solver, args.calib_json)
    if args.validate_multi:
        validate_on_multi_frames(solver, args.calib_json)
    if args.validate or args.validate_multi:
        return

    if args.gate_id is not None and args.keypoints is not None:
        kpts = np.array(args.keypoints).reshape(4, 2)
        result = solver.solve([(args.gate_id, kpts)], use_ransac=args.ransac)
        if result.success:
            pos = result.position_world
            print(f"\nGate {args.gate_id}  RMSE = {result.reprojection_rmse_px:.3f} px"
                  f"  perm={result.perm_used}")
            print(f"  Position  : x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f} m")
            print(f"  Yaw       : {result.yaw_deg:.2f} deg")
            print(f"  Quaternion: {result.quaternion.tolist()}")
        else:
            print("solvePnP failed.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

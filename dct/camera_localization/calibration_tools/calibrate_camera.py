#!/usr/bin/env python3
"""
Camera calibration from annotated gate corners.

Reads 2D-3D keypoint correspondences from calibration_frames_card.json
and estimates the camera intrinsic matrix K plus distortion coefficients
using cv2.calibrateCamera.

Strategy
--------
* ``fov_frames`` (single-gate) form the primary calibration set.
* Additional multi-gate frames listed in ``EXTRA_CALIB_FROM_MULTI`` are folded
  into that set after parsing (validated annotations; avoids polluting ``K``
  with all noisy multi_views).
* Remaining ``multi_frames`` are used only for forward validation.
* 3D object points are centred per-frame before passing to calibrateCamera.
  The K matrix is scale/translation-invariant (projection is homogeneous in
  the 3D point), so this normalisation does not change K but dramatically
  improves the numerical conditioning of the Levenberg-Marquardt solver when
  world coordinates span tens of metres.
* Initial K guess (analytically derived from frame_000107, gate 0):
    fx ~ 300 px (FOV ~ 130 deg), cx ~ 616, cy ~ 284

Input : calibration/calibration_frames_card.json
Output: config/camera_calibration.json
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CAMERA_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CALIB_JSON = CAMERA_ROOT / "calibration" / "calibration_frames_card.json"
DEFAULT_OUTPUT = CAMERA_ROOT / "config" / "camera_calibration.json"

IMAGE_SIZE = (1280, 720)  # (width, height) — fixed for this dataset

# Multi-gate image filenames promoted from ``multi_frames`` into the calibration
# pool (same 2D-3D stacking as ``parse_frames``). Keep this list short: bad
# multi-gate annotations distort ``K``.
EXTRA_CALIB_FROM_MULTI: tuple[str, ...] = ("frame_000158.jpg",)

# Fixed order matching the 3D gate model convention
CORNER_ORDER = [
    "inner_top_left",
    "inner_top_right",
    "inner_bottom_right",
    "inner_bottom_left",
]

# Initial K estimate derived geometrically from frame_000107 (gate 0, yaw~90 deg):
#   TL=(545,299) TR=(687,295) BR=(684,424) BL=(559,431)
#   Gate corners span 1.56m at ~3.3m depth.
#   Horizontal: 142px / (1.56m/3.3m) => fx ~ 300 px  (FOV ~ 130 deg)
#   Vertical:   solving TL/BL pair for fy gives cy ~ 284, fy ~ 283 px
#   Horizontal centre cx: midpoint of symmetric TL/TR => cx ~ 616 px
K_INIT_GUESS = np.array(
    [
        [300.0, 0.0, 616.0],
        [0.0, 300.0, 284.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_correspondences(calib_json_path: Path):
    """
    Parse calibration_frames_card.json.

    Returns
    -------
    fov_obj  : list[ndarray (N,3) float32]  object points for fov_frames
    fov_img  : list[ndarray (N,2) float32]  image  points for fov_frames
    fov_gt   : list[dict]                   ground_truth dicts for fov_frames
    fov_names: list[str]
    multi_obj: list[ndarray (N,3) float32]  object points for multi_frames
    multi_img: list[ndarray (N,2) float32]  image  points for multi_frames
    multi_names: list[str]
    """
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)

    def parse_frames(frame_list):
        obj_list, img_list, gt_list, name_list = [], [], [], []
        for frame in frame_list:
            pts_3d, pts_2d = [], []
            for ann in frame["annotations"]:
                cw = ann["gate_world"]["corners_world"]
                kp_map = {kp["name"]: (kp["x_px"], kp["y_px"])
                          for kp in ann["keypoints"]}
                for name in CORNER_ORDER:
                    pts_3d.append(cw[name])
                    pts_2d.append(list(kp_map[name]))
            obj_list.append(np.array(pts_3d, dtype=np.float32))
            img_list.append(np.array(pts_2d, dtype=np.float32))
            gt_list.append(frame.get("ground_truth", {}))
            name_list.append(frame["image_filename"])
        return obj_list, img_list, gt_list, name_list

    fov_obj, fov_img, fov_gt, fov_names = parse_frames(data.get("fov_frames", []))
    multi_obj, multi_img, _, multi_names = parse_frames(data.get("multi_frames", []))

    return fov_obj, fov_img, fov_gt, fov_names, multi_obj, multi_img, multi_names


def promote_multi_frames_to_calibration(
    fov_obj: list[np.ndarray],
    fov_img: list[np.ndarray],
    fov_names: list[str],
    multi_obj: list[np.ndarray],
    multi_img: list[np.ndarray],
    multi_names: list[str],
    filenames: tuple[str, ...],
) -> None:
    """
    Move the given ``multi_frames`` entries into ``fov_*`` lists in place.

    Removal uses descending indices so multiple promotions stay valid.
    """
    if len(filenames) != len(set(filenames)):
        raise ValueError("EXTRA_CALIB_FROM_MULTI contains duplicate filenames")
    missing = [n for n in filenames if n not in multi_names]
    if missing:
        raise ValueError(
            f"Extra calibration frames not found in multi_frames: {missing}"
        )
    indices = sorted(
        (multi_names.index(name) for name in filenames),
        reverse=True,
    )
    for i in indices:
        fov_obj.append(multi_obj.pop(i))
        fov_img.append(multi_img.pop(i))
        fov_names.append(multi_names.pop(i))


def normalize_object_points(obj_pts: list[np.ndarray]) -> list[np.ndarray]:
    """
    Centre each frame's 3D points around their own centroid.

    K is unchanged (scale/translation-invariant), but the internal Jacobian
    of calibrateCamera becomes well-conditioned instead of having ill-scaled
    translation components at world-scale (~50 m) distances.
    """
    return [pts - pts.mean(axis=0) for pts in obj_pts]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_camera(
    obj_pts: list[np.ndarray],
    img_pts: list[np.ndarray],
    image_size: tuple[int, int],
    fix_distortion: bool = True,
):
    """
    Estimate camera intrinsics via cv2.calibrateCamera using centred
    object points for numerical stability.

    Parameters
    ----------
    fix_distortion : bool
        If True (default), forces all distortion to zero (Liftoff simulator
        uses an ideal pinhole lens with no optical aberration).

    Returns
    -------
    rms         : float        overall RMS reprojection error (px)
    K           : ndarray 3x3  camera matrix
    dist_coeffs : ndarray (5,) distortion vector
    rvecs       : list         per-frame rotation vectors
    tvecs       : list         per-frame translation vectors
    """
    dist_init = np.zeros((5, 1), dtype=np.float64)

    # Free all four intrinsic parameters independently.
    # For a virtual camera fx should ~= fy, but not hard-enforcing it lets
    # the optimiser compensate for any slight asymmetry.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    if fix_distortion:
        flags |= (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
        )

    obj_pts_norm = normalize_object_points(obj_pts)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=[p.reshape(-1, 1, 3) for p in obj_pts_norm],
        imagePoints=[p.reshape(-1, 1, 2) for p in img_pts],
        imageSize=image_size,
        cameraMatrix=K_INIT_GUESS.copy(),
        distCoeffs=dist_init,
        flags=flags,
    )
    return rms, K, dist, rvecs, tvecs


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def per_frame_rmse(
    obj_pts: list[np.ndarray],
    img_pts: list[np.ndarray],
    rvecs: list,
    tvecs: list,
    K: np.ndarray,
    dist: np.ndarray,
) -> list[float]:
    """Compute per-frame RMS reprojection error."""
    errors = []
    for pts3, pts2, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rv, tv, K, dist)
        diff = pts2 - proj.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(diff**2))))
    return errors


def validate_fov_frames(
    fov_obj: list[np.ndarray],
    fov_img: list[np.ndarray],
    fov_names: list[str],
    K: np.ndarray,
    dist: np.ndarray,
) -> None:
    """
    Re-run solvePnP with the calibrated K on each fov_frame and report
    the reprojection RMSE.

    This is an independent check: calibrateCamera used centred object
    points while this validation uses the original world coordinates.
    If the two agree closely, K is self-consistent.

    Note on position accuracy: for 4 coplanar gate points, solvePnP
    has a mirror ambiguity (two solutions with identical reprojection
    error).  Disambiguation — choosing the solution where the camera is
    in front of the gate — is handled by the pnp_solver module.
    Position validation against ground-truth is therefore deferred to
    that module.
    """
    print("\nForward validation: solvePnP reprojection (world coords, ITERATIVE):")
    header = f"  {'Frame':45s}  {'Reproj RMSE':>11s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for fname, pts3, pts2 in zip(fov_names, fov_obj, fov_img):
        ok, rvec, tvec = cv2.solvePnP(
            pts3.reshape(-1, 1, 3),
            pts2.reshape(-1, 1, 2),
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            print(f"  {fname:45s}  FAILED")
            continue
        proj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec, tvec, K, dist)
        rmse = float(np.sqrt(np.mean((pts2 - proj.reshape(-1, 2)) ** 2)))
        flag = "  <-- check K" if rmse > 3.0 else ""
        print(f"  {fname:45s}  {rmse:9.4f} px{flag}")


def validate_multi_frames(
    multi_obj: list[np.ndarray],
    multi_img: list[np.ndarray],
    multi_names: list[str],
    K: np.ndarray,
    dist: np.ndarray,
) -> None:
    """
    For each multi-gate frame run solvePnP and report reprojection RMSE.
    These frames were NOT used in calibration — low error confirms that K
    generalises to multi-gate scenes. (Frames listed in ``EXTRA_CALIB_FROM_MULTI``
    are calibrated and therefore omitted here.)
    """
    print("\nMulti-gate forward validation (not used in calibration):")
    header = f"  {'Frame':45s}  {'PnP RMSE':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for fname, pts3, pts2 in zip(multi_names, multi_obj, multi_img):
        ok, rvec, tvec = cv2.solvePnP(
            pts3.reshape(-1, 1, 3),
            pts2.reshape(-1, 1, 2),
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            print(f"  {fname:45s}  FAILED")
            continue
        proj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec, tvec, K, dist)
        rmse = float(np.sqrt(np.mean((pts2 - proj.reshape(-1, 2)) ** 2)))
        flag = "  <-- check K" if rmse > 10.0 else ""
        print(f"  {fname:45s}  {rmse:7.3f} px{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from annotated gate corners"
    )
    parser.add_argument(
        "--calib-json",
        type=Path,
        default=DEFAULT_CALIB_JSON,
        help="Path to calibration_frames_card.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write camera_calibration.json",
    )
    parser.add_argument(
        "--allow-distortion",
        action="store_true",
        help="Also estimate radial distortion K1 (off by default for simulator data)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the solvePnP ground-truth validation step",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"Loading correspondences from:\n  {args.calib_json}\n")
    (
        fov_obj, fov_img, _fov_gt, fov_names,
        multi_obj, multi_img, multi_names,
    ) = load_correspondences(args.calib_json)

    if EXTRA_CALIB_FROM_MULTI:
        promote_multi_frames_to_calibration(
            fov_obj,
            fov_img,
            fov_names,
            multi_obj,
            multi_img,
            multi_names,
            EXTRA_CALIB_FROM_MULTI,
        )

    n_fov_pts = sum(len(p) for p in fov_obj)
    print(
        f"  fov_frames + extras: {len(fov_obj)} frames, {n_fov_pts} points  "
        f"(used for calibration)"
    )
    if EXTRA_CALIB_FROM_MULTI:
        print(f"    also from multi_frames: {list(EXTRA_CALIB_FROM_MULTI)}")
    print(
        f"  multi_frames: {len(multi_obj)} frames, "
        f"{sum(len(p) for p in multi_obj)} points  (used for validation)"
    )

    # ------------------------------------------------------------------
    # 2. Calibrate  (fov_frames only)
    # ------------------------------------------------------------------
    fix_dist = not args.allow_distortion
    dist_label = "zero (simulator)" if fix_dist else "K1 free"
    print(f"\nRunning cv2.calibrateCamera [distortion: {dist_label}]")
    print(f"  Initial K guess:\n{K_INIT_GUESS}")

    rms, K, dist_coeffs, rvecs, tvecs = calibrate_camera(
        fov_obj, fov_img, IMAGE_SIZE, fix_distortion=fix_dist
    )

    # ------------------------------------------------------------------
    # 3. Report calibration quality
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Overall calibration RMS : {rms:.4f} px")
    print(f"{'='*55}")
    print(f"\nCamera matrix K:\n{K}")
    print(f"\nDistortion coefficients : {dist_coeffs.ravel()}")
    print(f"\n  fx = {K[0,0]:.2f} px   fy = {K[1,1]:.2f} px")
    print(f"  cx = {K[0,2]:.2f} px   cy = {K[1,2]:.2f} px")
    fov_h = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / 2 / K[0, 0]))
    fov_v = 2 * np.degrees(np.arctan(IMAGE_SIZE[1] / 2 / K[1, 1]))
    print(f"  Horiz FOV ~= {fov_h:.1f} deg   Vert FOV ~= {fov_v:.1f} deg")

    obj_fov_norm = normalize_object_points(fov_obj)
    per_errors = per_frame_rmse(obj_fov_norm, fov_img, rvecs, tvecs, K, dist_coeffs)
    print("\nPer-frame reprojection errors (fov calibration set):")
    for name, err in zip(fov_names, per_errors):
        marker = "  !" if err > 5.0 else ""
        print(f"  {name:45s}  {err:.4f} px{marker}")

    # ------------------------------------------------------------------
    # 4. Forward validation
    # ------------------------------------------------------------------
    if not args.no_validate:
        validate_fov_frames(fov_obj, fov_img, fov_names, K, dist_coeffs)
        validate_multi_frames(multi_obj, multi_img, multi_names, K, dist_coeffs)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if EXTRA_CALIB_FROM_MULTI:
        notes_base = (
            "Calibrated on Liftoff simulator data: all fov_frames plus "
            f"{list(EXTRA_CALIB_FROM_MULTI)} from multi_frames "
            "(2D–3D gate corners from calibration_frames_card.json). "
        )
    else:
        notes_base = (
            "Calibrated on Liftoff simulator data using fov_frames only "
            "(single-gate 2D–3D correspondences from calibration_frames_card.json). "
        )

    result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "image_size": list(IMAGE_SIZE),
        "reprojection_error_px": round(float(rms), 6),
        "calibration_frames": fov_names,
        "n_point_correspondences": n_fov_pts,
        "notes": notes_base
        + f"Distortion model: {dist_label}. "
        "Per-frame 3D centring applied for numerical stability.",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nCalibration saved -> {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gate model: load track map and compute world 3D corner coordinates.

All 3D positions used by the localization pipeline come from this module.
The module can also be used as a standalone script to verify corner
coordinates against the calibration reference data.

Input : config/track.json
Output: per-gate dict with 4 world-space 3D corners (numpy arrays)
"""

import json
import argparse
import math
import numpy as np
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CAMERA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK = CAMERA_ROOT / "config" / "track.json"
DEFAULT_CALIB_JSON = CAMERA_ROOT / "calibration" / "calibration_frames_card.json"

# ---------------------------------------------------------------------------
# Corner names in fixed order (matches PnP keypoint order)
# ---------------------------------------------------------------------------
CORNER_NAMES = [
    "inner_top_left",
    "inner_top_right",
    "inner_bottom_right",
    "inner_bottom_left",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class GateCorners(NamedTuple):
    """Four world-space corners of one gate (float32 numpy arrays)."""
    inner_top_left: np.ndarray      # shape (3,)
    inner_top_right: np.ndarray
    inner_bottom_right: np.ndarray
    inner_bottom_left: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return (4, 3) float32 array in TL → TR → BR → BL order."""
        return np.array(
            [self.inner_top_left,
             self.inner_top_right,
             self.inner_bottom_right,
             self.inner_bottom_left],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {name: list(map(float, arr))
                for name, arr in zip(CORNER_NAMES, self)}


class GateInfo(NamedTuple):
    """Gate metadata + computed corners."""
    gate_id: int
    name: str
    position: np.ndarray    # bottom-center world coords (3,)
    yaw_deg: float          # rotation[1] — yaw around Y
    size: tuple[float, float]   # (width, height) of inner opening
    radius: float
    corners: GateCorners


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def rotation_y(deg: float) -> np.ndarray:
    """
    Right-hand rotation matrix around the world Y axis.

    Convention (Y-up world frame):
        yaw = 0   -> gate face perpendicular to world +Z
        yaw = 90  -> gate face perpendicular to world +X
        yaw = 180 -> gate face perpendicular to world -Z

    Parameters
    ----------
    deg : float  rotation angle in degrees

    Returns
    -------
    R : (3, 3) float64 ndarray
    """
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[ c, 0, s],
         [ 0, 1, 0],
         [-s, 0, c]],
        dtype=np.float64,
    )


def compute_corners(
    position: np.ndarray,
    yaw_deg: float,
    width: float,
    height: float,
) -> GateCorners:
    """
    Compute the four inner corners of a gate in world coordinates.

    Local frame (before yaw rotation), origin at bottom-center:

        inner_top_left    = (-w/2,  h, 0)
        inner_top_right   = ( w/2,  h, 0)
        inner_bottom_right= ( w/2,  0, 0)
        inner_bottom_left = (-w/2,  0, 0)

    Transformation:
        corner_world = position + R_y(yaw_deg) @ corner_local

    Parameters
    ----------
    position : (3,) array  bottom-center world coordinates
    yaw_deg  : float        yaw rotation around Y in degrees
    width    : float        inner opening width  (size[0])
    height   : float        inner opening height (size[1])

    Returns
    -------
    GateCorners named-tuple
    """
    w2 = width / 2.0
    local_corners = np.array(
        [[-w2, height, 0.0],   # TL
         [ w2, height, 0.0],   # TR
         [ w2,    0.0, 0.0],   # BR
         [-w2,    0.0, 0.0]],  # BL
        dtype=np.float64,
    )

    R = rotation_y(yaw_deg)
    world_corners = (R @ local_corners.T).T + position  # (4, 3)

    return GateCorners(
        inner_top_left=world_corners[0].astype(np.float32),
        inner_top_right=world_corners[1].astype(np.float32),
        inner_bottom_right=world_corners[2].astype(np.float32),
        inner_bottom_left=world_corners[3].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Track map loading
# ---------------------------------------------------------------------------

class GateMap:
    """
    Container for all gates on the track.

    Attributes
    ----------
    gates : dict[int, GateInfo]  keyed by gate_id
    """

    def __init__(self, track_json_path: Path = DEFAULT_TRACK):
        self.path = track_json_path
        self.gates: dict[int, GateInfo] = {}
        self._load(track_json_path)

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for g in data["gates"]:
            if "size" not in g:
                continue
            pos = np.array(g["position"], dtype=np.float64)
            yaw = float(g["rotation"][1])          # rotation[1] = yaw around Y
            w, h = float(g["size"][0]), float(g["size"][1])
            corners = compute_corners(pos, yaw, w, h)

            self.gates[g["id"]] = GateInfo(
                gate_id=g["id"],
                name=g.get("name", f"Gate {g['id']}"),
                position=pos.astype(np.float32),
                yaw_deg=yaw,
                size=(w, h),
                radius=float(g.get("radius", 2.0)),
                corners=corners,
            )

    def __len__(self) -> int:
        return len(self.gates)

    def __getitem__(self, gate_id: int) -> GateInfo:
        return self.gates[gate_id]

    def __iter__(self):
        return iter(self.gates.values())

    def get_corners(self, gate_id: int) -> np.ndarray:
        """Return (4, 3) float32 array [TL, TR, BR, BL] in world frame."""
        return self.gates[gate_id].corners.as_array()

    def get_all_ids(self) -> list[int]:
        return sorted(self.gates)


# ---------------------------------------------------------------------------
# Verification against calibration reference data
# ---------------------------------------------------------------------------

def verify_against_calibration(
    gate_map: GateMap,
    calib_json_path: Path = DEFAULT_CALIB_JSON,
    tol_m: float = 1e-3,
) -> bool:
    """
    Compare gate_model computed corners against the reference corners
    stored in calibration_frames_card.json.

    Each annotation in the calibration JSON contains corners_world
    computed offline by the same formula.  This check confirms that the
    Python implementation is bit-for-bit consistent with those values.

    Returns True if all errors are within tol_m metres, False otherwise.
    """
    with open(calib_json_path, encoding="utf-8") as f:
        data = json.load(f)

    max_err = 0.0
    all_ok = True
    rows: list[tuple] = []

    for section in ["fov_frames", "multi_frames"]:
        for frame in data.get(section, []):
            for ann in frame["annotations"]:
                gid = ann["gate_world"]["gate_id"]
                ref_cw = ann["gate_world"]["corners_world"]

                if gid not in gate_map.gates:
                    rows.append((gid, frame["image_filename"], "NOT IN MAP", "", ""))
                    all_ok = False
                    continue

                computed = gate_map.get_corners(gid)  # (4, 3)

                for i, name in enumerate(CORNER_NAMES):
                    ref = np.array(ref_cw[name], dtype=np.float64)
                    err = float(np.linalg.norm(computed[i].astype(np.float64) - ref))
                    max_err = max(max_err, err)
                    if err > tol_m:
                        rows.append((gid, frame["image_filename"], name, f"{err:.6f} m", "FAIL"))
                        all_ok = False

    print(f"\nVerification vs calibration_frames_card.json  (tol = {tol_m*1000:.1f} mm)")
    if all_ok:
        print(f"  All corners match  (max_err = {max_err*1000:.3f} mm)  OK")
    else:
        print(f"  FAILURES detected  (max_err = {max_err*1000:.3f} mm)")
        for gid, fname, name, err, status in rows:
            print(f"  Gate {gid}  {fname:45s}  {name:22s}  {err}  {status}")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Show gate corners from track.json"
    )
    parser.add_argument(
        "--track",
        type=Path,
        default=DEFAULT_TRACK,
        help="Path to track.json",
    )
    parser.add_argument(
        "--gate-id",
        type=int,
        default=None,
        help="Print corners for a single gate",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify computed corners against calibration_frames_card.json",
    )
    parser.add_argument(
        "--calib-json",
        type=Path,
        default=DEFAULT_CALIB_JSON,
        help="Path to calibration_frames_card.json (used with --verify)",
    )
    args = parser.parse_args()

    gate_map = GateMap(args.track)
    print(f"Loaded {len(gate_map)} gates from {args.track}")

    if args.gate_id is not None:
        if args.gate_id not in gate_map.gates:
            print(f"Gate {args.gate_id} not found. Available: {gate_map.get_all_ids()}")
            return
        info = gate_map[args.gate_id]
        pos = [round(float(v), 4) for v in info.position]
        print(f"\nGate {info.gate_id}: {info.name}")
        print(f"  position (bottom-center): {pos}")
        print(f"  yaw = {info.yaw_deg} deg   size = {info.size[0]} x {info.size[1]} m")
        print(f"  Corners (world):")
        for name, corner in zip(CORNER_NAMES, info.corners):
            c = [round(float(v), 4) for v in corner]
            print(f"    {name:22s}  {c}")
    else:
        print(f"\n{'ID':>4}  {'Name':20s}  {'Bottom-center':30s}  {'Yaw':>8s}  {'Elevated'}")
        print("  " + "-" * 78)
        for info in sorted(gate_map, key=lambda g: g.gate_id):
            elev = f"y={info.position[1]:.3f} m" if info.position[1] > 0.0 else ""
            pos_str = f"[{info.position[0]:.2f}, {info.position[1]:.3f}, {info.position[2]:.2f}]"
            print(f"  {info.gate_id:3d}  {info.name:20s}  {pos_str:30s}  {info.yaw_deg:7.1f}  {elev}")

    if args.verify:
        verify_against_calibration(gate_map, args.calib_json)


if __name__ == "__main__":
    main()

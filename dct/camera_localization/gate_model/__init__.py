"""Gate geometry helpers for experimental camera localization."""

from .gate_model import CORNER_NAMES, GateCorners, GateInfo, GateMap, compute_corners, rotation_y

__all__ = [
    "CORNER_NAMES",
    "GateCorners",
    "GateInfo",
    "GateMap",
    "compute_corners",
    "rotation_y",
]

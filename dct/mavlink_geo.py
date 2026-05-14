"""Small local-track to geodetic transform used by Replay MAVLink output."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    alt: float


@dataclass(frozen=True)
class TrackBounds:
    origin_x: float
    origin_z: float
    size_x: float
    size_z: float

    @classmethod
    def from_track(cls, track: dict) -> "TrackBounds | None":
        bounds = track.get("bounds") if isinstance(track, dict) else None
        if not isinstance(bounds, dict):
            return None
        try:
            return cls(
                origin_x=float(bounds.get("origin_x", 0.0)),
                origin_z=float(bounds.get("origin_z", 0.0)),
                size_x=float(bounds["x"]),
                size_z=float(bounds.get("z", bounds["y"])),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _deg_to_enu(point: GeoPoint, origin: GeoPoint) -> np.ndarray:
    lat0 = math.radians(origin.lat)
    east = math.radians(point.lon - origin.lon) * EARTH_RADIUS_M * math.cos(lat0)
    north = math.radians(point.lat - origin.lat) * EARTH_RADIUS_M
    up = point.alt - origin.alt
    return np.array([east, north, up], dtype=float)


def _enu_to_deg(enu: np.ndarray, origin: GeoPoint) -> GeoPoint:
    lat0 = math.radians(origin.lat)
    lat = origin.lat + math.degrees(float(enu[1]) / EARTH_RADIUS_M)
    cos_lat0 = max(math.cos(lat0), 1e-9)
    lon = origin.lon + math.degrees(float(enu[0]) / (EARTH_RADIUS_M * cos_lat0))
    alt = origin.alt + float(enu[2])
    return GeoPoint(lat=lat, lon=lon, alt=alt)


class TrackGeoTransform:
    """Affine transform from DCT track xyz meters to geodetic coordinates."""

    def __init__(
        self,
        bounds: TrackBounds,
        origin_geo: GeoPoint,
        x_geo: GeoPoint,
        z_geo: GeoPoint,
    ) -> None:
        if bounds.size_x <= 0 or bounds.size_z <= 0:
            raise ValueError("Track bounds size_x and size_z must be positive")
        self.bounds = bounds
        self.origin_geo = origin_geo
        self._origin_enu = np.zeros(3, dtype=float)
        self._x_basis = _deg_to_enu(x_geo, origin_geo) / bounds.size_x
        self._z_basis = _deg_to_enu(z_geo, origin_geo) / bounds.size_z

    def to_geo(self, xyz: Sequence[float]) -> GeoPoint:
        pos = np.asarray(xyz, dtype=float).reshape(3)
        dx = float(pos[0] - self.bounds.origin_x)
        dz = float(pos[2] - self.bounds.origin_z)
        enu = self._origin_enu + dx * self._x_basis + dz * self._z_basis
        enu[2] += float(pos[1])
        return _enu_to_deg(enu, self.origin_geo)


def build_transform_from_settings(
    track: dict | None,
    settings: dict,
) -> TrackGeoTransform | None:
    bounds = TrackBounds.from_track(track or {})
    if bounds is None:
        return None
    anchors = settings.get("anchors") if isinstance(settings, dict) else None
    if not isinstance(anchors, dict):
        return None
    try:
        origin = GeoPoint(**anchors["origin"])
        x_point = GeoPoint(**anchors["x"])
        z_point = GeoPoint(**anchors["z"])
    except (KeyError, TypeError, ValueError):
        return None
    return TrackGeoTransform(bounds, origin, x_point, z_point)

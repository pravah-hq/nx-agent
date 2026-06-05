"""
Geographic helpers: haversine distance and bearings.

Used by graph edges, viewshed, map rendering, and targeting.
"""

from __future__ import annotations

import math

from agent.types import Pano


def normalize_deg(value: float) -> float:
    return value % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest angle between two compass directions (0–180)."""
    return abs(((a - b + 540.0) % 360.0) - 180.0)


def bearing_deg(from_pano: Pano, to_lat: float, to_lon: float) -> float:
    """Compass bearing from pano position toward (to_lat, to_lon)."""
    lon1 = math.radians(from_pano.lon)
    lat1 = math.radians(from_pano.lat)
    lon2 = math.radians(to_lon)
    lat2 = math.radians(to_lat)
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return normalize_deg(math.degrees(math.atan2(y, x)))


def distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Haversine distance in meters."""
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    d_lat = lat2 - lat1
    d_lon = math.radians(b_lon - a_lon)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def distance_between_panos(a: Pano, b: Pano) -> float:
    return distance_m(a.lat, a.lon, b.lat, b.lon)


def move_bearing_between_panos(from_pano: Pano, to_pano: Pano) -> float:
    """Compass bearing (degrees) of the step from from_pano to to_pano."""
    return bearing_deg(from_pano, to_pano.lat, to_pano.lon)


def destination_lat_lon(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """Point reached from (lat, lon) along bearing for distance_m (matches frontend)."""
    radius = 6_371_000
    delta = distance_m / radius
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat_deg)
    lambda1 = math.radians(lon_deg)
    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(
        theta
    )
    phi2 = math.asin(sin_phi2)
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lambda2 = lambda1 + math.atan2(y, x)
    return math.degrees(phi2), math.degrees(lambda2)

"""
Fetch and stitch map tiles for VLM screenshots (matches frontend "Streets" basemap).

Uses OpenStreetMap standard tiles — see https://operations.osm.org/policy/tiles/
"""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent.data_loader import repo_root

TILE_SIZE = 256
# Same URL as src/App.tsx MAP_TYPES id "streets"
STREETS_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_USER_AGENT = "nx-agent/1.0 (+https://github.com; local VLM navigation research)"
TILE_FETCH_RETRIES = 2
TILE_FETCH_TIMEOUT_S = 12


def _tile_cache_dir() -> Path:
    cache = repo_root() / ".cache" / "osm_tiles"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _fetch_tile(z: int, x: int, y: int) -> bytes:
    cache_path = _tile_cache_dir() / str(z) / str(x) / f"{y}.png"
    if cache_path.is_file():
        return cache_path.read_bytes()

    url = STREETS_TILE_URL.format(z=z, x=x, y=y)
    last_err: Exception | None = None
    for attempt in range(TILE_FETCH_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": TILE_USER_AGENT})
            with urllib.request.urlopen(req, timeout=TILE_FETCH_TIMEOUT_S) as resp:
                data = resp.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < TILE_FETCH_RETRIES:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch tile z={z} x={x} y={y}: {last_err}") from last_err


def _lon_to_tile_x(lon: float, zoom: int) -> float:
    return (lon + 180.0) / 360.0 * (2**zoom)


def _lat_to_tile_y(lat: float, zoom: int) -> float:
    lat_rad = math.radians(lat)
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2**zoom)


def _meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)


def pick_zoom_for_span_m(span_m: float, lat: float, image_size: int) -> int:
    """Zoom level so span_m fits the output image with street detail."""
    for z in range(21, 14, -1):
        mpp = _meters_per_pixel(lat, z)
        if span_m / mpp <= image_size * 0.92:
            return z
    return 15


def stitch_streets_basemap(
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    out_size: int,
) -> "Image.Image":
    """Return a PIL RGB image of OSM Streets tiles cropped to lat/lon bounds."""
    from io import BytesIO

    from PIL import Image

    lat_mid = (min_lat + max_lat) / 2.0
    lon_mid = (min_lon + max_lon) / 2.0
    from agent.geo import distance_m

    span_m = max(
        distance_m(min_lat, min_lon, min_lat, max_lon),
        distance_m(min_lat, min_lon, max_lat, min_lon),
        20.0,
    )
    zoom = pick_zoom_for_span_m(span_m, lat_mid, out_size)

    x0 = _lon_to_tile_x(min_lon, zoom)
    x1 = _lon_to_tile_x(max_lon, zoom)
    y0 = _lat_to_tile_y(max_lat, zoom)
    y1 = _lat_to_tile_y(min_lat, zoom)

    tile_x_min = int(math.floor(x0))
    tile_x_max = int(math.floor(x1))
    tile_y_min = int(math.floor(y0))
    tile_y_max = int(math.floor(y1))

    n_tiles_x = tile_x_max - tile_x_min + 1
    n_tiles_y = tile_y_max - tile_y_min + 1
    mosaic = Image.new("RGB", (n_tiles_x * TILE_SIZE, n_tiles_y * TILE_SIZE))

    for tx in range(tile_x_min, tile_x_max + 1):
        for ty in range(tile_y_min, tile_y_max + 1):
            data = _fetch_tile(zoom, tx, ty)
            tile = Image.open(BytesIO(data)).convert("RGB")
            paste_x = (tx - tile_x_min) * TILE_SIZE
            paste_y = (ty - tile_y_min) * TILE_SIZE
            mosaic.paste(tile, (paste_x, paste_y))

    world_w = TILE_SIZE * (2**zoom)
    px_left = x0 * TILE_SIZE - tile_x_min * TILE_SIZE
    px_right = x1 * TILE_SIZE - tile_x_min * TILE_SIZE
    px_top = y0 * TILE_SIZE - tile_y_min * TILE_SIZE
    px_bottom = y1 * TILE_SIZE - tile_y_min * TILE_SIZE

    crop = mosaic.crop(
        (
            int(math.floor(px_left)),
            int(math.floor(px_top)),
            int(math.ceil(px_right)),
            int(math.ceil(px_bottom)),
        )
    )
    return crop.resize((out_size, out_size), Image.Resampling.LANCZOS)

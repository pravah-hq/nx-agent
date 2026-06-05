"""
Carto dark basemap tiles — matches the frontend default map (App.tsx MAP_TYPES dark).
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

from agent.data_loader import repo_root
from PIL import Image

TILE_SIZE = 256
SUBDOMAINS = ("a", "b", "c", "d")
CARTO_DARK_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
FALLBACK_RGB = (15, 23, 42)


def _tile_cache_dir() -> Path:
    cache = repo_root() / ".cache" / "map_tiles"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _lat_lon_to_tile_xy(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    scale = TILE_SIZE * (2**zoom)
    x = (lon_deg + 180.0) / 360.0 * scale
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def _choose_zoom(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    *,
    target_px: int,
) -> int:
    for zoom in range(19, 11, -1):
        x0, y0 = _lat_lon_to_tile_xy(max_lat, min_lon, zoom)
        x1, y1 = _lat_lon_to_tile_xy(min_lat, max_lon, zoom)
        if max(x1 - x0, y1 - y0) <= target_px:
            return zoom
    return 12


def _fetch_tile(zoom: int, x: int, y: int) -> Image.Image:
    cache_path = _tile_cache_dir() / str(zoom) / str(x) / f"{y}.png"
    if cache_path.is_file():
        return Image.open(cache_path).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    subdomain = SUBDOMAINS[(x + y) % len(SUBDOMAINS)]
    url = CARTO_DARK_URL.format(s=subdomain, z=zoom, x=x, y=y)
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            data = response.read()
        cache_path.write_bytes(data)
        return Image.open(cache_path).convert("RGB")
    except OSError:
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), FALLBACK_RGB)


def render_carto_dark_basemap(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    *,
    size: int,
    padding_px: int,
) -> Image.Image:
    """Stitch Carto dark tiles for the lat/lon bounds into a size×size RGB image."""
    target = max(size - 2 * padding_px, 64)
    zoom = _choose_zoom(min_lat, min_lon, max_lat, max_lon, target_px=target)

    nw_x, nw_y = _lat_lon_to_tile_xy(max_lat, min_lon, zoom)
    se_x, se_y = _lat_lon_to_tile_xy(min_lat, max_lon, zoom)

    x0 = int(math.floor(nw_x / TILE_SIZE))
    y0 = int(math.floor(nw_y / TILE_SIZE))
    x1 = int(math.floor(se_x / TILE_SIZE))
    y1 = int(math.floor(se_y / TILE_SIZE))

    mosaic_w = (x1 - x0 + 1) * TILE_SIZE
    mosaic_h = (y1 - y0 + 1) * TILE_SIZE
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), FALLBACK_RGB)

    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            tile = _fetch_tile(zoom, tx, ty)
            mosaic.paste(tile, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))

    origin_x = nw_x - x0 * TILE_SIZE
    origin_y = nw_y - y0 * TILE_SIZE
    crop_w = max(int(se_x - nw_x), 1)
    crop_h = max(int(se_y - nw_y), 1)
    crop = mosaic.crop(
        (
            int(origin_x),
            int(origin_y),
            int(origin_x + crop_w),
            int(origin_y + crop_h),
        )
    )
    return crop.resize((size, size), Image.Resampling.BILINEAR)

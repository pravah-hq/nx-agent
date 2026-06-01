"""
Street-view crops from equirectangular panoramas for the VLM.

Cached under .cache/agent_views/ — delete cache after changing FOV or bin logic.
Env: VLM_ASSESS_CROP_FOV (degrees horizontal slice from full pano width).
"""

from __future__ import annotations

from pathlib import Path

from agent.data_loader import repo_root
from agent.types import DIRECTION_BIN_WIDTH_DEG, Pano

DEFAULT_CROP_FOV_DEG = 90
DEFAULT_VIEW_SIZE = (768, 768)


def panorama_path(pano: Pano, panoramas_root: Path | None = None) -> Path:
    root = panoramas_root or repo_root() / "data" / "panoramas"
    return root / Path(pano.image_path)


def render_direction_crop(
    pano: Pano,
    direction_bin: int,
    *,
    panoramas_root: Path | None = None,
    cache_dir: Path | None = None,
    crop_fov_deg: float = DEFAULT_CROP_FOV_DEG,
    out_size: tuple[int, int] = DEFAULT_VIEW_SIZE,
) -> Path:
    """
    Rectilinear crop centered on direction_bin (offset from pano forward, 30° per bin).

    Handles equirectangular wrap at the 0°/360° seam.
    """
    from PIL import Image

    source = panorama_path(pano, panoramas_root)
    if not source.is_file():
        raise FileNotFoundError(f"Panorama image not found: {source}")

    cache = cache_dir or repo_root() / ".cache" / "agent_views"
    cache.mkdir(parents=True, exist_ok=True)
    fov_key = int(round(crop_fov_deg))
    out_path = cache / f"{pano.id.replace('/', '_')}_bin{direction_bin}_fov{fov_key}.jpg"
    if out_path.is_file():
        return out_path

    image = Image.open(source).convert("RGB")
    width, height = image.size
    offset_deg = direction_bin * DIRECTION_BIN_WIDTH_DEG
    center_x = int(round((0.5 + offset_deg / 360.0) * width)) % width
    crop_w = max(32, int(round(width * (crop_fov_deg / 360.0))))
    half = crop_w // 2
    left = center_x - half
    right = center_x + half

    if left >= 0 and right <= width:
        crop = image.crop((left, 0, right, height))
    else:
        # Wrap around the equirectangular seam.
        if left < 0:
            left_part = image.crop((width + left, 0, width, height))
            right_part = image.crop((0, 0, right, height))
        else:
            left_part = image.crop((left, 0, width, height))
            right_part = image.crop((0, 0, right - width, height))
        crop = Image.new("RGB", (crop_w, height))
        left_part_width = left_part.size[0]
        crop.paste(left_part, (0, 0))
        crop.paste(right_part, (left_part_width, 0))

    crop = crop.resize(out_size, Image.Resampling.LANCZOS)
    crop.save(out_path, format="JPEG", quality=90)
    return out_path

from __future__ import annotations

from agent.types import DIRECTION_BIN_COUNT, DIRECTION_BIN_WIDTH_DEG, Pano
from agent.geo import normalize_deg


def clamp_bin(value: int) -> int:
    return value % DIRECTION_BIN_COUNT


def bin_center_world_yaw(pano: Pano, direction_bin: int) -> float:
    """View direction in world coordinates: pano forward + bin * 30°."""
    return normalize_deg(pano.heading_deg + direction_bin * DIRECTION_BIN_WIDTH_DEG)


def bin_label(direction_bin: int) -> str:
    return f"bin {direction_bin} ({direction_bin * DIRECTION_BIN_WIDTH_DEG} deg from pano forward)"

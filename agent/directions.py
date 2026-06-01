"""
Discrete viewing directions (12 bins × 30°).

View yaw = pano.heading_deg + direction_bin * 30°. Used for poles_in_view and crops.
"""

from __future__ import annotations

from agent.types import DIRECTION_BIN_COUNT, DIRECTION_BIN_WIDTH_DEG, Pano
from agent.geo import normalize_deg


def clamp_bin(value: int) -> int:
    return value % DIRECTION_BIN_COUNT


def bin_center_world_yaw(pano: Pano, direction_bin: int) -> float:
    """World compass yaw of the center of this direction bin."""
    return normalize_deg(pano.heading_deg + direction_bin * DIRECTION_BIN_WIDTH_DEG)


def bin_label(direction_bin: int) -> str:
    return f"bin {direction_bin} ({direction_bin * DIRECTION_BIN_WIDTH_DEG} deg from pano forward)"

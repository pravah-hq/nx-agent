"""
Normalize and compare pole id strings from VLM JSON vs metadata.

VLM may return "57" or "POLE_000057"; both should match metadata pole_id.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.environment import World


def normalize_pole_id(pole_id: str) -> str:
    text = pole_id.strip().upper()
    if text.startswith("POLE_"):
        return text
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"POLE_{digits.zfill(6)}" if len(digits) <= 6 else f"POLE_{digits}"
    return text


def pole_ids_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return normalize_pole_id(a) == normalize_pole_id(b)


def track_id_for_pole_id(world: "World", pole_id: str) -> str | None:  # noqa: F821
    """Map a VLM/map pole_id string to metadata track_id."""
    target = normalize_pole_id(pole_id)
    for pole in world.poles:
        if normalize_pole_id(pole.pole_id) == target:
            return pole.track_id
    return None

"""
Temporary fixed pano route for navigation debugging.

Set DEBUG_FIXED_ROUTE = False to restore pole-based goals and default start.
"""

from __future__ import annotations

from agent.environment import World
from agent.targeting import pano_compact_id

DEBUG_FIXED_ROUTE = True
DEBUG_START_PANO = "209"
DEBUG_GOAL_PANO = "238"


def _normalize_compact(token: str) -> str:
    needle = token.strip().removesuffix(".jpg")
    if needle.isdigit():
        return needle.zfill(10)
    return needle


def find_pano_by_compact(world: World, compact: str) -> str | None:
    """Resolve a map label like 209 to a full pano id."""
    needle = _normalize_compact(compact)
    for pano_id in world.panos_by_id:
        if pano_compact_id(pano_id) == needle:
            return pano_id
    return None


def debug_start_pano_id(world: World) -> str | None:
    if not DEBUG_FIXED_ROUTE:
        return None
    return find_pano_by_compact(world, DEBUG_START_PANO)


def debug_goal_pano_id(world: World) -> str | None:
    if not DEBUG_FIXED_ROUTE:
        return None
    return find_pano_by_compact(world, DEBUG_GOAL_PANO)

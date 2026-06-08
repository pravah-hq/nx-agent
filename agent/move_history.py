"""
Track and describe the agent's previous move for map overlays and VLM prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.directions import bin_center_world_yaw
from agent.geo import move_bearing_between_panos
from agent.types import AgentState

if TYPE_CHECKING:
    from agent.environment import World


def signed_bearing_delta(from_deg: float, to_deg: float) -> float:
    """Signed shortest angle from_deg -> to_deg, in [-180, 180]."""
    return ((to_deg - from_deg + 540.0) % 360.0) - 180.0


def last_move_relative_to_view_deg(world: "World", state: AgentState) -> float | None:
    """Signed angle from current view to the backtrack direction (where you came from)."""
    if state.last_move_bearing_deg is None:
        return None
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    came_from_bearing = (state.last_move_bearing_deg + 180.0) % 360.0
    return signed_bearing_delta(view_yaw, came_from_bearing)


def last_move_context(world: "World", state: AgentState) -> dict:
    """JSON-friendly summary for navigation prompts."""
    rel = last_move_relative_to_view_deg(world, state)
    blocked = state.last_move_from_pano_id
    return {
        "purpose": (
            "Avoid backtracking: the magenta arrow on the map points toward the pano "
            "you came from. Do not move there again unless you turned away first."
        ),
        "last_move_bearing_deg": state.last_move_bearing_deg,
        "came_from_bearing_deg": (
            None
            if state.last_move_bearing_deg is None
            else (state.last_move_bearing_deg + 180.0) % 360.0
        ),
        "last_move_relative_to_view_deg": rel,
        "last_move_from_pano_id": blocked,
        "do_not_backtrack_to_pano_id": blocked,
        "on_map": (
            "magenta arrow from YOU points toward the pano you came from (north-up); "
            "do not move to do_not_backtrack_to_pano_id"
            if state.last_move_bearing_deg is not None
            else None
        ),
    }

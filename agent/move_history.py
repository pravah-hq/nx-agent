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
    """How the last move bearing relates to current view (0 = straight ahead)."""
    if state.last_move_bearing_deg is None:
        return None
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    return signed_bearing_delta(view_yaw, state.last_move_bearing_deg)


def last_move_context(world: "World", state: AgentState) -> dict:
    """JSON-friendly summary for navigation prompts."""
    rel = last_move_relative_to_view_deg(world, state)
    return {
        "last_move_bearing_deg": state.last_move_bearing_deg,
        "last_move_relative_to_view_deg": rel,
        "last_move_from_pano_id": state.last_move_from_pano_id,
        "on_map": (
            "magenta arrow from YOU labeled last move (heading-up map); "
            "0 deg relative = straight behind you on the map"
            if state.last_move_bearing_deg is not None
            else None
        ),
    }

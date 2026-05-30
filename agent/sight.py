from __future__ import annotations

import os

from agent.environment import World
from agent.types import AgentState, PoleInView

CLEAR_VIEW_MAX_DISTANCE_M = float(os.environ.get("VLM_CLEAR_VIEW_MAX_M", "40"))
CLEAR_VIEW_MAX_ANGLE_DEG = float(os.environ.get("VLM_CLEAR_VIEW_MAX_ANGLE_DEG", "45"))


def geometric_sight_clear(sight: PoleInView | None) -> bool:
    if sight is None:
        return False
    return (
        sight.distance_m <= CLEAR_VIEW_MAX_DISTANCE_M
        and sight.angle_from_view_deg <= CLEAR_VIEW_MAX_ANGLE_DEG
    )


def target_pole_primary_in_viewshed(world: World, state: AgentState) -> bool:
    """Target is in viewshed and is the most centered visible pole."""
    track = state.pole_in_consideration
    if not track:
        return False
    visible = world.poles_in_view(state)
    if not visible:
        return False
    target = next((p for p in visible if p.track_id == track), None)
    if target is None:
        return False
    if not geometric_sight_clear(target):
        return False
    best_angle = min(p.angle_from_view_deg for p in visible)
    return target.angle_from_view_deg <= best_angle + 5

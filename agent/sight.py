"""
Geometric sight helpers (stub policy + optional corroboration).

Optional geometry hints for VLM prompts only (not used to set clear view).
Tune VLM_CLEAR_VIEW_MAX_M / VLM_CLEAR_VIEW_MAX_ANGLE_DEG for hint thresholds.
"""

from __future__ import annotations

import os

from agent.environment import World
from agent.geo import angle_diff_deg, bearing_deg, distance_m
from agent.types import (
    DIRECTION_BIN_WIDTH_DEG,
    VIEW_SHED_RADIUS_M,
    AgentState,
    PoleInView,
)

CLEAR_VIEW_MAX_DISTANCE_M = float(os.environ.get("VLM_CLEAR_VIEW_MAX_M", "55"))
CLEAR_VIEW_MAX_ANGLE_DEG = float(os.environ.get("VLM_CLEAR_VIEW_MAX_ANGLE_DEG", "70"))


def compute_target_pole_sight(world: World, state: AgentState) -> PoleInView | None:
    """Target pole geometry from current pano + facing (wider than poles_in_view list)."""
    track = state.pole_in_consideration
    if not track:
        return None
    pole = world.poles_by_track.get(track)
    if not pole:
        return None

    pano = world.panos_by_id[state.pano_id]
    view_yaw = world.view_yaw_deg(state)
    dist = distance_m(pano.lat, pano.lon, pole.lat, pole.lon)
    max_dist = max(VIEW_SHED_RADIUS_M, 55.0)
    if dist > max_dist:
        return None

    bearing = bearing_deg(pano, pole.lat, pole.lon)
    angle_from_view = angle_diff_deg(view_yaw, bearing)
    return PoleInView(
        track_id=pole.track_id,
        pole_id=pole.pole_id,
        bearing_deg=bearing,
        distance_m=dist,
        angle_from_view_deg=angle_from_view,
        in_center=angle_from_view <= DIRECTION_BIN_WIDTH_DEG / 2,
    )


def geometric_sight_clear(sight: PoleInView | None) -> bool:
    """True if target pole entry is within distance/angle thresholds."""
    if sight is None:
        return False
    return (
        sight.distance_m <= CLEAR_VIEW_MAX_DISTANCE_M
        and sight.angle_from_view_deg <= CLEAR_VIEW_MAX_ANGLE_DEG
    )


def target_pole_primary_in_viewshed(world: World, state: AgentState) -> bool:
    """True if target is visible and most centered among visible poles."""
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

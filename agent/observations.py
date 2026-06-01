"""
Human-readable and JSON observations printed each step (CLI verbose mode).

target_pole_in_clear_view in JSON mirrors pole_in_clear_view bool from the loop.
"""

from __future__ import annotations

import json
from typing import Any

from agent.directions import bin_center_world_yaw, bin_label
from agent.environment import World
from agent.graph import get_neighbors
from agent.sight import compute_target_pole_sight, geometric_sight_clear
from agent.types import AgentState


def format_state(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
) -> str:
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    target = state.pole_in_consideration
    target_pole_id = (
        world.poles_by_track[target].pole_id if target and target in world.poles_by_track else "none"
    )
    sight = compute_target_pole_sight(world, state)
    geo_line = "n/a"
    if sight:
        geo_line = (
            f"{sight.distance_m:.0f}m, {sight.angle_from_view_deg:.0f}° off-axis, "
            f"geometric_clear={geometric_sight_clear(sight)}"
        )
    lines = [
        f"pano: {pano.id}",
        f"direction: {bin_label(state.direction_bin)} (world yaw {view_yaw:.1f} deg)",
        f"pole_in_consideration: {target or 'none'} ({target_pole_id})",
        f"target_pole_in_clear_view: {pole_in_clear_view} (VLM)",
        f"target_geometry_hint: {geo_line}",
        f"pole_guess: {_format_guess(state)}",
        f"classified: {len(state.classified)}/{len(world.poles)}",
        f"neighbors ({len(neighbors)}): {', '.join(_compact_id(n) for n in neighbors[:6])}"
        + (" ..." if len(neighbors) > 6 else ""),
        f"remaining targets: {', '.join(world.remaining_pole_ids(state)) or 'none'}",
    ]
    return "\n".join(lines)


def state_to_json(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
) -> dict[str, Any]:
    pano = world.panos_by_id[state.pano_id]
    sight = compute_target_pole_sight(world, state)
    target_geometry = None
    if sight:
        target_geometry = {
            "distance_m": round(sight.distance_m, 1),
            "angle_from_view_deg": round(sight.angle_from_view_deg, 1),
            "in_viewshed_cone": geometric_sight_clear(sight),
        }
    return {
        "pano_id": state.pano_id,
        "direction_bin": state.direction_bin,
        "view_yaw_deg": bin_center_world_yaw(pano, state.direction_bin),
        "pole_in_consideration": state.pole_in_consideration,
        "target_pole_in_clear_view": pole_in_clear_view,
        "target_geometry": target_geometry,
        "pole_guess": None
        if state.pole_guess is None
        else {
            "track_id": state.pole_guess.track_id,
            "pole_id": state.pole_guess.pole_id,
            "pole_type": state.pole_guess.pole_type,
            "confidence": state.pole_guess.confidence,
            "note": state.pole_guess.note,
        },
        "classified": state.classified,
        "neighbor_ids": get_neighbors(world.neighbor_map, state.pano_id),
        "remaining_pole_ids": world.remaining_pole_ids(state),
        "task_complete": world.is_task_complete(state),
    }


def print_vlm_step_responses(policy) -> None:
    """Print all VLM raw responses recorded this agent step (VlmPolicy only)."""
    from agent.vlm_policy import VlmPolicy

    if not isinstance(policy, VlmPolicy) or not policy.vlm_step_calls:
        return
    print("vlm responses this step:")
    for entry in policy.vlm_step_calls:
        phase = entry.get("phase", "?")
        attempt = entry.get("attempt", 1)
        response = entry.get("response", "")
        label = phase if attempt == 1 else f"{phase} (attempt {attempt})"
        print(f"  [{label}]")
        print(response)
        print()


def print_observation(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
    as_json: bool = False,
) -> None:
    if as_json:
        print(
            json.dumps(
                state_to_json(world, state, pole_in_clear_view=pole_in_clear_view),
                indent=2,
            )
        )
    else:
        print(format_state(world, state, pole_in_clear_view=pole_in_clear_view))


def _format_guess(state: AgentState) -> str:
    if state.pole_guess is None:
        return "none"
    g = state.pole_guess
    parts = [g.pole_id]
    if g.pole_type:
        parts.append(str(g.pole_type))
    if g.note:
        parts.append(f"({g.note})")
    return " ".join(parts)


def _compact_id(pano_id: str) -> str:
    return pano_id.split("/")[-1]

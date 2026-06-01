"""
Human-readable and JSON observations printed each step (CLI verbose mode).
"""

from __future__ import annotations

import json
from typing import Any

from agent.directions import bin_center_world_yaw, bin_label
from agent.environment import World
from agent.graph import get_neighbors
from agent.types import AgentState


def format_state(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
    visible_pole_id: str | None = None,
) -> str:
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    nav_track = state.pole_in_consideration
    nav_pole_id = (
        world.poles_by_track[nav_track].pole_id
        if nav_track and nav_track in world.poles_by_track
        else "none"
    )
    lines = [
        f"pano: {pano.id}",
        f"direction: {bin_label(state.direction_bin)} (world yaw {view_yaw:.1f} deg)",
        f"pole_in_clear_view: {pole_in_clear_view} (VLM)",
        f"visible_pole_for_classify: {visible_pole_id or 'none'}",
        f"nav_pole_in_consideration: {nav_track or 'none'} ({nav_pole_id})",
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
    visible_pole_id: str | None = None,
) -> dict[str, Any]:
    pano = world.panos_by_id[state.pano_id]
    return {
        "pano_id": state.pano_id,
        "direction_bin": state.direction_bin,
        "view_yaw_deg": bin_center_world_yaw(pano, state.direction_bin),
        "pole_in_clear_view": pole_in_clear_view,
        "visible_pole_id": visible_pole_id,
        "pole_in_consideration": state.pole_in_consideration,
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
        "already_classified_pole_ids": [
            world.poles_by_track[t].pole_id
            for t in state.classified
            if t in world.poles_by_track
        ],
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
    policy=None,
) -> None:
    visible_pole_id: str | None = None
    from agent.vlm_policy import VlmPolicy

    if isinstance(policy, VlmPolicy) and policy.last_visible_track_id:
        pole = world.poles_by_track.get(policy.last_visible_track_id)
        if pole:
            visible_pole_id = pole.pole_id

    if as_json:
        print(
            json.dumps(
                state_to_json(
                    world,
                    state,
                    pole_in_clear_view=pole_in_clear_view,
                    visible_pole_id=visible_pole_id,
                ),
                indent=2,
            )
        )
    else:
        print(
            format_state(
                world,
                state,
                pole_in_clear_view=pole_in_clear_view,
                visible_pole_id=visible_pole_id,
            )
        )


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

from __future__ import annotations

import json
from typing import Any

from agent.directions import bin_center_world_yaw, bin_label
from agent.environment import World
from agent.graph import get_neighbors
from agent.types import AgentState


def format_state(world: World, state: AgentState, poles_in_view) -> str:
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    lines = [
        f"pano: {pano.id}",
        f"direction: {bin_label(state.direction_bin)} (world yaw {view_yaw:.1f} deg)",
        f"pole_in_consideration: {state.pole_in_consideration or 'none'}",
        f"pole_guess: {_format_guess(state)}",
        f"classified: {len(state.classified)}/{len(world.poles)}",
        f"neighbors ({len(neighbors)}): {', '.join(_compact_id(n) for n in neighbors[:6])}"
        + (" ..." if len(neighbors) > 6 else ""),
        f"poles_in_view ({len(poles_in_view)}):",
    ]
    if not poles_in_view:
        lines.append("  (none)")
    else:
        for pole in poles_in_view:
            lines.append(
                f"  {pole.pole_id} @ {pole.bearing_deg:.0f} deg "
                f"{pole.distance_m:.1f} m (dview {pole.angle_from_view_deg:.0f} deg)"
            )
    lines.append(f"remaining targets: {', '.join(world.remaining_pole_ids(state)) or 'none'}")
    return "\n".join(lines)


def state_to_json(world: World, state: AgentState, poles_in_view) -> dict[str, Any]:
    pano = world.panos_by_id[state.pano_id]
    return {
        "pano_id": state.pano_id,
        "direction_bin": state.direction_bin,
        "view_yaw_deg": bin_center_world_yaw(pano, state.direction_bin),
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
        "poles_in_view": [
            {
                "track_id": p.track_id,
                "pole_id": p.pole_id,
                "bearing_deg": p.bearing_deg,
                "distance_m": p.distance_m,
                "angle_from_view_deg": p.angle_from_view_deg,
                "in_center": p.in_center,
            }
            for p in poles_in_view
        ],
        "neighbor_ids": get_neighbors(world.neighbor_map, state.pano_id),
        "remaining_pole_ids": world.remaining_pole_ids(state),
        "task_complete": world.is_task_complete(state),
    }


def print_observation(world: World, state: AgentState, poles_in_view, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(state_to_json(world, state, poles_in_view), indent=2))
    else:
        print(format_state(world, state, poles_in_view))


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

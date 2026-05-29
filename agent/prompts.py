from __future__ import annotations

import json

from agent.environment import World
from agent.graph import get_neighbors
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleInView


def build_vlm_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
    *,
    allowed_actions: list[str],
) -> str:
    payload = state_to_json(world, state, poles_in_view)
    payload["allowed_actions"] = allowed_actions
    payload["pole_types"] = list(POLE_TYPES)
    payload["rules"] = [
        "You are at one street panorama. You may only move to neighbor panorama ids listed.",
        "Neighbors are within 20 m. Do not invent pano ids.",
        "classify_or_stop requires pole_type when classifying an visible pole.",
        "Use stop_after true only when all poles are classified or you must end.",
    ]
    return (
        "You are a pole-hunting street-view agent in Bhelupur. "
        "Choose exactly one next action as JSON.\n\n"
        "Schema:\n"
        '{"action":"turn_left|turn_right|move|classify_or_stop",'
        '"pole_type":"distribution_transformer|lamp_post|billboard_pole|low_tension_pole|null",'
        '"stop_after":false,"reason":"short"}\n\n'
        f"State:\n{json.dumps(payload, indent=2)}"
    )


def allowed_actions(world: World, state: AgentState, poles_in_view: list[PoleInView]) -> list[str]:
    actions = ["turn_left", "turn_right"]
    if world.neighbor_panos_for_move(state):
        actions.append("move")
    if state.pole_in_consideration and any(
        p.track_id == state.pole_in_consideration for p in poles_in_view
    ):
        actions.append("classify_or_stop")
    elif poles_in_view:
        actions.append("classify_or_stop")
    return actions

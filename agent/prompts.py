from __future__ import annotations

import json

from agent.environment import World
from agent.graph import get_neighbors
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleInView


def build_map_navigation_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
    *,
    allowed_actions: list[str],
) -> str:
    payload = state_to_json(world, state, poles_in_view)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    payload["neighbor_pano_ids"] = neighbors
    payload["allowed_actions"] = allowed_actions
    payload["pole_types"] = list(POLE_TYPES)
    payload["map_legend"] = {
        "blue_dot": "current panorama",
        "light_dots": "reachable neighbors (within 20 m)",
        "orange": "target pole in consideration",
        "green": "other poles",
        "gray": "already classified",
        "wedge": "current viewing direction",
    }
    payload["rules"] = [
        "Use the MAP IMAGE to decide where to go and which way to face.",
        "Do not use geographic shortest-path on coordinates alone; use the map layout.",
        "move requires target_pano_id from neighbor_pano_ids only.",
        "Do NOT classify from this step. Use assess_classify when you want to check the street view.",
        "classify_or_stop is NOT allowed in this step.",
    ]
    return (
        "You navigate a street panorama agent using the MAP screenshot.\n"
        "Choose exactly one action as JSON.\n\n"
        "Schema:\n"
        '{"action":"turn_left|turn_right|move|assess_classify",'
        '"target_pano_id":"neighbor id or null",'
        '"reason":"short"}\n\n'
        f"State:\n{json.dumps(payload, indent=2)}"
    )


def build_classify_assessment_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
) -> str:
    pole = (
        world.poles_by_track[state.pole_in_consideration]
        if state.pole_in_consideration
        else None
    )
    payload = state_to_json(world, state, poles_in_view)
    payload["pole_types"] = list(POLE_TYPES)
    payload["assessment_rules"] = [
        "You see the STREET VIEW (panorama crop), not the map.",
        f"Target pole: {pole.pole_id if pole else 'none'}.",
        "Classify ONLY if the target pole is clearly visible, unobstructed, and complete enough to identify type.",
        "Partial poles, heavy occlusion, or extreme distance = view_clear false.",
        "If view_clear is true, you must provide pole_type from the allowed list.",
    ]
    return (
        "Assess whether the target pole can be classified from this street view.\n"
        "Reply with JSON only:\n"
        '{"view_clear":true|false,'
        '"pole_type":"distribution_transformer|lamp_post|billboard_pole|low_tension_pole|null",'
        '"stop_after":false,'
        '"reason":"short"}\n\n'
        f"State:\n{json.dumps(payload, indent=2)}"
    )


def navigation_allowed_actions(world: World, state: AgentState) -> list[str]:
    actions = ["turn_left", "turn_right", "assess_classify"]
    if get_neighbors(world.neighbor_map, state.pano_id):
        actions.insert(2, "move")
    return actions


def parse_assessment_response(text: str) -> dict | None:
    from agent.action_parse import extract_json_object

    payload = extract_json_object(text)
    if not payload:
        return None
    return payload

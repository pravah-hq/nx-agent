from __future__ import annotations

import json

from agent.environment import World
from agent.graph import get_neighbors
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleInView, PoleType

POLE_TYPE_GUIDE: dict[PoleType, str] = {
    "distribution_transformer": (
        "TWO poles with a large distribution transformer mounted BETWEEN them "
        "(box/cylinder on cross-arm or platform). Not a single pole."
    ),
    "lamp_post": (
        "ONE slender pole with a street LIGHT / luminaire at the top "
        "(fixture aimed toward street). No large transformer between two poles."
    ),
    "billboard_pole": (
        "ONE pole supporting an advertising BILLBOARD / large flat sign board "
        "(rectangular panel), not just a light fixture."
    ),
    "low_tension_pole": (
        "ONE ordinary utility pole with low-tension distribution wires "
        "(multiple small lines along the pole / cross-arm). No billboard, "
        "no street lamp as primary feature, not a two-pole transformer setup."
    ),
}


def build_map_navigation_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
    *,
    allowed_actions: list[str],
    blocked_move_targets: list[str] | None = None,
    neighbor_moves: list[dict] | None = None,
    goal_pano_id: str | None = None,
    planned_next_hop: str | None = None,
) -> str:
    payload = state_to_json(world, state, poles_in_view)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    payload["neighbor_pano_ids"] = neighbors
    if blocked_move_targets:
        payload["blocked_move_targets"] = blocked_move_targets
        payload["allowed_move_targets"] = [
            n for n in neighbors if n not in blocked_move_targets
        ]
    payload["allowed_actions"] = allowed_actions
    if neighbor_moves is not None:
        payload["neighbor_moves"] = neighbor_moves
    if goal_pano_id:
        from agent.targeting import pano_compact_id

        payload["goal_view_pano_id"] = goal_pano_id
        payload["goal_view_pano_label"] = pano_compact_id(goal_pano_id)
    if planned_next_hop:
        from agent.targeting import pano_compact_id

        payload["planned_next_hop"] = planned_next_hop
        payload["planned_next_hop_label"] = pano_compact_id(planned_next_hop)
    payload["pole_types"] = list(POLE_TYPES)
    payload["map_legend"] = {
        "blue_dot": "you (current pano)",
        "yellow_ring": "planned next pano hop",
        "light_dots": "neighbors reachable by move (20 m edges)",
        "gray_lines": "pano graph edges (move only along edges to light dots)",
        "orange": "target pole to find",
        "green": "other unclassified poles",
        "gray": "classified poles",
        "wedge": "viewing direction",
    }
    payload["rules"] = [
        "The map is a LOCAL zoom around you; gray lines are the only valid move links.",
        "For move, copy target_pano_id EXACTLY from neighbor_moves[].target_pano_id (not the label).",
        "Prefer neighbor_moves where recommended_next_hop is true, or lower distance_to_target_pole_m.",
        "Do not move to blocked_move_targets (immediate backtrack).",
        "Navigate toward goal_view_pano_id along the graph, not across empty map space.",
        "Do NOT classify from this step; assess_classify is automatic when the target pole is in poles_in_view.",
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


def _target_pole(world: World, state: AgentState):
    if not state.pole_in_consideration:
        return None
    return world.poles_by_track.get(state.pole_in_consideration)


def _assessment_context(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
) -> dict:
    payload = state_to_json(world, state, poles_in_view)
    payload.pop("pole_guess", None)
    pole = _target_pole(world, state)
    if pole:
        payload["target_pole_id"] = pole.pole_id
    return payload


def build_visibility_assessment_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
) -> str:
    pole = _target_pole(world, state)
    payload = _assessment_context(world, state, poles_in_view)
    return (
        "You see a STREET VIEW crop (not the map).\n"
        f"Target pole to classify later: {pole.pole_id if pole else 'unknown'}.\n"
        "Decide ONLY if that pole is visible enough to identify its type.\n"
        "Reply JSON only:\n"
        '{"view_clear":true|false,"reason":"short"}\n\n'
        "view_clear=true requires: pole structure readable, not mostly hidden, "
        "not extremely tiny in the image.\n\n"
        f"Context:\n{json.dumps(payload, indent=2)}"
    )


def build_pole_type_classification_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
) -> str:
    pole = _target_pole(world, state)
    payload = _assessment_context(world, state, poles_in_view)
    payload["pole_type_definitions"] = POLE_TYPE_GUIDE
    payload["allowed_pole_types"] = list(POLE_TYPES)
    return (
        "You see a STREET VIEW crop. Classify the TARGET pole's type.\n"
        f"Target pole: {pole.pole_id if pole else 'unknown'} "
        "(match the structure at that location; ignore other poles if possible).\n\n"
        "Definitions — pick the ONE best match:\n"
        + "\n".join(f"- {key}: {desc}" for key, desc in POLE_TYPE_GUIDE.items())
        + "\n\n"
        "Rules:\n"
        "- Do NOT default to lamp_post. Use lamp_post ONLY if a street light is "
        "clearly on top of a single pole.\n"
        "- distribution_transformer needs TWO poles + transformer between.\n"
        "- billboard_pole needs a sign/board, not just a lamp.\n"
        "- low_tension_pole is a plain utility pole with small LT wires, none of the above.\n"
        "Reply JSON only:\n"
        '{"pole_type":"distribution_transformer|lamp_post|billboard_pole|low_tension_pole",'
        '"confidence":"low|medium|high","reason":"what visual features you used"}\n\n'
        f"Context:\n{json.dumps(payload, indent=2)}"
    )


def build_classify_assessment_prompt(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
) -> str:
    """Legacy combined prompt; prefer two-step visibility + type prompts."""
    return build_pole_type_classification_prompt(world, state, poles_in_view)


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

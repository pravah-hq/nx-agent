"""
All VLM prompt text — primary place to tune behavior.

POLE_TYPE_GUIDE: edit definitions for classification / clear-view.
build_map_navigation_prompt / build_pole_in_clear_view_prompt: JSON schemas the model must follow.
"""

from __future__ import annotations

import json

from agent.environment import World
from agent.graph import get_neighbors
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleInView, PoleType

DUAL_IMAGE_GUIDE = {
    "image_1": "LOCAL MAP — pano graph (YOU, neighbors, edges, poles, goal hop)",
    "image_2": "STREET VIEW — panorama crop from your current position and facing",
    "use_both": (
        "Use the MAP for where to move along gray edges; use STREET VIEW for "
        "what is ahead and whether to turn before moving or assessing."
    ),
}

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
        "ONE ordinary utility pole with clearly visible low-tension distribution wires "
        "(multiple small lines along the pole / cross-arm). Use ONLY when lamp, "
        "billboard, and two-pole transformer setups are ruled out — never as a default."
    ),
}


def build_map_navigation_prompt(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
    allowed_actions: list[str],
    blocked_move_targets: list[str] | None = None,
    neighbor_moves: list[dict] | None = None,
    goal_pano_id: str | None = None,
    planned_next_hop: str | None = None,
) -> str:
    payload = state_to_json(world, state, pole_in_clear_view=pole_in_clear_view)
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
    payload["images"] = DUAL_IMAGE_GUIDE
    payload["map_legend"] = {
        "blue_dot": "you (current pano)",
        "yellow_ring": "planned next pano hop",
        "light_dots": "neighbors reachable by move (20 m edges)",
        "gray_lines": "pano graph edges (move only along edges to light dots)",
        "orange": "target pole to find",
        "green": "other unclassified poles",
        "gray": "classified poles",
        "wedge": "viewing direction (should match street view)",
    }
    payload["rules"] = [
        DUAL_IMAGE_GUIDE["use_both"],
        "target_pole_in_clear_view means VLM judged the target unambiguously identifiable (not your action).",
        "If true, the agent classifies using that type; you only navigate when false.",
        "MAP (image 1): choose move along gray lines to light neighbor dots only.",
        "STREET VIEW (image 2): decide if you should turn to find the orange target pole.",
        "For move, copy target_pano_id EXACTLY from neighbor_moves[].target_pano_id (not the label).",
        "Prefer neighbor_moves where recommended_next_hop is true, or lower distance_to_target_pole_m.",
        "Do not move to blocked_move_targets (immediate backtrack).",
        "Navigate toward goal_view_pano_id along the graph, not across empty map space.",
        "classify_or_stop is NOT allowed in this step.",
    ]
    return (
        "You control a street panorama agent. You receive TWO images every step.\n"
        "Image 1 = local MAP. Image 2 = STREET VIEW from current pano and facing.\n"
        "Use BOTH to decide where to go next (only when pole_in_clear_view is false).\n"
        "Choose exactly one action as JSON.\n\n"
        "Schema:\n"
        '{"action":"turn_left|turn_right|move",'
        '"target_pano_id":"neighbor id or null",'
        '"reason":"short"}\n\n'
        f"State:\n{json.dumps(payload, indent=2)}"
    )


def _target_pole(world: World, state: AgentState):
    if not state.pole_in_consideration:
        return None
    return world.poles_by_track.get(state.pole_in_consideration)


def _classification_context(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool,
) -> dict:
    payload = state_to_json(world, state, pole_in_clear_view=pole_in_clear_view)
    payload.pop("pole_guess", None)
    pole = _target_pole(world, state)
    if pole:
        payload["target_pole_id"] = pole.pole_id
    return payload


def build_pole_in_clear_view_prompt(
    world: World,
    state: AgentState,
    *,
    target_sight: PoleInView | None = None,
) -> str:
    pole = _target_pole(world, state)
    payload = _classification_context(world, state, pole_in_clear_view=False)
    payload["images"] = DUAL_IMAGE_GUIDE
    if pole:
        payload["target_pole"] = {
            "track_id": pole.track_id,
            "pole_id": pole.pole_id,
            "note": "ONLY this pole may set pole_in_clear_view true",
        }
        if target_sight:
            payload["target_pole"]["bearing_deg"] = round(target_sight.bearing_deg, 1)
            payload["target_pole"]["distance_m"] = round(target_sight.distance_m, 1)
            payload["target_pole"]["angle_from_view_deg"] = round(
                target_sight.angle_from_view_deg, 1
            )
    if target_sight:
        payload["geometric_target_in_viewshed"] = True
        payload["hint"] = (
            "Geometry says the TARGET pole is in the current street-view cone. "
            "If you see a pole at that bearing with a clear best-matching type, "
            "set pole_in_clear_view true and fill identifiable_pole_type."
        )
    else:
        payload["geometric_target_in_viewshed"] = False
        payload["hint"] = (
            "Target may be off-screen or far — use pole_in_clear_view false unless "
            "you still clearly see that specific pole in street view."
        )
    payload["pole_type_definitions"] = POLE_TYPE_GUIDE
    payload["allowed_pole_types"] = list(POLE_TYPES)
    return (
        "You receive TWO images: (1) MAP — orange dot = TARGET pole "
        "(2) STREET VIEW — current facing.\n\n"
        f"TARGET (pole_in_consideration): {pole.pole_id if pole else 'unknown'}.\n\n"
        "Set pole_in_clear_view=true when BOTH hold:\n"
        "1) The TARGET pole (orange on map) is visible in street view — not a different pole.\n"
        "2) You can assign its best-matching type from the four definitions below "
        "(single best guess is OK; you do not need 100% certainty).\n\n"
        "Set pole_in_clear_view=false when:\n"
        "- Target not visible, too small, heavily occluded, or wrong pole dominates view\n"
        "- You cannot pick any of the four types\n\n"
        "Optional: unambiguous_identifiable true if you are very confident; "
        "confirmed_target_pole_id if you can read the pole id.\n\n"
        "Definitions:\n"
        + "\n".join(f"- {key}: {desc}" for key, desc in POLE_TYPE_GUIDE.items())
        + "\n\n"
        "Reply JSON only:\n"
        '{"pole_in_clear_view":true|false,'
        '"unambiguous_identifiable":true|false,'
        f'"identifiable_pole_type":"one of {list(POLE_TYPES)} or null",'
        f'"confirmed_target_pole_id":"{pole.pole_id if pole else "null"}" or null,'
        '"reason":"what you see at the target bearing"}\n\n'
        f"Context:\n{json.dumps(payload, indent=2)}"
    )


def build_pole_type_classification_prompt(
    world: World,
    state: AgentState,
    *,
    pole_in_clear_view: bool = True,
) -> str:
    pole = _target_pole(world, state)
    payload = _classification_context(world, state, pole_in_clear_view=pole_in_clear_view)
    payload["pole_type_definitions"] = POLE_TYPE_GUIDE
    payload["allowed_pole_types"] = list(POLE_TYPES)
    payload["images"] = DUAL_IMAGE_GUIDE
    return (
        "You receive TWO images: (1) MAP with orange target (2) STREET VIEW crop.\n"
        "Classify the TARGET pole's type using STREET VIEW; MAP confirms which pole is target.\n"
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
        "- low_tension_pole: ONLY if clearly a plain utility pole with LT wires and "
        "NOT a lamp, billboard, or transformer setup.\n"
        "- If ambiguous, use confidence low and do NOT use low_tension_pole as a guess.\n"
        "Reply JSON only:\n"
        '{"classified_pole_id":"must equal target pole_id",'
        '"pole_type":"distribution_transformer|lamp_post|billboard_pole|low_tension_pole",'
        '"confidence":"low|medium|high","reason":"features of the TARGET pole only"}\n\n'
        f"Context:\n{json.dumps(payload, indent=2)}"
    )


def navigation_allowed_actions(world: World, state: AgentState) -> list[str]:
    actions = ["turn_left", "turn_right"]
    if get_neighbors(world.neighbor_map, state.pano_id):
        actions.insert(2, "move")
    return actions

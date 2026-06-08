"""
All VLM prompt text — primary place to tune behavior.

POLE_TYPE_GUIDE: edit definitions for classification / clear-view.
build_map_navigation_prompt / build_pole_in_clear_view_prompt: JSON schemas the model must follow.
"""

from __future__ import annotations

import json

from agent.environment import World
from agent.graph import get_neighbors
from agent.move_history import last_move_context
from agent.navigation import backtrack_blocked_ids, build_neighbor_move_options
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleType

MAP_IMAGE_GUIDE = {
    "image_1": "MAP OVERVIEW — entire map area (all panos/poles), same as web UI fitBounds",
    "image_2": "MAP ZOOM — zoomed crop around you and neighbors (north-up)",
    "image_3": "STREET VIEW — panorama crop from your current position and facing",
    "use_all": (
        "Maps match the frontend: dark basemap, cyan view wedge, magenta backtrack arrow. "
        "The magenta arrow points toward the pano you came from — do not move back along it. "
        "Use MAP OVERVIEW for global direction; MAP ZOOM for neighbor pano labels. "
        "Use STREET VIEW for turns and what is ahead."
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
    neighbor_moves: list[dict] | None = None,
    goal_pano_id: str | None = None,
) -> str:
    payload = state_to_json(world, state, pole_in_clear_view=pole_in_clear_view)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    payload["neighbor_pano_ids"] = neighbors
    payload["allowed_actions"] = allowed_actions
    blocked = backtrack_blocked_ids(state.last_move_from_pano_id)
    payload["backtrack_blocked_pano_ids"] = sorted(blocked)
    payload["last_move"] = last_move_context(world, state)
    if neighbor_moves is None:
        neighbor_moves = build_neighbor_move_options(neighbors)
    payload["neighbor_moves"] = [
        {
            **move,
            "backtrack_blocked": move["target_pano_id"] in blocked,
        }
        for move in neighbor_moves
    ]
    if goal_pano_id:
        from agent.targeting import pano_compact_id

        payload["goal_view_pano_id"] = goal_pano_id
        payload["goal_view_pano_label"] = pano_compact_id(goal_pano_id)
    payload["pole_types"] = list(POLE_TYPES)

    image_guide = MAP_IMAGE_GUIDE
    payload["images"] = image_guide
    payload["map_legend"] = {
        "cyan_wedge": "your view cone (same as frontend map)",
        "blue_pano": "your current pano",
        "white_panos": "unvisited neighbors",
        "purple_panos": "visited pano nodes",
        "gray_edges": "20 m pano graph edges",
        "green_pole": "target pole in consideration",
        "gray_poles": "classified poles",
        "magenta_arrow": "where you came from — do NOT backtrack to that neighbor",
        "north_up": "map is north-up like the web UI",
        "zoom_labels": "compact neighbor pano ids on MAP ZOOM only",
        "merged_dots": "on MAP OVERVIEW, nearby panos are merged into one dot",
    }
    rules = [
        image_guide["use_all"],
        "pole_in_clear_view is set by a prior VLM check (not your action).",
        "If true in state JSON, the agent classifies the visible pole; you only navigate when false.",
        "Move only along gray edges to neighbor panos listed in neighbor_moves.",
        "For move, set target_pano_id from neighbor_moves where backtrack_blocked is false.",
        "Never move to do_not_backtrack_to_pano_id / last_move_from_pano_id (same pano you just left).",
        "The magenta arrow on the map points toward that blocked pano — pick a different neighbor or turn first.",
        "Use last_move_relative_to_view_deg: ~0° means the blocked direction is straight ahead in your view.",
        "STREET VIEW: turn_left/turn_right before moving or when street context matters.",
        "Navigate toward goal_view_pano_id along the graph.",
        "classify_or_stop is NOT allowed in this step.",
    ]
    payload["rules"] = rules
    return (
        "You control a street panorama agent.\n"
        "You receive THREE images: (1) MAP OVERVIEW (2) MAP ZOOM (3) STREET VIEW.\n"
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
    classified_pole_ids: frozenset[str],
) -> str:
    payload = _classification_context(world, state, pole_in_clear_view=False)
    payload["images"] = MAP_IMAGE_GUIDE
    payload["map_legend"] = {
        "green": "unclassified poles (candidates you may identify)",
        "gray": "already classified — do NOT set pole_in_clear_view for these",
        "orange": "navigation hint only (where the agent is heading)",
        "cyan_wedge": "your view cone on the map",
        "north_up": "map is north-up like the web UI",
    }
    payload["already_classified_pole_ids"] = sorted(classified_pole_ids)
    payload["unclassified_poles"] = [
        {
            "pole_id": p.pole_id,
            "map_label": p.pole_id.replace("POLE_", ""),
            "on_map_color": "green",
        }
        for p in world.poles
        if p.track_id not in state.classified
    ]
    payload["pole_type_definitions"] = POLE_TYPE_GUIDE
    payload["allowed_pole_types"] = list(POLE_TYPES)
    return (
        "You receive THREE images: (1) MAP OVERVIEW (2) MAP ZOOM (3) STREET VIEW ahead.\n\n"
        "You do NOT know any pole types yet — only whether a pole is visible enough to classify.\n\n"
        "Set pole_in_clear_view=true ONLY when ALL are true:\n"
        "1) STREET VIEW shows a utility pole clearly (unobstructed, large enough to judge).\n"
        "2) That pole matches one GREEN labeled pole on the map (visible_pole_id).\n"
        "3) visible_pole_id is NOT listed in already_classified_pole_ids (not gray / done).\n\n"
        "Set pole_in_clear_view=false when: no pole visible, pole too far/occluded, only "
        "classified (gray) poles visible, or you cannot match the street pole to a green map label.\n\n"
        "If true, set visible_pole_id to the green pole label (e.g. POLE_000057) and "
        "identifiable_pole_type to your best type guess (or null if visible but type unclear).\n\n"
        "Definitions (for type guess only):\n"
        + "\n".join(f"- {key}: {desc}" for key, desc in POLE_TYPE_GUIDE.items())
        + "\n\n"
        "Reply JSON only:\n"
        '{"pole_in_clear_view":true|false,'
        '"visible_pole_id":"POLE_... or null",'
        f'"identifiable_pole_type":"one of {list(POLE_TYPES)} or null",'
        '"reason":"what pole you see and why it is or is not a new unclassified pole"}\n\n'
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
    payload["images"] = MAP_IMAGE_GUIDE
    return (
        "You receive THREE images: (1) MAP OVERVIEW (2) MAP ZOOM (3) STREET VIEW crop.\n"
        "Classify the visible pole's type using STREET VIEW.\n"
        f"Pole to classify: {pole.pole_id if pole else 'unknown'} "
        "(the pole identified in the prior clear-view step).\n\n"
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
        '{"classified_pole_id":"must equal the pole you are classifying",'
        '"pole_type":"distribution_transformer|lamp_post|billboard_pole|low_tension_pole",'
        '"confidence":"low|medium|high","reason":"features of the TARGET pole only"}\n\n'
        f"Context:\n{json.dumps(payload, indent=2)}"
    )


def navigation_allowed_actions(world: World, state: AgentState) -> list[str]:
    actions = ["turn_left", "turn_right"]
    if get_neighbors(world.neighbor_map, state.pano_id):
        actions.insert(2, "move")
    return actions

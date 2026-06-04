"""
All VLM prompt text — primary place to tune behavior.

POLE_TYPE_GUIDE: edit definitions for classification / clear-view.
build_map_navigation_prompt / build_pole_in_clear_view_prompt: JSON schemas the model must follow.
"""

from __future__ import annotations

import json

from agent.environment import World
from agent.map_render import MAP_SIZE, NAV_STREETS_RADIUS_M
from agent.graph import get_neighbors
from agent.observations import state_to_json
from agent.types import POLE_TYPES, AgentState, PoleType

DUAL_IMAGE_GUIDE = {
    "image_1": (
        f"STREETS MAP — {MAP_SIZE}x{MAP_SIZE} px OpenStreetMap; poles marked; no pano nodes"
    ),
    "image_2": "STREET VIEW — panorama crop from your current position and facing",
    "use_both": (
        "Use the STREETS MAP for context; use STREET VIEW for whether a pole is visible."
    ),
}

NAV_IMAGE_GUIDE_STREETS_AND_STREET = {
    "image_1": (
        f"STREETS MAP — {MAP_SIZE}x{MAP_SIZE} px OpenStreetMap (same style as UI 'Streets'); "
        f"blue circle = {NAV_STREETS_RADIUS_M} m from YOU; orange = target pole"
    ),
    "image_2": "STREET VIEW — panorama crop from your current position and facing",
    "use_both": (
        f"For move, pick map_point_x and map_point_y on image 1 on a visible street "
        f"INSIDE the {NAV_STREETS_RADIUS_M} m blue circle (toward the orange pole). "
        "The agent moves to the nearest panorama node to that point. "
        "Use STREET VIEW for turn_left/turn_right before moving or when street context matters."
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
) -> str:
    payload = state_to_json(world, state, pole_in_clear_view=pole_in_clear_view)
    neighbors = get_neighbors(world.neighbor_map, state.pano_id)
    payload["neighbor_pano_ids"] = neighbors
    payload["neighbor_count"] = len(neighbors)
    payload["allowed_actions"] = allowed_actions
    payload["map_image_size_px"] = MAP_SIZE
    payload["move_pick_max_radius_m"] = NAV_STREETS_RADIUS_M
    payload["pole_types"] = list(POLE_TYPES)

    image_guide = NAV_IMAGE_GUIDE_STREETS_AND_STREET
    payload["images"] = image_guide
    payload["map_legend"] = {
        "streets_basemap": "OpenStreetMap tiles (UI 'Streets' style); no routes or pano nodes drawn",
        "you": "blue dot and wedge — current position and viewing direction",
        "blue_circle": f"{NAV_STREETS_RADIUS_M} m radius — move picks must be inside on a street",
        "orange": "target pole to find",
        "green": "other unclassified poles",
        "gray": "classified poles",
    }
    image_intro = (
        "You receive TWO images: (1) STREETS MAP (2) STREET VIEW.\n"
        "Decide where to go next (only when pole_in_clear_view is false).\n"
    )
    rules = [
        image_guide["use_both"],
        "pole_in_clear_view is set by a prior VLM check (not your action).",
        "If true in state JSON, the agent classifies the visible pole; you only navigate when false.",
        "STREETS MAP: for move, set map_point_x and map_point_y (integers, 0–899, top-left origin) "
        f"on a street inside the {NAV_STREETS_RADIUS_M} m blue circle, generally toward the orange pole.",
        "No suggested path is shown — use street geometry and pole positions only.",
        "STREET VIEW: turn_left/turn_right before moving or when street context matters.",
        "classify_or_stop is NOT allowed in this step.",
    ]

    payload["rules"] = rules
    return (
        "You control a street panorama agent.\n"
        f"{image_intro}"
        "Choose exactly one action as JSON.\n\n"
        "Schema:\n"
        '{"action":"turn_left|turn_right|move",'
        f'"map_point_x":0-{MAP_SIZE - 1} or null,'
        f'"map_point_y":0-{MAP_SIZE - 1} or null,'
        '"reason":"short"}\n\n'
        "For turn_left/turn_right, set map_point_x and map_point_y to null.\n"
        "For move, both map_point_x and map_point_y are required.\n\n"
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
    payload["images"] = DUAL_IMAGE_GUIDE
    payload["map_legend"] = {
        "green": "unclassified poles (candidates you may identify)",
        "gray": "already classified — do NOT set pole_in_clear_view for these",
        "orange": "navigation hint only (where the agent is heading)",
        "wedge": "your viewing direction (street view)",
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
        "You receive TWO images: (1) LOCAL MAP with pole labels (2) STREET VIEW ahead.\n\n"
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
    payload["images"] = DUAL_IMAGE_GUIDE
    return (
        "You receive TWO images: (1) MAP (2) STREET VIEW crop.\n"
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

"""
Clear-view gate — VLM only (map + street).

The model decides pole_in_clear_view: a utility pole is clearly visible in street view
and visible_pole_id is an unclassified pole (not in already_classified_pole_ids).

Step order: apply_consideration (nav hint) → observe → choose (classify or navigate).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.pole_ids import track_id_for_pole_id
from agent.prompts import build_pole_in_clear_view_prompt
from agent.types import AgentState, PoleType


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    return False


def _classified_pole_ids(world: World, state: AgentState) -> frozenset[str]:
    ids: set[str] = set()
    for track_id in state.classified:
        pole = world.poles_by_track.get(track_id)
        if pole:
            ids.add(pole.pole_id)
    return frozenset(ids)


def evaluate_pole_in_clear_view(
    client: VlmClient,
    world: World,
    state: AgentState,
    overview_path: Path,
    zoom_path: Path,
    street_path: Path,
    *,
    parse_retries: int = 2,
    record_vlm: Callable[..., None] | None = None,
) -> tuple[bool, PoleType | None, str | None, str, str]:
    """
    Returns (clear, pole_type, visible_track_id, prompt, raw_response).
    """
    if world.is_task_complete(state):
        return False, None, None, "", ""

    classified_ids = _classified_pole_ids(world, state)
    prompt = build_pole_in_clear_view_prompt(world, state, classified_pole_ids=classified_ids)

    last_error = ""
    last_raw = ""

    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = (
                f"\n\nInvalid ({last_error}). If you see a NEW unclassified pole, reply "
                '{"pole_in_clear_view":true,"visible_pole_id":"POLE_...","identifiable_pole_type":"..."}. '
                "visible_pole_id must be green on the map and NOT in already_classified_pole_ids. "
                "If no new pole is visible, pole_in_clear_view false."
            )
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [overview_path, zoom_path, street_path])
        last_raw = raw
        if record_vlm is not None:
            record_vlm("pole_in_clear_view", raw, attempt=attempt + 1)
        clear, pole_type, track_id, err = parse_pole_in_clear_view_response(
            raw,
            classified_pole_ids=classified_ids,
            resolve_track_id=lambda pid: track_id_for_pole_id(world, pid),
        )
        if err:
            last_error = err
            continue
        return bool(clear), pole_type, track_id, full_prompt, raw

    return False, None, None, prompt, last_raw

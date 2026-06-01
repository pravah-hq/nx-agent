"""
Target pole clear-view gate — VLM only (map + street images).

Step order in AgentLoop: apply_consideration (pick target) → observe (this module) →
choose (classify if clear, else navigate).

Does not set target_pole_in_clear_view from geometry; sight data is only context in the prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.prompts import build_pole_in_clear_view_prompt
from agent.sight import compute_target_pole_sight
from agent.types import AgentState, PoleInView, PoleType


def target_pole_sight(world: World, state: AgentState) -> PoleInView | None:
    """Optional geometry hints for the VLM prompt (not used to force clear view)."""
    return compute_target_pole_sight(world, state)


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    """Non-VLM paths never auto-clear from geometry."""
    return False


def evaluate_pole_in_clear_view(
    client: VlmClient,
    world: World,
    state: AgentState,
    map_path: Path,
    street_path: Path,
    *,
    parse_retries: int = 2,
    record_vlm: Callable[..., None] | None = None,
) -> tuple[bool, PoleType | None, str, str]:
    track = state.pole_in_consideration
    if not track or track in state.classified:
        return False, None, "", ""

    pole = world.poles_by_track.get(track)
    if not pole:
        return False, None, "", ""

    sight = compute_target_pole_sight(world, state)
    prompt = build_pole_in_clear_view_prompt(world, state, target_sight=sight)
    expected_id = pole.pole_id

    last_error = ""
    last_raw = ""

    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = (
                f"\n\nInvalid ({last_error}). If the TARGET pole is visible in street view, "
                f'reply {{"pole_in_clear_view":true,"identifiable_pole_type":"<one of four types>"}}. '
                f"If not visible, pole_in_clear_view false. Target id: {expected_id}."
            )
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [map_path, street_path])
        last_raw = raw
        if record_vlm is not None:
            record_vlm("pole_in_clear_view", raw, attempt=attempt + 1)
        clear, pole_type, err = parse_pole_in_clear_view_response(
            raw,
            expected_pole_id=expected_id,
        )
        if err:
            last_error = err
            continue
        return bool(clear), pole_type, full_prompt, raw

    return False, None, prompt, last_raw

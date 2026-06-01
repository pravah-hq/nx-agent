"""
Target pole "clear view" gate — VLM must say type is unambiguous before classify.

Flow: VlmPolicy.observe() -> evaluate_pole_in_clear_view() -> parse_pole_in_clear_view_response()
Tweak prompts in prompts.build_pole_in_clear_view_prompt; parser in action_parse.
"""

from __future__ import annotations

from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.prompts import build_pole_in_clear_view_prompt
from agent.types import AgentState, PoleInView, PoleType


def target_pole_sight(world: World, state: AgentState) -> PoleInView | None:
    """Geometric visibility entry for the current target pole, if any."""
    track = state.pole_in_consideration
    if not track:
        return None
    return next((p for p in world.poles_in_view(state) if p.track_id == track), None)


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    """Non-VLM policies: we do not auto-classify from geometry alone."""
    return False


def evaluate_pole_in_clear_view(
    client: VlmClient,
    world: World,
    state: AgentState,
    map_path: Path,
    street_path: Path,
    *,
    parse_retries: int = 2,
) -> tuple[bool, PoleType | None, str, str]:
    """
    Dual-image VLM call for clear-view gate.

    Returns:
        clear — may classify this step if True
        identifiable_pole_type — stored on VlmPolicy for classify_or_stop
        prompt, raw_response — for VLM_TRACE_DIR debugging
    """
    track = state.pole_in_consideration
    if not track or track in state.classified:
        return False, None, "", ""

    pole = world.poles_by_track.get(track)
    if not pole:
        return False, None, "", ""

    sight = target_pole_sight(world, state)
    prompt = build_pole_in_clear_view_prompt(world, state, target_sight=sight)
    expected_id = pole.pole_id

    last_error = ""
    last_raw = ""
    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = (
                f"\n\nInvalid ({last_error}). When unambiguous, reply with "
                f"pole_in_clear_view true, unambiguous_identifiable true, "
                f'identifiable_pole_type one of the four types, '
                f'confirmed_target_pole_id "{expected_id}". '
                "If unsure between types, all must be false/null."
            )
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [map_path, street_path])
        last_raw = raw
        clear, pole_type, err = parse_pole_in_clear_view_response(
            raw,
            expected_pole_id=expected_id,
        )
        if err:
            last_error = err
            continue
        return bool(clear), pole_type, full_prompt, raw

    return False, None, prompt, last_raw

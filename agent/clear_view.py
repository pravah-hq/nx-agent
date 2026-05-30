from __future__ import annotations

from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.prompts import build_pole_in_clear_view_prompt
from agent.types import AgentState, PoleInView


def target_pole_sight(world: World, state: AgentState) -> PoleInView | None:
    """Geometric viewshed: is pole_in_consideration in the 100° cone within 42 m?"""
    track = state.pole_in_consideration
    if not track:
        return None
    return next((p for p in world.poles_in_view(state) if p.track_id == track), None)


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    """Stub: target pole in viewshed and within classify distance."""
    sight = target_pole_sight(world, state)
    return sight is not None and sight.distance_m <= 25


def evaluate_pole_in_clear_view(
    client: VlmClient,
    world: World,
    state: AgentState,
    map_path: Path,
    street_path: Path,
    *,
    parse_retries: int = 2,
) -> tuple[bool, str, str]:
    """
    VLM checks whether pole_in_consideration (target) is in clear view — not any pole.
    """
    track = state.pole_in_consideration
    if not track or track in state.classified:
        return False, "", ""

    pole = world.poles_by_track.get(track)
    if not pole:
        return False, "", ""

    sight = target_pole_sight(world, state)
    if sight is None:
        return (
            False,
            "",
            '{"pole_in_clear_view":false,"reason":"target not in geometric viewshed"}',
        )

    prompt = build_pole_in_clear_view_prompt(world, state, target_sight=sight)
    expected_id = pole.pole_id
    last_error = ""
    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = (
                f"\n\nInvalid ({last_error}). JSON must include pole_in_clear_view "
                f"and confirmed_target_pole_id (must be {expected_id} when true)."
            )
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [map_path, street_path])
        clear, err = parse_pole_in_clear_view_response(raw, expected_pole_id=expected_id)
        if err:
            last_error = err
            continue
        return bool(clear), full_prompt, raw
    return False, prompt, ""

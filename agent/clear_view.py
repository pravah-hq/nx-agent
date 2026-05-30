from __future__ import annotations

from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.prompts import build_pole_in_clear_view_prompt
from agent.sight import (
    geometric_sight_clear,
    target_pole_primary_in_viewshed,
)
from agent.types import AgentState, PoleInView


def target_pole_sight(world: World, state: AgentState) -> PoleInView | None:
    """Target pole (pole_in_consideration) in the 100° cone within 42 m viewshed."""
    track = state.pole_in_consideration
    if not track:
        return None
    return next((p for p in world.poles_in_view(state) if p.track_id == track), None)


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    """Stub: target in viewshed and close enough to classify."""
    sight = target_pole_sight(world, state)
    if sight is None:
        return False
    return geometric_sight_clear(sight)


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
    VLM: is pole_in_consideration in clear view (not any pole).
    No hard block when target is outside viewshed — VLM may still say false.
    """
    track = state.pole_in_consideration
    if not track or track in state.classified:
        return False, "", ""

    pole = world.poles_by_track.get(track)
    if not pole:
        return False, "", ""

    sight = target_pole_sight(world, state)
    prompt = build_pole_in_clear_view_prompt(world, state, target_sight=sight)
    expected_id = pole.pole_id

    last_error = ""
    last_raw = ""
    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = (
                f"\n\nInvalid ({last_error}). JSON: "
                f'{{"pole_in_clear_view":bool,"confirmed_target_pole_id":"{expected_id}"|null,"reason":"..."}}'
            )
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [map_path, street_path])
        last_raw = raw
        clear, err = parse_pole_in_clear_view_response(
            raw,
            expected_pole_id=expected_id,
            geometric_sight=sight,
        )
        if err:
            last_error = err
            continue
        if clear:
            return True, full_prompt, raw

        # Target is in the viewshed cone but VLM said false — one simpler retry.
        if geometric_sight_clear(sight) and attempt == 0:
            retry_prompt = (
                f"The target pole {expected_id} is in the viewshed at "
                f"{sight.distance_m:.0f} m, {sight.angle_from_view_deg:.0f} deg from view center. "
                "Look at STREET VIEW (image 2). If that pole is visible and classifiable, "
                f'reply {{"pole_in_clear_view":true,"confirmed_target_pole_id":"{expected_id}","reason":"..."}}. '
                "Otherwise false.\n"
            )
            raw2 = client.complete_images(retry_prompt, [map_path, street_path])
            last_raw = raw2
            clear2, err2 = parse_pole_in_clear_view_response(
                raw2,
                expected_pole_id=expected_id,
                geometric_sight=sight,
            )
            if not err2 and clear2:
                return True, retry_prompt, raw2

        last_error = "pole_in_clear_view is false"
        continue

    # VLM said false or failed to parse — allow geometry only for the target pole.
    if target_pole_primary_in_viewshed(world, state):
        return (
            True,
            prompt,
            last_raw
            or '{"pole_in_clear_view":true,"reason":"target primary in viewshed"}',
        )

    return False, prompt, last_raw

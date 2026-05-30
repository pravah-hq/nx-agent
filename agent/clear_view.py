from __future__ import annotations

from pathlib import Path

from agent.action_parse import parse_pole_in_clear_view_response
from agent.environment import World
from agent.model_client import VlmClient
from agent.prompts import build_pole_in_clear_view_prompt
from agent.types import AgentState


def evaluate_pole_in_clear_view(
    client: VlmClient,
    world: World,
    state: AgentState,
    street_path: Path,
    *,
    parse_retries: int = 2,
) -> tuple[bool, str, str]:
    """VLM reads street view; returns (clear, prompt, raw_response)."""
    if not state.pole_in_consideration:
        return False, "", ""

    prompt = build_pole_in_clear_view_prompt(world, state)
    last_error = ""
    for attempt in range(parse_retries + 1):
        extra = ""
        if attempt > 0:
            extra = f"\n\nInvalid ({last_error}). JSON: pole_in_clear_view boolean only."
        full_prompt = prompt + extra
        raw = client.complete_images(full_prompt, [street_path])
        clear, err = parse_pole_in_clear_view_response(raw)
        if err:
            last_error = err
            continue
        return bool(clear), full_prompt, raw
    return False, prompt, ""


def geometric_pole_in_clear_view(world: World, state: AgentState) -> bool:
    """Stub policy: geometric viewshed + distance threshold."""
    if not state.pole_in_consideration:
        return False
    visible = world.poles_in_view(state)
    match = next(
        (p for p in visible if p.track_id == state.pole_in_consideration),
        None,
    )
    return match is not None and match.distance_m <= 25

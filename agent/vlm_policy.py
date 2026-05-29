from __future__ import annotations

import os
from pathlib import Path

from agent.action_parse import parse_action_response
from agent.environment import World
from agent.policy import Policy
from agent.model_client import VlmClient
from agent.prompts import allowed_actions, build_vlm_prompt
from agent.types import Action, ActionType, AgentState, PoleGuess, PoleType
from agent.views import render_direction_crop


class VlmPolicy(Policy):
    """Vision-language policy: crop view + Qwen3-VL JSON action."""

    def __init__(
        self,
        client: VlmClientProtocol | None = None,
        *,
        parse_retries: int = 2,
        fallback_type: PoleType = "lamp_post",
    ) -> None:
        self.client = client or VlmClient()
        self.parse_retries = parse_retries
        self.fallback_type = fallback_type
        self.last_prompt: str = ""
        self.last_response: str = ""
        self.last_image: Path | None = None

    def reset(self) -> None:
        self.last_prompt = ""
        self.last_response = ""
        self.last_image = None

    def choose(self, world: World, state: AgentState, poles_in_view) -> Action:
        if world.is_task_complete(state):
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        pano = world.panos_by_id[state.pano_id]
        image_path = render_direction_crop(pano, state.direction_bin)
        self.last_image = image_path

        legal = allowed_actions(world, state, poles_in_view)
        prompt = build_vlm_prompt(world, state, poles_in_view, allowed_actions=legal)
        self.last_prompt = prompt

        trace_dir = os.environ.get("VLM_TRACE_DIR")
        if trace_dir:
            self._write_trace(trace_dir, state, prompt, image_path)

        last_error = ""
        for attempt in range(self.parse_retries + 1):
            extra = ""
            if attempt > 0:
                extra = (
                    f"\n\nYour previous answer was invalid ({last_error}). "
                    f"Reply with JSON only. Allowed actions: {', '.join(legal)}"
                )
            raw = self.client.complete(prompt + extra, image_path)
            self.last_response = raw
            action = parse_action_response(raw, allowed=legal)
            if action is None:
                last_error = "could not parse JSON action"
                continue
            if action.type == ActionType.CLASSIFY_OR_STOP and action.pole_type is None:
                action = Action(
                    type=action.type,
                    pole_type=self.fallback_type,
                    stop_after=action.stop_after,
                )
            if action.type == ActionType.CLASSIFY_OR_STOP and not state.pole_in_consideration:
                if poles_in_view:
                    # Classify will run after apply_consideration sync on next step;
                    # loop calls consideration before choose, so set via classify target below.
                    pass
                else:
                    last_error = "classify_or_stop but no pole in view"
                    continue
            return action

        return Action(type=ActionType.TURN_RIGHT)

    def _write_trace(
        self,
        trace_dir: str,
        state: AgentState,
        prompt: str,
        image_path: Path,
    ) -> None:
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{state.pano_id.replace('/', '_')}_bin{state.direction_bin}"
        (root / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
        if self.last_response:
            (root / f"{stem}.response.txt").write_text(self.last_response, encoding="utf-8")


def apply_vlm_consideration(
    state: AgentState,
    world: World,
    poles_in_view,
    *,
    fallback_type: PoleType = "lamp_post",
) -> AgentState:
    next_state = state.copy()
    track_id = next_state.pole_in_consideration
    if track_id and track_id in next_state.classified:
        track_id = None
    if not track_id and poles_in_view:
        track_id = poles_in_view[0].track_id
    if not track_id:
        return next_state

    pole = world.poles_by_track[track_id]
    next_state.pole_in_consideration = track_id
    next_state.pole_guess = PoleGuess(
        track_id=pole.track_id,
        pole_id=pole.pole_id,
        pole_type=fallback_type,
        note="vlm hypothesis",
    )
    return next_state

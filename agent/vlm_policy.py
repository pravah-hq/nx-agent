from __future__ import annotations

import os
from pathlib import Path

from agent.action_parse import extract_json_object, parse_navigation_response
from agent.environment import World
from agent.graph import get_neighbors
from agent.model_client import VlmClient
from agent.map_render import render_map_image
from agent.navigation import (
    backtrack_blocked_ids,
    build_neighbor_move_options,
    pick_planned_neighbor,
)
from agent.targeting import next_path_hop, plan_mission_to_pole, select_target_pole
from agent.policy import Policy
from agent.prompts import (
    build_classify_assessment_prompt,
    build_map_navigation_prompt,
    navigation_allowed_actions,
)
from agent.types import Action, ActionType, AgentState, PoleGuess, PoleType
from agent.views import render_direction_crop


class VlmPolicy(Policy):
    """
    Map-based VLM navigation + street-view assessment before classify.
    No graph planner; the model reads the map screenshot to choose moves and facing.
    """

    def __init__(
        self,
        client: VlmClient | None = None,
        *,
        parse_retries: int = 2,
        fallback_type: PoleType = "lamp_post",
    ) -> None:
        self.client = client or VlmClient()
        self.parse_retries = parse_retries
        self.fallback_type = fallback_type
        self.last_prompt: str = ""
        self.last_response: str = ""
        self.last_map_image: Path | None = None
        self.last_street_image: Path | None = None
        self.last_phase: str = ""
        self._last_pano_id: str | None = None
        self._goal_pano_id: str | None = None
        self._nav_path: list[str] = []

    def reset(self) -> None:
        self.last_prompt = ""
        self.last_response = ""
        self.last_map_image = None
        self.last_street_image = None
        self.last_phase = ""
        self._last_pano_id = None
        self._goal_pano_id = None
        self._nav_path = []

    def record_step(self, before: AgentState, action: Action, after: AgentState) -> None:
        if (
            action.type == ActionType.MOVE
            and before.pano_id != after.pano_id
        ):
            self._last_pano_id = before.pano_id
            if self._nav_path and after.pano_id == self._nav_path[0]:
                self._nav_path = self._nav_path[1:]
            if after.pano_id == self._goal_pano_id:
                self._nav_path = []

    def choose(self, world: World, state: AgentState, poles_in_view) -> Action:
        if world.is_task_complete(state):
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        self._ensure_navigation_plan(world, state)

        if self._target_pole_in_view(state, poles_in_view):
            action = self._assess_street_view(world, state, poles_in_view)
            self._maybe_trace(state)
            return action

        nav_action, assess = self._navigate_from_map(world, state, poles_in_view)
        self._maybe_trace(state)
        if assess:
            action = self._assess_street_view(world, state, poles_in_view)
            self._maybe_trace(state)
            return action

        if nav_action is not None:
            return nav_action

        return Action(type=ActionType.TURN_RIGHT)

    def _target_pole_in_view(self, state: AgentState, poles_in_view) -> bool:
        track_id = state.pole_in_consideration
        if not track_id or track_id in state.classified:
            return False
        return any(p.track_id == track_id for p in poles_in_view)

    def _ensure_navigation_plan(self, world: World, state: AgentState) -> None:
        track = state.pole_in_consideration
        if not track or track in state.classified:
            track = select_target_pole(world, state)
            if track:
                state.pole_in_consideration = track
            else:
                self._goal_pano_id = None
                self._nav_path = []
                return

        goal, path = plan_mission_to_pole(world, state, track)
        self._goal_pano_id = goal
        self._nav_path = path[1:] if path else []

    def _navigate_from_map(
        self,
        world: World,
        state: AgentState,
        poles_in_view,
    ) -> tuple[Action | None, bool]:
        map_path = render_map_image(
            world,
            state,
            poles_in_view,
            goal_pano_id=self._goal_pano_id,
            nav_path=self._nav_path,
        )
        self.last_map_image = map_path
        self.last_street_image = None
        self.last_phase = "map_navigation"

        neighbors = get_neighbors(world.neighbor_map, state.pano_id)
        blocked = backtrack_blocked_ids(self._last_pano_id)
        legal = navigation_allowed_actions(world, state)
        full_path = [state.pano_id, *self._nav_path] if self._nav_path else [state.pano_id]
        prompt = build_map_navigation_prompt(
            world,
            state,
            poles_in_view,
            allowed_actions=legal,
            blocked_move_targets=sorted(blocked) if blocked else None,
            neighbor_moves=build_neighbor_move_options(
                world,
                state,
                neighbors,
                blocked,
                nav_path=full_path,
            ),
            goal_pano_id=self._goal_pano_id,
            planned_next_hop=next_path_hop(full_path),
        )
        self.last_prompt = prompt

        last_error = ""
        for attempt in range(self.parse_retries + 1):
            extra = ""
            if attempt > 0:
                extra = (
                    f"\n\nPrevious reply invalid ({last_error}). JSON only. "
                    f"Allowed: {', '.join(legal)}. move needs target_pano_id from neighbors."
                )
                if blocked:
                    extra += (
                        f" Do not move to blocked_move_targets: "
                        f"{', '.join(sorted(blocked))}."
                    )
            raw = self.client.complete_images(prompt + extra, [map_path])
            self.last_response = raw
            action, assess = parse_navigation_response(
                raw,
                allowed=legal,
                neighbor_ids=neighbors,
                blocked_move_targets=blocked,
            )
            if assess:
                return None, True
            if action is not None:
                return action, False
            last_error = "could not parse navigation JSON"
            if blocked and "move" in legal:
                last_error = "invalid action or backtrack move to previous pano"

        full_path = [state.pano_id, *self._nav_path] if self._nav_path else [state.pano_id]
        fallback = pick_planned_neighbor(
            world,
            state,
            neighbors,
            blocked,
            nav_path=full_path,
        )
        if fallback:
            return Action(type=ActionType.MOVE, target_pano_id=fallback), False
        return None, False

    def _assess_street_view(
        self,
        world: World,
        state: AgentState,
        poles_in_view,
    ) -> Action:
        self.last_phase = "classify_assessment"
        if not state.pole_in_consideration:
            return Action(type=ActionType.TURN_RIGHT)

        pano = world.panos_by_id[state.pano_id]
        street_path = render_direction_crop(pano, state.direction_bin)
        self.last_street_image = street_path

        prompt = build_classify_assessment_prompt(world, state, poles_in_view)
        self.last_prompt = prompt

        last_error = ""
        for attempt in range(self.parse_retries + 1):
            extra = ""
            if attempt > 0:
                extra = f"\n\nPrevious reply invalid ({last_error}). JSON with view_clear and pole_type."
            raw = self.client.complete_images(prompt + extra, [street_path])
            self.last_response = raw
            payload = extract_json_object(raw)
            if not payload:
                last_error = "no JSON"
                continue

            view_clear = payload.get("view_clear") in (True, "true", "True", 1, "1")
            if not view_clear:
                return Action(
                    type=ActionType.TURN_RIGHT,
                )

            raw_type = payload.get("pole_type")
            pole_type = self.fallback_type
            if raw_type is not None and str(raw_type).lower() not in {"null", "none", ""}:
                candidate = str(raw_type).strip().lower()
                from agent.types import POLE_TYPES

                if candidate not in POLE_TYPES:
                    last_error = f"invalid pole_type {candidate}"
                    continue
                pole_type = candidate  # type: ignore[assignment]

            would_complete = len(state.classified) + 1 >= len(world.poles)
            stop_after = bool(payload.get("stop_after", False)) or would_complete
            return Action(
                type=ActionType.CLASSIFY_OR_STOP,
                pole_type=pole_type,
                stop_after=stop_after,
            )

        return Action(type=ActionType.TURN_RIGHT)

    def _maybe_trace(self, state: AgentState) -> None:
        trace_dir = os.environ.get("VLM_TRACE_DIR")
        if not trace_dir:
            return
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{state.pano_id.replace('/', '_')}_bin{state.direction_bin}_{self.last_phase}"
        (root / f"{stem}.prompt.txt").write_text(self.last_prompt, encoding="utf-8")
        if self.last_response:
            (root / f"{stem}.response.txt").write_text(self.last_response, encoding="utf-8")
        image = self.last_street_image or self.last_map_image
        if image and image.is_file():
            dest = root / f"{stem}{image.suffix}"
            dest.write_bytes(image.read_bytes())


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
    if not track_id:
        track_id = select_target_pole(world, next_state)
    if not track_id:
        return next_state

    pole = world.poles_by_track[track_id]
    next_state.pole_in_consideration = track_id
    next_state.pole_guess = PoleGuess(
        track_id=pole.track_id,
        pole_id=pole.pole_id,
        pole_type=fallback_type,
        note="vlm map-navigation target",
    )
    return next_state

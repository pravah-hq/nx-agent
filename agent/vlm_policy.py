"""
VLM policy: dual map + street images every step.

Tinker here:
  observe()     — clear-view gate (evaluate_pole_in_clear_view)
  choose()      — classify if clear, else _navigate()
  prompts       — agent/prompts.py (POLE_TYPE_GUIDE, build_*_prompt)
  parsers       — agent/action_parse.py
  dry run       — VLM_DRY_RUN=1 (model_client keyword stubs)

Env: VLM_TRACE_DIR, VLM_ASSESS_CROP_FOV
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.action_parse import parse_navigation_response, parse_pole_type_response
from agent.clear_view import evaluate_pole_in_clear_view
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
    build_map_navigation_prompt,
    build_pole_type_classification_prompt,
    navigation_allowed_actions,
)
from agent.types import Action, ActionType, AgentState, PoleGuess, PoleType
from agent.views import render_direction_crop


class VlmPolicy(Policy):
    """
    MAP + STREET VIEW each step. pole_in_clear_view comes from VLM street view;
    when true, classify (no assess_classify action).
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
        self.last_pole_in_clear_view: bool = False
        self.last_identifiable_pole_type: PoleType | None = None
        self._last_pano_id: str | None = None
        self._goal_pano_id: str | None = None
        self._nav_path: list[str] = []
        self._cached_pano_id: str | None = None
        self._cached_direction_bin: int | None = None
        self._cached_map_path: Path | None = None
        self._cached_street_path: Path | None = None

    def reset(self) -> None:
        self.last_prompt = ""
        self.last_response = ""
        self.last_map_image = None
        self.last_street_image = None
        self.last_phase = ""
        self.last_pole_in_clear_view = False
        self.last_identifiable_pole_type = None
        self._last_pano_id = None
        self._goal_pano_id = None
        self._nav_path = []
        self._cached_pano_id = None
        self._cached_direction_bin = None
        self._cached_map_path = None
        self._cached_street_path = None

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
        self._invalidate_cache()

    def observe(self, world: World, state: AgentState) -> bool:
        """VLM: target pole unambiguously identifiable as one pole type?"""
        self._ensure_navigation_plan(world, state)
        map_path, street_path = self._render_dual_observation(world, state)
        self.last_phase = "pole_in_clear_view"
        clear, pole_type, prompt, raw = evaluate_pole_in_clear_view(
            self.client,
            world,
            state,
            map_path,
            street_path,
            parse_retries=self.parse_retries,
        )
        self.last_prompt = prompt
        self.last_response = raw
        self.last_pole_in_clear_view = clear
        self.last_identifiable_pole_type = pole_type if clear else None
        return clear

    def choose(self, world: World, state: AgentState, pole_in_clear_view: bool) -> Action:
        if world.is_task_complete(state):
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        self._ensure_navigation_plan(world, state)
        map_path, street_path = self._render_dual_observation(world, state)

        if pole_in_clear_view:
            pole_type = self.last_identifiable_pole_type
            if pole_type is None:
                pole_type = self._classify_pole_type(
                    world, state, map_path, street_path, pole_in_clear_view=True
                )
            self._maybe_trace(state)
            if pole_type is None:
                return Action(type=ActionType.TURN_RIGHT)
            would_complete = len(state.classified) + 1 >= len(world.poles)
            return Action(
                type=ActionType.CLASSIFY_OR_STOP,
                pole_type=pole_type,
                stop_after=would_complete,
            )

        nav_action = self._navigate(
            world, state, pole_in_clear_view, map_path, street_path
        )
        self._maybe_trace(state)
        if nav_action is not None:
            return nav_action
        return Action(type=ActionType.TURN_RIGHT)

    def _invalidate_cache(self) -> None:
        self._cached_pano_id = None
        self._cached_direction_bin = None
        self._cached_map_path = None
        self._cached_street_path = None

    def _assess_crop_fov_deg(self) -> float:
        return float(os.environ.get("VLM_ASSESS_CROP_FOV", "100"))

    def _render_dual_observation(
        self,
        world: World,
        state: AgentState,
    ) -> tuple[Path, Path]:
        if (
            self._cached_map_path
            and self._cached_street_path
            and self._cached_pano_id == state.pano_id
            and self._cached_direction_bin == state.direction_bin
        ):
            self.last_map_image = self._cached_map_path
            self.last_street_image = self._cached_street_path
            return self._cached_map_path, self._cached_street_path

        map_path = render_map_image(
            world,
            state,
            goal_pano_id=self._goal_pano_id,
            nav_path=self._nav_path,
        )
        pano = world.panos_by_id[state.pano_id]
        street_path = render_direction_crop(
            pano,
            state.direction_bin,
            crop_fov_deg=self._assess_crop_fov_deg(),
        )
        self._cached_pano_id = state.pano_id
        self._cached_direction_bin = state.direction_bin
        self._cached_map_path = map_path
        self._cached_street_path = street_path
        self.last_map_image = map_path
        self.last_street_image = street_path
        return map_path, street_path

    def _vlm_dual(
        self,
        prompt: str,
        map_path: Path,
        street_path: Path,
    ) -> str:
        raw = self.client.complete_images(prompt, [map_path, street_path])
        self.last_response = raw
        return raw

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

    def _navigate(
        self,
        world: World,
        state: AgentState,
        pole_in_clear_view: bool,
        map_path: Path,
        street_path: Path,
    ) -> Action | None:
        self.last_phase = "dual_navigation"

        neighbors = get_neighbors(world.neighbor_map, state.pano_id)
        blocked = backtrack_blocked_ids(self._last_pano_id)
        legal = navigation_allowed_actions(world, state)
        full_path = [state.pano_id, *self._nav_path] if self._nav_path else [state.pano_id]
        prompt = build_map_navigation_prompt(
            world,
            state,
            pole_in_clear_view=pole_in_clear_view,
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
            self.last_prompt = prompt + extra
            raw = self._vlm_dual(self.last_prompt, map_path, street_path)
            action = parse_navigation_response(
                raw,
                allowed=legal,
                neighbor_ids=neighbors,
                blocked_move_targets=blocked,
            )
            if action is not None:
                return action
            last_error = "could not parse navigation JSON"
            if blocked and "move" in legal:
                last_error = "invalid action or backtrack move to previous pano"

        fallback = pick_planned_neighbor(
            world,
            state,
            neighbors,
            blocked,
            nav_path=full_path,
        )
        if fallback:
            return Action(type=ActionType.MOVE, target_pano_id=fallback)
        return None

    def _classify_pole_type(
        self,
        world: World,
        state: AgentState,
        map_path: Path,
        street_path: Path,
        *,
        pole_in_clear_view: bool,
    ) -> PoleType | None:
        self.last_phase = "dual_pole_type_classification"
        prompt = build_pole_type_classification_prompt(
            world, state, pole_in_clear_view=pole_in_clear_view
        )
        last_error = ""
        for attempt in range(self.parse_retries + 1):
            extra = ""
            if attempt > 0:
                extra = (
                    f"\n\nInvalid ({last_error}). Pick one specific pole_type; "
                    "do not default to lamp_post unless a street light is clearly on top."
                )
            self.last_prompt = prompt + extra
            pole = world.poles_by_track.get(state.pole_in_consideration or "")
            if not pole:
                return None
            raw = self._vlm_dual(self.last_prompt, map_path, street_path)
            pole_type, err = parse_pole_type_response(
                raw, expected_pole_id=pole.pole_id
            )
            if err:
                last_error = err
                continue
            return pole_type
        return None

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
        for label, image in (("map", self.last_map_image), ("street", self.last_street_image)):
            if image and image.is_file():
                dest = root / f"{stem}_{label}{image.suffix}"
                dest.write_bytes(image.read_bytes())


def apply_vlm_consideration(
    state: AgentState,
    world: World,
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
        pole_type=None,
        note="vlm navigation target (type chosen from street view only)",
    )
    return next_state

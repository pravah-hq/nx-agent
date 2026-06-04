"""
VLM policy: OSM Streets map (map-point moves) + street images for navigation.

Tinker here:
  observe()     — VLM clear-view gate only (after apply_consideration in loop)
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
from agent.map_render import OverviewMapBounds, render_map_overview_image
from agent.road_geometry import closest_neighbor_to_road_pick
from agent.targeting import plan_mission_to_pole, select_target_pole
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
        self._last_overview_bounds: OverviewMapBounds | None = None
        self.last_phase: str = ""
        self.last_pole_in_clear_view: bool = False
        self.last_identifiable_pole_type: PoleType | None = None
        self.last_visible_track_id: str | None = None
        self._goal_pano_id: str | None = None
        self._cached_pano_id: str | None = None
        self._cached_direction_bin: int | None = None
        self._cached_map_overview_path: Path | None = None
        self._cached_street_path: Path | None = None
        self._cached_overview_bounds: OverviewMapBounds | None = None
        self._cached_road_segments: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self._last_road_segments: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self.vlm_step_calls: list[dict[str, str | int]] = []

    def begin_agent_step(self) -> None:
        """Clear per-step VLM log; call once at the start of each agent step."""
        self.vlm_step_calls = []

    def _record_vlm(
        self,
        phase: str,
        response: str,
        *,
        attempt: int = 1,
    ) -> None:
        self.vlm_step_calls.append(
            {"phase": phase, "attempt": attempt, "response": response}
        )
        self.last_response = response

    def reset(self) -> None:
        self.last_prompt = ""
        self.last_response = ""
        self.last_map_image = None
        self.last_street_image = None
        self._last_overview_bounds = None
        self.last_phase = ""
        self.last_pole_in_clear_view = False
        self.last_identifiable_pole_type = None
        self.last_visible_track_id = None
        self._goal_pano_id = None
        self._cached_pano_id = None
        self._cached_direction_bin = None
        self._cached_map_overview_path = None
        self._cached_street_path = None
        self._cached_overview_bounds = None
        self._cached_road_segments = []
        self._last_road_segments = []
        self.vlm_step_calls = []

    def record_step(self, before: AgentState, action: Action, after: AgentState) -> None:
        self._invalidate_cache()

    def observe(self, world: World, state: AgentState) -> bool:
        """VLM: is an unclassified pole clearly visible in street view?"""
        self._ensure_navigation_plan(world, state)
        map_path, street_path = self._render_dual_observation(world, state)
        self.last_phase = "pole_in_clear_view"
        clear, pole_type, track_id, prompt, raw = evaluate_pole_in_clear_view(
            self.client,
            world,
            state,
            map_path,
            street_path,
            parse_retries=self.parse_retries,
            record_vlm=lambda phase, raw, attempt=1: self._record_vlm(
                phase, raw, attempt=attempt
            ),
        )
        self.last_prompt = prompt
        self.last_pole_in_clear_view = clear
        self.last_visible_track_id = track_id if clear else None
        self.last_identifiable_pole_type = pole_type if clear else None
        return clear

    def choose(self, world: World, state: AgentState, pole_in_clear_view: bool) -> Action:
        if world.is_task_complete(state):
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        self._ensure_navigation_plan(world, state)

        if pole_in_clear_view:
            map_path, street_path = self._render_dual_observation(world, state)
            track_id = self.last_visible_track_id
            if not track_id or track_id in state.classified:
                return Action(type=ActionType.TURN_RIGHT)
            state.pole_in_consideration = track_id
            pole = world.poles_by_track[track_id]
            state.pole_guess = PoleGuess(
                track_id=track_id,
                pole_id=pole.pole_id,
                pole_type=None,
                note="vlm visible pole",
            )
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

        nav_action = self._navigate(world, state, pole_in_clear_view)
        self._maybe_trace(state)
        if nav_action is not None:
            return nav_action
        return Action(type=ActionType.TURN_RIGHT)

    def _invalidate_cache(self) -> None:
        self._cached_pano_id = None
        self._cached_direction_bin = None
        self._cached_map_overview_path = None
        self._cached_street_path = None
        self._cached_overview_bounds = None
        self._cached_road_segments = []

    def _assess_crop_fov_deg(self) -> float:
        return float(os.environ.get("VLM_ASSESS_CROP_FOV", "100"))

    def _render_dual_observation(
        self,
        world: World,
        state: AgentState,
    ) -> tuple[Path, Path]:
        """Overview map + street (clear-view and classification)."""
        if (
            self._cached_map_overview_path
            and self._cached_street_path
            and self._cached_pano_id == state.pano_id
            and self._cached_direction_bin == state.direction_bin
        ):
            self.last_map_image = self._cached_map_overview_path
            self.last_street_image = self._cached_street_path
            return self._cached_map_overview_path, self._cached_street_path

        map_path, bounds, _ = render_map_overview_image(world, state)
        pano = world.panos_by_id[state.pano_id]
        street_path = render_direction_crop(
            pano,
            state.direction_bin,
            crop_fov_deg=self._assess_crop_fov_deg(),
        )
        self._cached_pano_id = state.pano_id
        self._cached_direction_bin = state.direction_bin
        self._cached_map_overview_path = map_path
        self._cached_overview_bounds = bounds
        self._cached_road_segments = []
        self._cached_street_path = street_path
        self._last_overview_bounds = bounds
        self._last_road_segments = []
        self.last_map_image = map_path
        self.last_street_image = street_path
        return map_path, street_path

    def _render_navigation_images(
        self,
        world: World,
        state: AgentState,
    ) -> tuple[Path, Path]:
        """Overview map + street for navigation."""
        if (
            self._cached_map_overview_path
            and self._cached_street_path
            and             self._cached_overview_bounds
            and self._cached_pano_id == state.pano_id
            and self._cached_direction_bin == state.direction_bin
        ):
            self.last_map_image = self._cached_map_overview_path
            self.last_street_image = self._cached_street_path
            self._last_overview_bounds = self._cached_overview_bounds
            self._last_road_segments = self._cached_road_segments
            return self._cached_map_overview_path, self._cached_street_path

        overview_path, bounds, segments = render_map_overview_image(
            world,
            state,
            for_navigation=True,
        )
        pano = world.panos_by_id[state.pano_id]
        street_path = render_direction_crop(
            pano,
            state.direction_bin,
            crop_fov_deg=self._assess_crop_fov_deg(),
        )
        self._cached_pano_id = state.pano_id
        self._cached_direction_bin = state.direction_bin
        self._cached_map_overview_path = overview_path
        self._cached_overview_bounds = bounds
        self._cached_road_segments = segments
        self._cached_street_path = street_path
        self.last_map_image = overview_path
        self.last_street_image = street_path
        self._last_overview_bounds = bounds
        self._last_road_segments = segments
        return overview_path, street_path

    def _vlm_images(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        phase: str,
        attempt: int = 1,
    ) -> str:
        raw = self.client.complete_images(prompt, image_paths)
        self._record_vlm(phase, raw, attempt=attempt)
        return raw

    def _ensure_navigation_plan(self, world: World, state: AgentState) -> None:
        track = state.pole_in_consideration
        if not track or track in state.classified:
            track = select_target_pole(world, state)
            if track:
                state.pole_in_consideration = track
            else:
                self._goal_pano_id = None
                return

        goal, _ = plan_mission_to_pole(world, state, track)
        self._goal_pano_id = goal

    def _navigate(
        self,
        world: World,
        state: AgentState,
        pole_in_clear_view: bool,
    ) -> Action | None:
        overview_path, street_path = self._render_navigation_images(world, state)
        self.last_phase = "streets_navigation"

        neighbors = get_neighbors(world.neighbor_map, state.pano_id)
        legal = navigation_allowed_actions(world, state)
        prompt = build_map_navigation_prompt(
            world,
            state,
            pole_in_clear_view=pole_in_clear_view,
            allowed_actions=legal,
            map_bounds=self._last_overview_bounds,
        )

        image_paths = [overview_path, street_path]

        last_error = ""
        for attempt in range(self.parse_retries + 1):
            extra = ""
            if attempt > 0:
                extra = (
                    f"\n\nPrevious reply invalid ({last_error}). JSON only. "
                    f"Allowed: {', '.join(legal)}. "
                    "move needs road_point_x/y (0–899) ON a yellow road line."
                )
            self.last_prompt = prompt + extra
            raw = self._vlm_images(
                self.last_prompt,
                image_paths,
                phase=self.last_phase,
                attempt=attempt + 1,
            )
            action = parse_navigation_response(
                raw,
                allowed=legal,
                neighbor_ids=neighbors,
            )
            if action is not None:
                resolved = self._resolve_navigation_move(
                    world, state, action, neighbors
                )
                if resolved is not None:
                    return resolved
            last_error = "could not parse navigation JSON or resolve road point"

        return None

    def _resolve_navigation_move(
        self,
        world: World,
        state: AgentState,
        action: Action,
        neighbor_ids: list[str],
    ) -> Action | None:
        if action.type != ActionType.MOVE:
            return action
        if action.target_pano_id:
            if neighbor_ids and action.target_pano_id not in neighbor_ids:
                return None
            return action
        if (
            not action.map_point_px
            or not self._last_overview_bounds
            or not self._last_road_segments
        ):
            return None
        mx, my = action.map_point_px
        target = closest_neighbor_to_road_pick(
            world,
            state,
            neighbor_ids,
            mx,
            my,
            self._last_overview_bounds,
            self._last_road_segments,
        )
        if not target:
            return None
        return Action(type=ActionType.MOVE, target_pano_id=target)

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
            raw = self._vlm_images(
                self.last_prompt,
                [map_path, street_path],
                phase="dual_pole_type_classification",
                attempt=attempt + 1,
            )
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
        for label, image in (
            ("map_streets", self.last_map_image),
            ("street", self.last_street_image),
        ):
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
        note="graph navigation hint (classify uses VLM visible_pole_id)",
    )
    return next_state

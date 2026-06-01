"""
Policy interface and StubPolicy (graph BFS + scan, no VLM).

For VLM behavior see vlm_policy.py. apply_consideration() syncs pole_in_consideration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.directions import bin_center_world_yaw, clamp_bin
from agent.environment import World
from agent.geo import angle_diff_deg, bearing_deg, distance_m
from agent.graph import get_neighbors
from agent.pathfinding import closest_pano_to_pole, panos_near_pole, shortest_path
from agent.types import (
    DIRECTION_BIN_COUNT,
    Pano,
    Pole,
    VIEW_SHED_RADIUS_M,
    Action,
    ActionType,
    AgentState,
    PoleGuess,
    PoleType,
)


class Policy(ABC):
    """choose() is called after observe/clear-view and apply_consideration in AgentLoop."""

    @abstractmethod
    def choose(self, world: World, state: AgentState, pole_in_clear_view: bool) -> Action:
        raise NotImplementedError


class StubPolicy(Policy):
    """
    Planner stub: route along the 20 m pano graph to a viewpoint near each pole,
    scan, classify, then advance. No VLM.
    """

    CLASSIFY_MAX_DISTANCE_M = 35
    STUCK_STEP_LIMIT = 24
    VIEW_POLE_MAX_M = 20

    def __init__(self, placeholder_type: PoleType = "lamp_post") -> None:
        self.placeholder_type = placeholder_type
        self.target_track_id: str | None = None
        self.target_pano_id: str | None = None
        self.path: list[str] = []
        self.phase: str = "navigate"
        self.scan_bin: int = 0
        self.stuck_steps: int = 0
        self._last_signature: tuple[str, int, str, int] | None = None
        self._tried_views: set[tuple[str, str]] = set()

    def reset(self) -> None:
        self.target_track_id = None
        self.target_pano_id = None
        self.path = []
        self.phase = "navigate"
        self.scan_bin = 0
        self.stuck_steps = 0
        self._last_signature = None
        self._tried_views.clear()

    def choose(self, world: World, state: AgentState, pole_in_clear_view: bool) -> Action:
        if world.is_task_complete(state):
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        self._track_stuck(state)

        if self.target_track_id is None or self.target_track_id in state.classified:
            self._start_mission(world, state)
        if self.target_track_id is None:
            return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        if self.stuck_steps >= self.STUCK_STEP_LIMIT:
            self._abandon_mission(world, state)
            self._start_mission(world, state)
            if self.target_track_id is None:
                return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)

        pole = world.poles_by_track[self.target_track_id]

        if pole_in_clear_view:
            would_complete = len(state.classified) + 1 >= len(world.poles)
            return Action(
                type=ActionType.CLASSIFY_OR_STOP,
                pole_type=self.placeholder_type,
                stop_after=would_complete,
            )

        if self.phase == "navigate":
            return self._navigate(world, state)
        return self._scan(world, state, pole, world.poles_in_view(state))

    def _navigate(self, world: World, state: AgentState) -> Action:
        if self.target_pano_id is None:
            self.phase = "scan"
            return self._scan(
                world,
                state,
                world.poles_by_track[self.target_track_id],  # type: ignore[arg-type]
                world.poles_in_view(state),
            )

        if state.pano_id == self.target_pano_id:
            self.path = []
            self.phase = "scan"
            self.scan_bin = 0
            return self._scan(
                world,
                state,
                world.poles_by_track[self.target_track_id],  # type: ignore[arg-type]
                world.poles_in_view(state),
            )

        if not self.path:
            self._refresh_path(world, state)
        if not self.path:
            self._abandon_mission(world, state)
            self._start_mission(world, state)
            if self.target_track_id is None:
                return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)
            return self._navigate(world, state)

        next_id = self.path[0]
        if next_id not in get_neighbors(world.neighbor_map, state.pano_id):
            self._refresh_path(world, state)
            if not self.path:
                self.phase = "scan"
                return self._scan(
                    world,
                    state,
                    world.poles_by_track[self.target_track_id],  # type: ignore[arg-type]
                    world.poles_in_view(state),
                )
            next_id = self.path[0]

        return self._turn_and_move_toward(world, state, next_id)

    def _scan(
        self,
        world: World,
        state: AgentState,
        pole: Pole,
        poles_in_view,
    ) -> Action:
        pano = world.panos_by_id[state.pano_id]
        visible = next((p for p in poles_in_view if p.track_id == pole.track_id), None)
        if visible and visible.distance_m <= self.CLASSIFY_MAX_DISTANCE_M:
            would_complete = len(state.classified) + 1 >= len(world.poles)
            return Action(
                type=ActionType.CLASSIFY_OR_STOP,
                pole_type=self.placeholder_type,
                stop_after=would_complete,
            )

        target_bearing = bearing_deg(pano, pole.lat, pole.lon)
        view_yaw = world.view_yaw_deg(state)
        bearing_error = angle_diff_deg(view_yaw, target_bearing)

        if bearing_error <= 20:
            aligned_neighbors = world.neighbor_panos_for_move(state)
            if aligned_neighbors and state.pano_id != self.target_pano_id:
                return Action(type=ActionType.MOVE)

        if bearing_error > 15:
            return self._turn_toward(state.direction_bin, target_bearing, pano.heading_deg)

        if state.direction_bin != self.scan_bin:
            delta = (self.scan_bin - state.direction_bin) % DIRECTION_BIN_COUNT
            if delta <= DIRECTION_BIN_COUNT // 2:
                return Action(type=ActionType.TURN_RIGHT)
            return Action(type=ActionType.TURN_LEFT)

        self.scan_bin = clamp_bin(self.scan_bin + 1)
        if self.scan_bin == 0:
            self._abandon_mission(world, state)
            self._start_mission(world, state)
            if self.target_track_id is None:
                return Action(type=ActionType.CLASSIFY_OR_STOP, stop_after=True)
            return self._navigate(world, state)

        return Action(type=ActionType.TURN_RIGHT)

    def _start_mission(self, world: World, state: AgentState) -> None:
        best: tuple[int, float, str, str, list[str]] | None = None

        for pole in world.poles:
            if pole.track_id in state.classified:
                continue

            candidates = panos_near_pole(world.panos, pole, self.VIEW_POLE_MAX_M)
            if not candidates:
                view_pano, view_dist = closest_pano_to_pole(world.panos, pole)
                candidates = [(view_pano, view_dist)]

            for view_pano, view_dist in candidates:
                key = (pole.track_id, view_pano.id)
                if key in self._tried_views:
                    continue
                path = shortest_path(world.neighbor_map, state.pano_id, view_pano.id)
                if path is None:
                    continue
                hop_cost = len(path) - 1
                score = (hop_cost, view_dist)
                if best is None or score < (best[0], best[1]):
                    best = (hop_cost, view_dist, pole.track_id, view_pano.id, path)

        if best is None:
            self._tried_views.clear()
            for pole in world.poles:
                if pole.track_id in state.classified:
                    continue
                view_pano, view_dist = closest_pano_to_pole(world.panos, pole)
                path = shortest_path(world.neighbor_map, state.pano_id, view_pano.id)
                if path is None:
                    continue
                hop_cost = len(path) - 1
                score = (hop_cost, view_dist)
                if best is None or score < (best[0], best[1]):
                    best = (hop_cost, view_dist, pole.track_id, view_pano.id, path)

        if best is None:
            self.target_track_id = None
            self.target_pano_id = None
            self.path = []
            return

        _, _, track_id, pano_id, path = best
        self.target_track_id = track_id
        self.target_pano_id = pano_id
        self.path = path[1:]
        self.phase = "navigate"
        self.scan_bin = 0
        self.stuck_steps = 0
        self._last_signature = None

    def _refresh_path(self, world: World, state: AgentState) -> None:
        if self.target_pano_id is None:
            self.path = []
            return
        path = shortest_path(world.neighbor_map, state.pano_id, self.target_pano_id)
        self.path = path[1:] if path else []

    def _abandon_mission(self, world: World, state: AgentState) -> None:
        if self.target_track_id and self.target_pano_id:
            self._tried_views.add((self.target_track_id, self.target_pano_id))
        self.target_track_id = None
        self.target_pano_id = None
        self.path = []
        self.phase = "navigate"
        self.scan_bin = 0
        self.stuck_steps = 0
        self._last_signature = None

    def _track_stuck(self, state: AgentState) -> None:
        signature = (state.pano_id, state.direction_bin, self.phase, len(self.path))
        if signature == self._last_signature:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            self._last_signature = signature

    def _turn_and_move_toward(self, world: World, state: AgentState, target_pano_id: str) -> Action:
        pano = world.panos_by_id[state.pano_id]
        target = world.panos_by_id[target_pano_id]
        bearing = bearing_deg(pano, target.lat, target.lon)
        view_yaw = world.view_yaw_deg(state)
        error = angle_diff_deg(view_yaw, bearing)

        aligned = [n for n in world.neighbor_panos_for_move(state) if n.id == target_pano_id]
        if aligned:
            self.path = self.path[1:] if self.path and self.path[0] == target_pano_id else self.path
            return Action(type=ActionType.MOVE)

        return self._turn_toward(state.direction_bin, bearing, pano.heading_deg)

    def _turn_toward(self, direction_bin: int, target_bearing: float, pano_heading: float) -> Action:
        probe = Pano(
            id="_",
            image_path="_",
            session_id="_",
            order_in_session=0,
            lat=0.0,
            lon=0.0,
            heading_deg=pano_heading,
            width=0,
            height=0,
        )
        current_yaw = bin_center_world_yaw(probe, direction_bin)
        error = ((target_bearing - current_yaw + 540) % 360) - 180
        if error > 0:
            return Action(type=ActionType.TURN_RIGHT)
        return Action(type=ActionType.TURN_LEFT)

    def target_track_for_state(self, state: AgentState) -> str | None:
        if self.target_track_id and self.target_track_id not in state.classified:
            return self.target_track_id
        return None


def apply_consideration(
    state: AgentState,
    world: World,
    policy: Policy,
) -> AgentState:
    """Sync pole_in_consideration / pole_guess from the active policy."""
    from agent.vlm_policy import VlmPolicy, apply_vlm_consideration

    if isinstance(policy, VlmPolicy):
        return apply_vlm_consideration(state, world, fallback_type=policy.fallback_type)

    next_state = state.copy()
    track_id: str | None = None

    if isinstance(policy, StubPolicy):
        track_id = policy.target_track_for_state(state)
    elif next_state.pole_in_consideration and next_state.pole_in_consideration not in next_state.classified:
        track_id = next_state.pole_in_consideration

    if not track_id:
        return next_state

    pole = world.poles_by_track[track_id]
    next_state.pole_in_consideration = track_id
    placeholder = getattr(policy, "placeholder_type", None)
    next_state.pole_guess = PoleGuess(
        track_id=pole.track_id,
        pole_id=pole.pole_id,
        pole_type=placeholder,
        note="planner stub hypothesis",
    )
    return next_state

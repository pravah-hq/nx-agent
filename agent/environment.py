"""
Simulation world: panos, poles, 20 m graph, viewshed, and action physics.

World.apply_action implements turn / move / classify — policies only propose Actions.
Tweak poles_in_view() here to change geometric visibility (HFOV, VIEW_SHED_RADIUS_M).
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.directions import bin_center_world_yaw, clamp_bin
from agent.geo import angle_diff_deg, bearing_deg, distance_m
from agent.graph import build_neighbor_map, get_neighbors
from agent.move_history import move_bearing_between_panos
from agent.types import (
    DEFAULT_HFOV_DEG,
    DIRECTION_BIN_WIDTH_DEG,
    STEP_ALIGNMENT_MAX_DEG,
    VIEW_SHED_RADIUS_M,
    Action,
    ActionType,
    AgentState,
    Pano,
    Pole,
    PoleGuess,
    PoleInView,
    PoleType,
)


@dataclass
class World:
    """Loaded dataset + neighbor graph; use World.load() from CLI."""

    panos: list[Pano]
    panos_by_id: dict[str, Pano]
    poles: list[Pole]
    poles_by_track: dict[str, Pole]
    neighbor_map: dict[str, list[str]]
    target_pole_ids: frozenset[str]

    @classmethod
    def load(cls, metadata_dir=None) -> World:
        from agent.data_loader import load_panos, load_poles

        panos = load_panos(metadata_dir)
        poles = load_poles(metadata_dir)
        return cls(
            panos=panos,
            panos_by_id={p.id: p for p in panos},
            poles=poles,
            poles_by_track={p.track_id: p for p in poles},
            neighbor_map=build_neighbor_map(panos),
            target_pole_ids=frozenset(p.track_id for p in poles),
        )

    def initial_state(self, start_pano_id: str | None = None) -> AgentState:
        pano_id = start_pano_id or self.panos[0].id
        if pano_id not in self.panos_by_id:
            raise ValueError(f"Unknown pano id: {pano_id}")
        return AgentState(pano_id=pano_id, direction_bin=0)

    def view_yaw_deg(self, state: AgentState) -> float:
        pano = self.panos_by_id[state.pano_id]
        return bin_center_world_yaw(pano, state.direction_bin)

    def poles_in_view(self, state: AgentState) -> list[PoleInView]:
        """Poles within VIEW_SHED_RADIUS_M and DEFAULT_HFOV_DEG cone, sorted by angle."""
        pano = self.panos_by_id[state.pano_id]
        view_yaw = self.view_yaw_deg(state)
        half_fov = DEFAULT_HFOV_DEG / 2
        visible: list[PoleInView] = []

        for pole in self.poles:
            dist = distance_m(pano.lat, pano.lon, pole.lat, pole.lon)
            if dist > VIEW_SHED_RADIUS_M:
                continue
            bearing = bearing_deg(pano, pole.lat, pole.lon)
            angle_from_view = angle_diff_deg(view_yaw, bearing)
            if angle_from_view > half_fov:
                continue
            visible.append(
                PoleInView(
                    track_id=pole.track_id,
                    pole_id=pole.pole_id,
                    bearing_deg=bearing,
                    distance_m=dist,
                    angle_from_view_deg=angle_from_view,
                    in_center=angle_from_view <= DIRECTION_BIN_WIDTH_DEG / 2,
                )
            )

        visible.sort(key=lambda item: item.angle_from_view_deg)
        return visible

    def neighbor_panos_for_move(self, state: AgentState) -> list[Pano]:
        pano = self.panos_by_id[state.pano_id]
        view_yaw = self.view_yaw_deg(state)
        candidates: list[tuple[float, float, Pano]] = []

        for neighbor_id in get_neighbors(self.neighbor_map, state.pano_id):
            neighbor = self.panos_by_id[neighbor_id]
            bearing = bearing_deg(pano, neighbor.lat, neighbor.lon)
            angle = angle_diff_deg(view_yaw, bearing)
            if angle <= STEP_ALIGNMENT_MAX_DEG:
                candidates.append((angle, distance_m(pano.lat, pano.lon, neighbor.lat, neighbor.lon), neighbor))

        candidates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in candidates]

    def apply_action(self, state: AgentState, action: Action) -> tuple[AgentState, str]:
        """Apply one action; returns (new_state, human message). Move requires neighbor or alignment."""
        next_state = state.copy()

        if action.type == ActionType.TURN_LEFT:
            next_state.direction_bin = clamp_bin(state.direction_bin - 1)
            return next_state, "Turned left."

        if action.type == ActionType.TURN_RIGHT:
            next_state.direction_bin = clamp_bin(state.direction_bin + 1)
            return next_state, "Turned right."

        if action.type == ActionType.MOVE:
            neighbor_ids = get_neighbors(self.neighbor_map, state.pano_id)
            if action.target_pano_id:
                if action.target_pano_id not in neighbor_ids:
                    return state, "Move rejected: target is not a neighbor within 20 m."
                from_pano = self.panos_by_id[state.pano_id]
                to_pano = self.panos_by_id[action.target_pano_id]

                next_state.pano_id = action.target_pano_id
                next_state.last_move_from_pano_id = state.pano_id
                next_state.last_move_bearing_deg = move_bearing_between_panos(
                    from_pano, to_pano
                )
                return next_state, f"Moved to {action.target_pano_id}."
            candidates = self.neighbor_panos_for_move(state)
            if not candidates:
                return state, "Move rejected: no neighbor within view alignment."
            target = candidates[0]
            from_pano = self.panos_by_id[state.pano_id]

            next_state.pano_id = target.id
            next_state.last_move_from_pano_id = state.pano_id
            next_state.last_move_bearing_deg = move_bearing_between_panos(
                from_pano, target
            )
            return next_state, f"Moved to {target.id}."

        if action.type == ActionType.CLASSIFY_OR_STOP:
            if state.pole_in_consideration is None:
                return state, "Classify rejected: no pole in consideration."

            track_id = state.pole_in_consideration
            pole = self.poles_by_track[track_id]
            pole_type = action.pole_type

            if pole_type is None:
                if state.pole_guess and state.pole_guess.pole_type:
                    pole_type = state.pole_guess.pole_type
                else:
                    return state, "Classify rejected: provide pole_type (VLM/stub not set)."

            next_state.classified[track_id] = pole_type
            next_state.pole_guess = PoleGuess(
                track_id=track_id,
                pole_id=pole.pole_id,
                pole_type=pole_type,
                note="classified",
            )
            msg = f"Classified {pole.pole_id} as {pole_type}."
            if action.stop_after:
                return next_state, msg + " Stopping."
            next_state.pole_in_consideration = None
            next_state.pole_guess = None
            return next_state, msg + " Continuing."

        return state, "Unknown action."

    def remaining_pole_ids(self, state: AgentState) -> list[str]:
        return [p.pole_id for p in self.poles if p.track_id not in state.classified]

    def is_task_complete(self, state: AgentState) -> bool:
        return self.target_pole_ids.issubset(state.classified.keys())

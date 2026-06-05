"""Serialize agent step history for the web UI trace viewer."""

from __future__ import annotations

from typing import Any

from agent.environment import World
from agent.types import AgentState, StepRecord


def _state_dict(state: AgentState) -> dict[str, Any]:
    return {
        "pano_id": state.pano_id,
        "direction_bin": state.direction_bin,
        "pole_in_consideration": state.pole_in_consideration,
        "classified": dict(state.classified),
        "pole_guess": None
        if state.pole_guess is None
        else {
            "track_id": state.pole_guess.track_id,
            "pole_id": state.pole_guess.pole_id,
            "pole_type": state.pole_guess.pole_type,
            "note": state.pole_guess.note,
        },
        "last_move_bearing_deg": state.last_move_bearing_deg,
        "last_move_from_pano_id": state.last_move_from_pano_id,
        "visited_pano_ids": sorted(state.visited_pano_ids),
    }


def step_record_to_dict(
    world: World,
    record: StepRecord,
    *,
    vlm_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    visible_pole_id = record.visible_pole_id
    if not visible_pole_id:
        track = record.state_before.pole_in_consideration
        if track and track in world.poles_by_track:
            visible_pole_id = world.poles_by_track[track].pole_id

    return {
        "step": record.step,
        "action": record.action.type.value,
        "target_pano_id": record.action.target_pano_id,
        "pole_type": record.action.pole_type,
        "stop_after": record.action.stop_after,
        "pole_in_clear_view": record.pole_in_clear_view,
        "message": record.message,
        "state_before": _state_dict(record.state_before),
        "state_after": _state_dict(record.state_after),
        "visible_pole_id": visible_pole_id,
        "vlm_calls": vlm_calls or record.vlm_calls,
    }


def trace_document(
    world: World,
    history: list[StepRecord],
    *,
    policy: str = "unknown",
) -> dict[str, Any]:
    return {
        "version": 1,
        "policy": policy,
        "pole_count": len(world.poles),
        "steps": [step_record_to_dict(world, record) for record in history],
    }

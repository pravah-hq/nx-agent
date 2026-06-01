"""
Move selection helpers: anti-backtrack and graph-planned neighbor fallback.

When VLM navigation JSON fails, VlmPolicy uses pick_planned_neighbor.
"""

from __future__ import annotations

from agent.environment import World
from agent.geo import distance_m
from agent.targeting import next_path_hop, pano_compact_id, plan_mission_to_pole
from agent.types import AgentState


def backtrack_blocked_ids(last_pano_id: str | None) -> frozenset[str]:
    """Neighbor pano ids the agent must not move back to (immediate previous)."""
    if last_pano_id:
        return frozenset({last_pano_id})
    return frozenset()


def allowed_move_targets(neighbor_ids: list[str], blocked: frozenset[str]) -> list[str]:
    return [nid for nid in neighbor_ids if nid not in blocked]


def is_backtrack_move(target_pano_id: str | None, blocked: frozenset[str]) -> bool:
    return bool(target_pano_id and target_pano_id in blocked)


def pick_planned_neighbor(
    world: World,
    state: AgentState,
    neighbor_ids: list[str],
    blocked: frozenset[str],
    *,
    nav_path: list[str] | None = None,
) -> str | None:
    """
    1) Next hop on cached nav_path toward view pano
    2) Recompute path via plan_mission_to_pole
    3) Any neighbor closer to target pole
    4) First unblocked neighbor
    """
    candidates = allowed_move_targets(neighbor_ids, blocked)
    if not candidates:
        return None

    track_id = state.pole_in_consideration
    if track_id and track_id in world.poles_by_track:
        if nav_path:
            hop = next_path_hop(nav_path)
            if hop and hop in candidates:
                return hop

        _, path = plan_mission_to_pole(world, state, track_id)
        hop = next_path_hop(path)
        if hop and hop in candidates:
            return hop

        pole = world.poles_by_track[track_id]
        pano = world.panos_by_id[state.pano_id]
        current_dist = distance_m(pano.lat, pano.lon, pole.lat, pole.lon)
        best_id: str | None = None
        best_dist = float("inf")
        for nid in candidates:
            neighbor = world.panos_by_id[nid]
            dist = distance_m(neighbor.lat, neighbor.lon, pole.lat, pole.lon)
            if dist < best_dist:
                best_dist = dist
                best_id = nid
        if best_id and best_dist < current_dist:
            return best_id

    return candidates[0]


def build_neighbor_move_options(
    world: World,
    state: AgentState,
    neighbor_ids: list[str],
    blocked: frozenset[str],
    *,
    nav_path: list[str] | None = None,
) -> list[dict]:
    """Structured neighbor list embedded in VLM navigation prompts."""
    pole = (
        world.poles_by_track.get(state.pole_in_consideration)
        if state.pole_in_consideration
        else None
    )
    planned_hop = next_path_hop(nav_path) if nav_path else None
    options: list[dict] = []
    for nid in neighbor_ids:
        options.append(
            {
                "target_pano_id": nid,
                "label": pano_compact_id(nid),
                "blocked": nid in blocked,
                "distance_to_target_pole_m": round(distance_m(
                    world.panos_by_id[nid].lat,
                    world.panos_by_id[nid].lon,
                    pole.lat,
                    pole.lon,
                ), 1)
                if pole
                else None,
                "recommended_next_hop": nid == planned_hop,
            }
        )
    return options

from __future__ import annotations

from agent.environment import World
from agent.geo import distance_m
from agent.types import Action, ActionType, AgentState


def backtrack_blocked_ids(last_pano_id: str | None) -> frozenset[str]:
    if last_pano_id:
        return frozenset({last_pano_id})
    return frozenset()


def allowed_move_targets(neighbor_ids: list[str], blocked: frozenset[str]) -> list[str]:
    return [nid for nid in neighbor_ids if nid not in blocked]


def is_backtrack_move(target_pano_id: str | None, blocked: frozenset[str]) -> bool:
    return bool(target_pano_id and target_pano_id in blocked)


def pick_forward_neighbor(
    world: World,
    state: AgentState,
    neighbor_ids: list[str],
    blocked: frozenset[str],
) -> str | None:
    """Pick a neighbor that is not blocked; prefer lower distance to target pole."""
    candidates = allowed_move_targets(neighbor_ids, blocked)
    if not candidates:
        return None

    track_id = state.pole_in_consideration
    if not track_id or track_id not in world.poles_by_track:
        return candidates[0]

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

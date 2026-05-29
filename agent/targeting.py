from __future__ import annotations

from agent.environment import World
from agent.geo import distance_m
from agent.graph import get_neighbors
from agent.pathfinding import closest_pano_to_pole, panos_near_pole, shortest_path
from agent.types import VIEW_SHED_RADIUS_M, AgentState, Pole


def pano_compact_id(pano_id: str) -> str:
    return pano_id.split("/")[-1].replace(".jpg", "")


def distance_to_pole_m(world: World, pano_id: str, pole: Pole) -> float:
    pano = world.panos_by_id[pano_id]
    return distance_m(pano.lat, pano.lon, pole.lat, pole.lon)


def best_view_pano_for_pole(world: World, pole: Pole) -> tuple[str, float]:
    """Pano id to stand at for viewing, and straight-line distance to pole."""
    nearby = panos_near_pole(world.panos, pole, VIEW_SHED_RADIUS_M)
    if nearby:
        pano, dist = nearby[0]
        return pano.id, dist
    pano, dist = closest_pano_to_pole(world.panos, pole)
    return pano.id, dist


def path_to_pano(
    world: World,
    from_pano_id: str,
    goal_pano_id: str,
) -> list[str] | None:
    return shortest_path(world.neighbor_map, from_pano_id, goal_pano_id)


def select_target_pole(world: World, state: AgentState) -> str | None:
    """Pick unclassified pole minimizing pano-graph hops to a good view pano."""
    best: tuple[int, float, str] | None = None

    for pole in world.poles:
        if pole.track_id in state.classified:
            continue
        view_pano_id, view_dist = best_view_pano_for_pole(world, pole)
        path = path_to_pano(world, state.pano_id, view_pano_id)
        if path is None:
            continue
        hop_cost = len(path) - 1
        score = (hop_cost, view_dist)
        if best is None or score < (best[0], best[1]):
            best = (hop_cost, view_dist, pole.track_id)

    return best[2] if best else None


def plan_mission_to_pole(
    world: World,
    state: AgentState,
    pole_track_id: str,
) -> tuple[str | None, list[str]]:
    """
    Returns (goal_view_pano_id, full_path from current pano including current).
    """
    pole = world.poles_by_track[pole_track_id]
    view_pano_id, _ = best_view_pano_for_pole(world, pole)
    path = path_to_pano(world, state.pano_id, view_pano_id)
    if path is None:
        return None, []
    return view_pano_id, path


def next_path_hop(path_from_current: list[str]) -> str | None:
    """First pano to move to (not the current node). path includes current at [0]."""
    if len(path_from_current) < 2:
        return None
    return path_from_current[1]

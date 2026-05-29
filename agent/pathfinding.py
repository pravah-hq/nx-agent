from __future__ import annotations

from collections import deque

from agent.geo import distance_m
from agent.types import Pano, Pole


def shortest_path(
    neighbor_map: dict[str, list[str]],
    start_id: str,
    goal_id: str,
) -> list[str] | None:
    if start_id == goal_id:
        return [start_id]
    if start_id not in neighbor_map or goal_id not in neighbor_map:
        return None

    queue: deque[str] = deque([start_id])
    parent: dict[str, str | None] = {start_id: None}

    while queue:
        current = queue.popleft()
        if current == goal_id:
            path: list[str] = []
            node: str | None = goal_id
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for neighbor_id in neighbor_map.get(current, []):
            if neighbor_id in parent:
                continue
            parent[neighbor_id] = current
            queue.append(neighbor_id)

    return None


def closest_pano_to_pole(panos: list[Pano], pole: Pole) -> tuple[Pano, float]:
    best_pano = panos[0]
    best_dist = distance_m(best_pano.lat, best_pano.lon, pole.lat, pole.lon)
    for pano in panos[1:]:
        dist = distance_m(pano.lat, pano.lon, pole.lat, pole.lon)
        if dist < best_dist:
            best_pano = pano
            best_dist = dist
    return best_pano, best_dist


def panos_near_pole(panos: list[Pano], pole: Pole, max_m: float) -> list[tuple[Pano, float]]:
    nearby: list[tuple[Pano, float]] = []
    for pano in panos:
        dist = distance_m(pano.lat, pano.lon, pole.lat, pole.lon)
        if dist <= max_m:
            nearby.append((pano, dist))
    nearby.sort(key=lambda item: item[1])
    return nearby

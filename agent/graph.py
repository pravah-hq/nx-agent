from __future__ import annotations

from agent.geo import distance_between_panos
from agent.types import PANO_PROXIMITY_MAX_M, Pano


def build_neighbor_map(panos: list[Pano]) -> dict[str, list[str]]:
    by_id = {pano.id: pano for pano in panos}
    neighbors: dict[str, list[str]] = {pano.id: [] for pano in panos}

    for i, a in enumerate(panos):
        for b in panos[i + 1 :]:
            if distance_between_panos(a, b) <= PANO_PROXIMITY_MAX_M:
                neighbors[a.id].append(b.id)
                neighbors[b.id].append(a.id)

    for pano_id in neighbors:
        neighbors[pano_id].sort()

    return neighbors


def get_neighbors(neighbor_map: dict[str, list[str]], pano_id: str) -> list[str]:
    return list(neighbor_map.get(pano_id, []))

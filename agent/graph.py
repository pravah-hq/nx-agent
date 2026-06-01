"""
Pano graph: undirected edges when two panos are within PANO_PROXIMITY_MAX_M.

This is the same rule as the Leaflet UI — move only along these edges.
"""

from __future__ import annotations

from agent.geo import distance_between_panos
from agent.types import PANO_PROXIMITY_MAX_M, Pano


def build_neighbor_map(panos: list[Pano]) -> dict[str, list[str]]:
    """For each pano id, sorted list of neighbor pano ids within 20 m."""
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

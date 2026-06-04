"""
Pano-graph road geometry for VLM navigation (move picks snap to these segments).
"""

from __future__ import annotations

import math

from agent.environment import World
from agent.geo import distance_m
from agent.graph import get_neighbors
from agent.map_render import OverviewMapBounds, lat_lon_to_map_pixel, map_pixel_to_lat_lon
from agent.types import AgentState, PANO_PROXIMITY_MAX_M

# How far along the graph to draw / accept road picks (meters from current pano).
NAV_ROAD_DRAW_RADIUS_M = 50
# After snap, pick must be within this distance of a road segment.
MAX_SNAP_DISTANCE_M = 12


def _norm_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _visible_panos_near_agent(
    world: World,
    state: AgentState,
    *,
    max_range_m: float,
) -> set[str]:
    pano = world.panos_by_id[state.pano_id]
    visible: set[str] = {state.pano_id}
    for candidate in world.panos:
        if (
            distance_m(pano.lat, pano.lon, candidate.lat, candidate.lon)
            <= max_range_m
        ):
            visible.add(candidate.id)
    return visible


def _graph_edges(visible: set[str], neighbor_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for pano_id in visible:
        for nid in neighbor_map.get(pano_id, []):
            if nid in visible and pano_id < nid:
                edges.append((pano_id, nid))
    return edges


def extract_road_polylines(
    world: World,
    visible_panos: set[str],
) -> list[list[str]]:
    """Chain graph edges into polylines (pano id chains)."""
    neighbor_map = world.neighbor_map
    adj: dict[str, list[str]] = {p: [] for p in visible_panos}
    for a in visible_panos:
        for b in neighbor_map.get(a, []):
            if b in visible_panos and b not in adj[a]:
                adj[a].append(b)
                adj[b].append(a)

    used: set[tuple[str, str]] = set()
    polylines: list[list[str]] = []

    def extend_from_end(chain: list[str], *, backward: bool) -> None:
        while len(adj[chain[0 if backward else -1]]) == 2:
            tip = chain[0 if backward else -1]
            other = chain[1 if backward else -2]
            nxt = adj[tip][0] if adj[tip][1] == other else adj[tip][1]
            e = _norm_edge(tip, nxt)
            if e in used:
                break
            used.add(e)
            if backward:
                chain.insert(0, nxt)
            else:
                chain.append(nxt)

    for a, b in _graph_edges(visible_panos, neighbor_map):
        e = _norm_edge(a, b)
        if e in used:
            continue
        used.add(e)
        chain = [a, b]
        extend_from_end(chain, backward=True)
        extend_from_end(chain, backward=False)
        if len(chain) >= 2:
            polylines.append(chain)

    return polylines


def local_road_polylines_at_agent(
    world: World,
    state: AgentState,
    *,
    max_range_m: float = NAV_ROAD_DRAW_RADIUS_M,
) -> list[list[str]]:
    """Polylines on the pano graph that pass through the current pano."""
    visible = _visible_panos_near_agent(world, state, max_range_m=max_range_m)
    chains = extract_road_polylines(world, visible)
    current = state.pano_id
    return [chain for chain in chains if current in chain]


def polylines_to_latlon(
    world: World,
    chains: list[list[str]],
) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for chain in chains:
        pts: list[tuple[float, float]] = []
        for pid in chain:
            p = world.panos_by_id.get(pid)
            if p:
                pts.append((p.lat, p.lon))
        if len(pts) >= 2:
            out.append(pts)
    return out


def latlon_polylines_to_segments(
    chains: list[list[tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for chain in chains:
        for i in range(len(chain) - 1):
            segments.append((chain[i], chain[i + 1]))
    return segments


def _closest_point_on_segment(
    lat: float,
    lon: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float, float]:
    """Return (snapped_lat, snapped_lon, distance_m) to segment a-b."""
    a_lat, a_lon = a
    b_lat, b_lon = b
    dx = b_lon - a_lon
    dy = b_lat - a_lat
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-18:
        d = distance_m(lat, lon, a_lat, a_lon)
        return a_lat, a_lon, d
    t = ((lon - a_lon) * dx + (lat - a_lat) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    s_lat = a_lat + t * dy
    s_lon = a_lon + t * dx
    return s_lat, s_lon, distance_m(lat, lon, s_lat, s_lon)


def snap_lat_lon_to_roads(
    lat: float,
    lon: float,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float] | None:
    if not segments:
        return None
    best_lat, best_lon = lat, lon
    best_dist = float("inf")
    for a, b in segments:
        s_lat, s_lon, dist = _closest_point_on_segment(lat, lon, a, b)
        if dist < best_dist:
            best_dist = dist
            best_lat, best_lon = s_lat, s_lon
    if best_dist > MAX_SNAP_DISTANCE_M:
        return None
    return best_lat, best_lon


def navigation_bounds_for_roads(
    world: World,
    state: AgentState,
    chains_latlon: list[list[tuple[float, float]]],
) -> OverviewMapBounds:
    """Map frame covering local roads, agent, and target pole."""
    from agent.map_render import square_bounds_centered

    pano = world.panos_by_id[state.pano_id]
    lats = [pano.lat]
    lons = [pano.lon]
    for chain in chains_latlon:
        for lat, lon in chain:
            lats.append(lat)
            lons.append(lon)
    track = state.pole_in_consideration
    if track and track in world.poles_by_track:
        pole = world.poles_by_track[track]
        lats.append(pole.lat)
        lons.append(pole.lon)
    half_lat = max(abs(lat - pano.lat) for lat in lats) + 0.00005
    half_lon = max(abs(lon - pano.lon) for lon in lons) + 0.00005
    half_m = max(
        half_lat * 111_320.0,
        half_lon * 111_320.0 * max(math.cos(math.radians(pano.lat)), 1e-6),
        28.0,
    )
    return square_bounds_centered(pano.lat, pano.lon, half_m)


def closest_neighbor_to_road_pick(
    world: World,
    state: AgentState,
    neighbor_ids: list[str],
    map_x: int,
    map_y: int,
    bounds: OverviewMapBounds,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> str | None:
    """
    VLM pick on the map → snap to local road segment → nearest legal neighbor pano.
    """
    if not neighbor_ids or not segments:
        return None
    lat, lon = map_pixel_to_lat_lon(map_x, map_y, bounds)
    snapped = snap_lat_lon_to_roads(lat, lon, segments)
    if snapped is None:
        return None
    s_lat, s_lon = snapped
    best_id: str | None = None
    best_dist = float("inf")
    for nid in neighbor_ids:
        n = world.panos_by_id.get(nid)
        if not n:
            continue
        dist = distance_m(s_lat, s_lon, n.lat, n.lon)
        if dist < best_dist:
            best_dist = dist
            best_id = nid
    if best_id and best_dist > PANO_PROXIMITY_MAX_M + 2.0:
        return None
    return best_id


def road_neighbor_hints(
    world: World,
    state: AgentState,
    neighbor_ids: list[str],
    bounds: OverviewMapBounds,
) -> list[dict]:
    """Bearing/distance for each neighbor along roads (prompt context)."""
    from agent.directions import bin_center_world_yaw
    from agent.geo import bearing_deg

    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    hints: list[dict] = []
    for nid in neighbor_ids:
        n = world.panos_by_id.get(nid)
        if not n:
            continue
        brg = bearing_deg(pano, n.lat, n.lon)
        dist = distance_m(pano.lat, pano.lon, n.lat, n.lon)
        px, py = lat_lon_to_map_pixel(n.lat, n.lon, bounds)
        delta = ((brg - view_yaw + 540.0) % 360.0) - 180.0
        hints.append(
            {
                "neighbor_pano_id": nid,
                "distance_m": round(dist, 1),
                "bearing_deg": round(brg, 1),
                "bearing_vs_view_deg": round(delta, 1),
                "map_x": px,
                "map_y": py,
            }
        )
    return hints

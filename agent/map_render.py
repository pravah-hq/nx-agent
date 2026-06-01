"""
PNG local map for VLM image 1 (cached under .cache/agent_maps/).

Legend: YOU=blue+wedge, yellow ring=planned next hop, gray lines=20 m edges,
orange=target pole, green=other unclassified, gray=classified.
"""

from __future__ import annotations

import math
from pathlib import Path

from agent.data_loader import repo_root
from agent.directions import bin_center_world_yaw
from agent.environment import World
from agent.graph import get_neighbors
from agent.targeting import pano_compact_id
from agent.types import AgentState

MAP_SIZE = 900
PADDING_PX = 60


def _local_bounds(
    world: World,
    state: AgentState,
    *,
    goal_pano_id: str | None,
    nav_path: list[str],
) -> tuple[float, float, float, float]:
    pano = world.panos_by_id[state.pano_id]
    lats = [pano.lat]
    lons = [pano.lon]

    for nid in get_neighbors(world.neighbor_map, state.pano_id):
        n = world.panos_by_id[nid]
        lats.append(n.lat)
        lons.append(n.lon)

    for hop in nav_path[:4]:
        if hop in world.panos_by_id:
            h = world.panos_by_id[hop]
            lats.append(h.lat)
            lons.append(h.lon)

    if goal_pano_id and goal_pano_id in world.panos_by_id:
        g = world.panos_by_id[goal_pano_id]
        lats.append(g.lat)
        lons.append(g.lon)

    track = state.pole_in_consideration
    if track and track in world.poles_by_track:
        pole = world.poles_by_track[track]
        lats.append(pole.lat)
        lons.append(pole.lon)

    pad = 0.00008
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad


def _project(
    lat: float,
    lon: float,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> tuple[int, int]:
    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)
    x = PADDING_PX + int((lon - min_lon) / lon_span * (MAP_SIZE - 2 * PADDING_PX))
    y = PADDING_PX + int((1 - (lat - min_lat) / lat_span) * (MAP_SIZE - 2 * PADDING_PX))
    return x, y


def render_map_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
    nav_path: list[str] | None = None,
) -> Path:
    """Local zoom map for VLM: pano graph edges, poles, planned hop."""
    from PIL import Image, ImageDraw, ImageFont

    hops = nav_path or []
    planned_hop = hops[0] if hops else None
    min_lat, min_lon, max_lat, max_lon = _local_bounds(
        world, state, goal_pano_id=goal_pano_id, nav_path=hops
    )
    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    current_neighbors = set(get_neighbors(neighbor_map, state.pano_id))

    cache = cache_dir or repo_root() / ".cache" / "agent_maps"
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"map_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")

    visible_panos = {state.pano_id, *current_neighbors, *hops[:5]}
    if goal_pano_id:
        visible_panos.add(goal_pano_id)

    # Edges between visible panos (20 m links)
    for pano_id in visible_panos:
        a = world.panos_by_id.get(pano_id)
        if not a:
            continue
        ax, ay = _project(a.lat, a.lon, min_lat, min_lon, max_lat, max_lon)
        for nid in neighbor_map.get(pano_id, []):
            if nid not in visible_panos:
                continue
            b = world.panos_by_id.get(nid)
            if not b or pano_id > nid:
                continue
            bx, by = _project(b.lat, b.lon, min_lat, min_lon, max_lat, max_lon)
            is_from_current = pano_id == state.pano_id or nid == state.pano_id
            width = 3 if is_from_current else 1
            color = (148, 163, 184, 200) if is_from_current else (100, 116, 139, 140)
            draw.line((ax, ay, bx, by), fill=color, width=width)

    # Planned path highlight from current
    if planned_hop and planned_hop in world.panos_by_id:
        cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
        hop_pano = world.panos_by_id[planned_hop]
        hx, hy = _project(
            hop_pano.lat, hop_pano.lon, min_lat, min_lon, max_lat, max_lon
        )
        draw.line((cx, cy, hx, hy), fill=(250, 204, 21, 220), width=4)

    # Poles (only those near the local view)
    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id in state.classified:
            color = (100, 116, 139, 255)
            radius = 5
        elif pole.track_id == target_track:
            color = (249, 115, 22, 255)
            radius = 9
        else:
            color = (34, 197, 94, 255)
            radius = 6
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        draw.text((px + 8, py - 6), pole.pole_id.replace("POLE_", ""), fill=(226, 232, 240))

    # Panorama nodes
    for pano_id in visible_panos:
        p = world.panos_by_id[pano_id]
        px, py = _project(p.lat, p.lon, min_lat, min_lon, max_lat, max_lon)
        if pano_id == state.pano_id:
            continue
        is_neighbor = pano_id in current_neighbors
        is_goal = pano_id == goal_pano_id
        is_hop = pano_id == planned_hop
        fill = (224, 242, 254, 255) if is_neighbor else (71, 85, 105, 200)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=fill)
        label = pano_compact_id(pano_id)
        if is_neighbor:
            draw.text((px + 6, py + 6), label, fill=(186, 230, 253))
        if is_goal:
            draw.text((px - 8, py - 18), "GOAL", fill=(250, 204, 21))

    # Current pano + view wedge
    cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
    half_fov = 50
    wedge_len = 70
    points = [(cx, cy)]
    for offset in range(-half_fov, half_fov + 1, 10):
        angle = math.radians(view_yaw + offset - 90)
        wx = cx + int(math.cos(angle) * wedge_len)
        wy = cy + int(math.sin(angle) * wedge_len)
        points.append((wx, wy))
    draw.polygon(points, fill=(56, 189, 248, 70))
    draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(56, 189, 248, 255), outline=(255, 255, 255))
    draw.text((cx + 12, cy - 8), "YOU", fill=(255, 255, 255))

    if planned_hop and planned_hop in world.panos_by_id:
        hop_pano = world.panos_by_id[planned_hop]
        hx, hy = _project(
            hop_pano.lat, hop_pano.lon, min_lat, min_lon, max_lat, max_lon
        )
        draw.ellipse((hx - 12, hy - 12, hx + 12, hy + 12), outline=(250, 204, 21, 255), width=2)

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    legend = [
        "LOCAL map: YOU=blue, yellow ring=next hop, lines=20m moves",
        "move: use exact target_pano_id from neighbor_moves in JSON",
    ]
    y = 8
    for line in legend:
        draw.text((8, y), line, fill=(226, 232, 240), font=font)
        y += 14

    image.save(out_path, format="PNG")
    return out_path

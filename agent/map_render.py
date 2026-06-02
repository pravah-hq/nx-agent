"""
PNG local maps for VLM navigation (cached under .cache/agent_maps/).

Two map views for navigation:
  overview — target, nodes, connections across the local area
  node_zoom — tight view centered on YOU and immediate neighbors

Legend: YOU=blue+wedge, yellow ring=planned next hop, gray lines=20 m edges,
orange=target pole, green=other unclassified, gray=classified.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from agent.data_loader import repo_root
from agent.directions import bin_center_world_yaw
from agent.environment import World
from agent.graph import get_neighbors
from agent.targeting import pano_compact_id
from agent.types import AgentState

MAP_SIZE = 900
PADDING_PX = 60
OVERVIEW_PAD_DEG = 0.00008
NODE_ZOOM_PAD_DEG = 0.000028


def _overview_bounds(
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

    pad = OVERVIEW_PAD_DEG
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad


def _node_zoom_bounds(
    world: World,
    state: AgentState,
) -> tuple[float, float, float, float]:
    """Tight bounds around the current pano and its graph neighbors."""
    pano = world.panos_by_id[state.pano_id]
    lats = [pano.lat]
    lons = [pano.lon]

    for nid in get_neighbors(world.neighbor_map, state.pano_id):
        n = world.panos_by_id[nid]
        lats.append(n.lat)
        lons.append(n.lon)

    track = state.pole_in_consideration
    if track and track in world.poles_by_track:
        pole = world.poles_by_track[track]
        lats.append(pole.lat)
        lons.append(pole.lon)

    pad = NODE_ZOOM_PAD_DEG
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


def _render_map_core(
    world: World,
    state: AgentState,
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    goal_pano_id: str | None,
    nav_path: list[str] | None,
    map_variant: Literal["overview", "node_zoom"],
    out_path: Path,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    hops = nav_path or []
    planned_hop = hops[0] if hops else None
    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    current_neighbors = set(get_neighbors(neighbor_map, state.pano_id))

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")

    if map_variant == "node_zoom":
        visible_panos = {state.pano_id, *current_neighbors}
    else:
        visible_panos = {state.pano_id, *current_neighbors, *hops[:5]}
        if goal_pano_id:
            visible_panos.add(goal_pano_id)

    node_radius = 8 if map_variant == "node_zoom" else 5
    you_radius = 14 if map_variant == "node_zoom" else 10
    wedge_len = 110 if map_variant == "node_zoom" else 70
    neighbor_edge_width = 4 if map_variant == "node_zoom" else 3

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
            width = neighbor_edge_width if is_from_current else 1
            color = (148, 163, 184, 220) if is_from_current else (100, 116, 139, 140)
            draw.line((ax, ay, bx, by), fill=color, width=width)

    if planned_hop and planned_hop in world.panos_by_id:
        cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
        hop_pano = world.panos_by_id[planned_hop]
        hx, hy = _project(
            hop_pano.lat, hop_pano.lon, min_lat, min_lon, max_lat, max_lon
        )
        draw.line((cx, cy, hx, hy), fill=(250, 204, 21, 220), width=4)

    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id in state.classified:
            color = (100, 116, 139, 255)
            radius = 6 if map_variant == "node_zoom" else 5
        elif pole.track_id == target_track:
            color = (249, 115, 22, 255)
            radius = 12 if map_variant == "node_zoom" else 9
        else:
            color = (34, 197, 94, 255)
            radius = 8 if map_variant == "node_zoom" else 6
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        label = pole.pole_id.replace("POLE_", "")
        draw.text((px + 10, py - 8), label, fill=(226, 232, 240))

    for pano_id in visible_panos:
        p = world.panos_by_id[pano_id]
        px, py = _project(p.lat, p.lon, min_lat, min_lon, max_lat, max_lon)
        if pano_id == state.pano_id:
            continue
        is_neighbor = pano_id in current_neighbors
        is_goal = pano_id == goal_pano_id
        is_hop = pano_id == planned_hop
        fill = (224, 242, 254, 255) if is_neighbor else (71, 85, 105, 200)
        r = node_radius
        draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)
        label = pano_compact_id(pano_id)
        if is_neighbor:
            draw.text((px + 8, py + 8), label, fill=(186, 230, 253))
            if map_variant == "node_zoom":
                draw.text(
                    (px + 8, py + 22),
                    pano_id.split("/")[-1],
                    fill=(148, 163, 184),
                )
        if is_goal:
            draw.text((px - 8, py - 22), "GOAL", fill=(250, 204, 21))
        if is_hop and not is_neighbor:
            draw.ellipse((px - 12, py - 12, px + 12, py + 12), outline=(250, 204, 21, 255), width=2)

    cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
    half_fov = 50
    points = [(cx, cy)]
    for offset in range(-half_fov, half_fov + 1, 10):
        angle = math.radians(view_yaw + offset - 90)
        wx = cx + int(math.cos(angle) * wedge_len)
        wy = cy + int(math.sin(angle) * wedge_len)
        points.append((wx, wy))
    draw.polygon(points, fill=(56, 189, 248, 80))
    draw.ellipse(
        (cx - you_radius, cy - you_radius, cx + you_radius, cy + you_radius),
        fill=(56, 189, 248, 255),
        outline=(255, 255, 255),
    )
    draw.text((cx + you_radius + 4, cy - 10), "YOU", fill=(255, 255, 255))

    if planned_hop and planned_hop in world.panos_by_id and planned_hop in current_neighbors:
        hop_pano = world.panos_by_id[planned_hop]
        hx, hy = _project(
            hop_pano.lat, hop_pano.lon, min_lat, min_lon, max_lat, max_lon
        )
        draw.ellipse((hx - 14, hy - 14, hx + 14, hy + 14), outline=(250, 204, 21, 255), width=3)

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    if map_variant == "node_zoom":
        legend = [
            "NODE ZOOM: YOU + immediate neighbors (20 m moves)",
            "Use with overview map for direction; alone when near target pole",
            "move: copy exact target_pano_id from neighbor_moves JSON",
        ]
    else:
        legend = [
            "OVERVIEW MAP: target, nodes, connections across local area",
            "Use with node zoom map to pick the next move",
            "move: copy exact target_pano_id from neighbor_moves JSON",
        ]
    y = 8
    for line in legend:
        draw.text((8, y), line, fill=(226, 232, 240), font=font)
        y += 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return out_path


def _map_cache_dir(cache_dir: Path | None) -> Path:
    cache = cache_dir or repo_root() / ".cache" / "agent_maps"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def render_map_overview_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
    nav_path: list[str] | None = None,
) -> Path:
    """Wide local map: target pole, graph nodes, and connections."""
    hops = nav_path or []
    min_lat, min_lon, max_lat, max_lon = _overview_bounds(
        world, state, goal_pano_id=goal_pano_id, nav_path=hops
    )
    cache = _map_cache_dir(cache_dir)
    out_path = cache / f"map_overview_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"
    return _render_map_core(
        world,
        state,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        goal_pano_id=goal_pano_id,
        nav_path=hops,
        map_variant="overview",
        out_path=out_path,
    )


def render_map_node_zoom_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
    nav_path: list[str] | None = None,
) -> Path:
    """Tight map centered on the current pano and its immediate neighbors."""
    hops = nav_path or []
    min_lat, min_lon, max_lat, max_lon = _node_zoom_bounds(world, state)
    cache = _map_cache_dir(cache_dir)
    out_path = cache / f"map_zoom_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"
    return _render_map_core(
        world,
        state,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        goal_pano_id=goal_pano_id,
        nav_path=hops,
        map_variant="node_zoom",
        out_path=out_path,
    )


def render_map_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
    nav_path: list[str] | None = None,
) -> Path:
    """Overview map (used by clear-view / classification prompts)."""
    return render_map_overview_image(
        world,
        state,
        cache_dir=cache_dir,
        goal_pano_id=goal_pano_id,
        nav_path=nav_path,
    )

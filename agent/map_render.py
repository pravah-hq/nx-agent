from __future__ import annotations

import math
from pathlib import Path

from agent.data_loader import repo_root
from agent.directions import bin_center_world_yaw
from agent.environment import World
from agent.graph import get_neighbors
from agent.types import AgentState, PoleInView

MAP_SIZE = 900
PADDING_PX = 60


def _bounds(world: World) -> tuple[float, float, float, float]:
    lats = [p.lat for p in world.panos] + [p.lat for p in world.poles]
    lons = [p.lon for p in world.panos] + [p.lon for p in world.poles]
    pad = 0.00015
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


def _compact_id(pano_id: str) -> str:
    return pano_id.split("/")[-1].replace(".jpg", "")[-8:]


def render_map_image(
    world: World,
    state: AgentState,
    poles_in_view: list[PoleInView],
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Top-down map PNG for VLM navigation (panos, edges, poles, view wedge)."""
    from PIL import Image, ImageDraw, ImageFont

    min_lat, min_lon, max_lat, max_lon = _bounds(world)
    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)

    cache = cache_dir or repo_root() / ".cache" / "agent_maps"
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"map_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")

    # Edges (20 m links)
    for pano_id, neighbors in neighbor_map.items():
        a = world.panos_by_id.get(pano_id)
        if not a:
            continue
        ax, ay = _project(a.lat, a.lon, min_lat, min_lon, max_lat, max_lon)
        for nid in neighbors:
            b = world.panos_by_id.get(nid)
            if not b or pano_id > nid:
                continue
            bx, by = _project(b.lat, b.lon, min_lat, min_lon, max_lat, max_lon)
            draw.line((ax, ay, bx, by), fill=(148, 163, 184, 120), width=1)

    # Poles
    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
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
    for p in world.panos:
        px, py = _project(p.lat, p.lon, min_lat, min_lon, max_lat, max_lon)
        if p.id == state.pano_id:
            continue
        is_neighbor = p.id in get_neighbors(neighbor_map, state.pano_id)
        fill = (224, 242, 254, 255) if is_neighbor else (71, 85, 105, 200)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=fill)
        if is_neighbor:
            draw.text((px + 6, py + 6), _compact_id(p.id), fill=(186, 230, 253))

    # View wedge from current pano
    cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
    half_fov = 50
    wedge_len = 55
    points = [(cx, cy)]
    for offset in range(-half_fov, half_fov + 1, 10):
        angle = math.radians(view_yaw + offset - 90)
        wx = cx + int(math.cos(angle) * wedge_len)
        wy = cy + int(math.sin(angle) * wedge_len)
        points.append((wx, wy))
    draw.polygon(points, fill=(56, 189, 248, 70))
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(56, 189, 248, 255), outline=(255, 255, 255))

    # Legend
    legend = [
        "Map: blue=you, light=neighbor (<=20m), orange=target pole",
        "green=other poles, gray=classified, wedge=view direction",
    ]
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    y = 8
    for line in legend:
        draw.text((8, y), line, fill=(226, 232, 240), font=font)
        y += 14

    image.save(out_path, format="PNG")
    return out_path

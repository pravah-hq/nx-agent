"""
PNG local maps for VLM navigation (cached under .cache/agent_maps/).

Two map views for navigation:
  overview — clustered pano groups, thick edges (general direction only)
  node_zoom — YOU, neighbors, full MOVE ids for exact moves

Legend: overview merges nearby dots (×N); node zoom has MOVE boxes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from agent.data_loader import repo_root
from agent.directions import bin_center_world_yaw
from agent.environment import World
from agent.graph import get_neighbors
from agent.types import AgentState

MAP_SIZE = 900
PADDING_PX = 60
OVERVIEW_PAD_DEG = 0.00008
NODE_ZOOM_PAD_DEG = 0.000028
# Overview: merge panos whose projected centers are within this many pixels.
OVERVIEW_CLUSTER_PX = 34
OVERVIEW_EDGE_WIDTH = 6
OVERVIEW_EDGE_WIDTH_FROM_YOU = 8


def _load_map_font(size: int = 13):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _pano_id_display_lines(pano_id: str, *, max_chars: int = 44) -> list[str]:
    """Wrap the full pano id so rejoined lines equal target_pano_id exactly."""
    if len(pano_id) <= max_chars:
        return [pano_id]
    return [pano_id[i : i + max_chars] for i in range(0, len(pano_id), max_chars)]


def _text_block_size(
    draw,
    lines: list[str],
    font,
    *,
    title: str | None = None,
    pad_x: int = 6,
    pad_y: int = 4,
    line_gap: int = 2,
) -> tuple[int, int]:
    display = ([title] if title else []) + lines
    line_heights: list[int] = []
    max_w = 0
    for line in display:
        box = draw.textbbox((0, 0), line, font=font)
        w = box[2] - box[0]
        h = box[3] - box[1]
        max_w = max(max_w, w)
        line_heights.append(h)
    total_h = sum(line_heights) + line_gap * max(0, len(display) - 1)
    return max_w + 2 * pad_x, total_h + 2 * pad_y


def _draw_text_block(
    draw,
    top_left: tuple[int, int],
    lines: list[str],
    font,
    *,
    fill: tuple[int, int, int, int] = (15, 23, 42, 230),
    text_fill: tuple[int, int, int] = (226, 232, 240),
    title: str | None = None,
    pad_x: int = 6,
    pad_y: int = 4,
    line_gap: int = 2,
) -> None:
    display = ([title] if title else []) + lines
    w, h = _text_block_size(
        draw, lines, font, title=title, pad_x=pad_x, pad_y=pad_y, line_gap=line_gap
    )
    x0, y0 = top_left
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=fill)
    y = y0 + pad_y
    for line in display:
        draw.text((x0 + pad_x, y), line, fill=text_fill, font=font)
        box = draw.textbbox((0, 0), line, font=font)
        y += box[3] - box[1] + line_gap


def _neighbor_label_anchor(
    cx: int,
    cy: int,
    px: int,
    py: int,
    block_w: int,
    block_h: int,
) -> tuple[int, int]:
    """Place callout beside the neighbor, away from YOU."""
    dx = px - cx
    dy = py - cy
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return px + 14, py - block_h // 2
        return px - block_w - 14, py - block_h // 2
    if dy >= 0:
        return px - block_w // 2, py + 14
    return px - block_w // 2, py - block_h - 14


def _overview_bounds(
    world: World,
    state: AgentState,
    *,
    goal_pano_id: str | None,
) -> tuple[float, float, float, float]:
    pano = world.panos_by_id[state.pano_id]
    lats = [pano.lat]
    lons = [pano.lon]

    for nid in get_neighbors(world.neighbor_map, state.pano_id):
        n = world.panos_by_id[nid]
        lats.append(n.lat)
        lons.append(n.lon)

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


def _overview_visible_pano_ids(
    world: World,
    state: AgentState,
    *,
    goal_pano_id: str | None,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> set[str]:
    """Panos in the overview frame: local bounds, path to goal, and 1-hop around path."""
    from agent.targeting import path_to_pano

    visible: set[str] = {state.pano_id}
    visible.update(get_neighbors(world.neighbor_map, state.pano_id))
    if goal_pano_id:
        visible.add(goal_pano_id)

    for pano in world.panos:
        if min_lat <= pano.lat <= max_lat and min_lon <= pano.lon <= max_lon:
            visible.add(pano.id)

    if goal_pano_id:
        path = path_to_pano(world, state.pano_id, goal_pano_id)
        if path:
            visible.update(path)
            for pano_id in path:
                visible.update(get_neighbors(world.neighbor_map, pano_id))

    return visible


def _cluster_panos_by_screen_px(
    pano_ids: list[str],
    positions: dict[str, tuple[int, int]],
    *,
    threshold_px: int,
) -> dict[str, int]:
    """Union-find: panos closer than threshold_px on the map share a cluster id."""
    parent = {pid: pid for pid in pano_ids}

    def find(pid: str) -> str:
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    thresh_sq = threshold_px * threshold_px
    for i, a in enumerate(pano_ids):
        ax, ay = positions[a]
        for b in pano_ids[i + 1 :]:
            bx, by = positions[b]
            if (ax - bx) ** 2 + (ay - by) ** 2 <= thresh_sq:
                union(a, b)

    root_to_id: dict[str, int] = {}
    out: dict[str, int] = {}
    next_id = 0
    for pid in pano_ids:
        root = find(pid)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        out[pid] = root_to_id[root]
    return out


def _build_cluster_graph(
    world: World,
    visible_panos: set[str],
    pano_to_cluster: dict[str, int],
) -> set[tuple[int, int]]:
    """Undirected edges between clusters (from underlying 20 m pano graph)."""
    edges: set[tuple[int, int]] = set()
    for pano_id in visible_panos:
        ca = pano_to_cluster[pano_id]
        for nid in world.neighbor_map.get(pano_id, []):
            if nid not in visible_panos:
                continue
            cb = pano_to_cluster[nid]
            if ca == cb:
                continue
            edges.add((min(ca, cb), max(ca, cb)))
    return edges


def _render_overview_clustered(
    world: World,
    state: AgentState,
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    goal_pano_id: str | None,
    out_path: Path,
) -> Path:
    """Overview map with nearby panos merged into clusters; thick inter-cluster edges."""
    from PIL import Image, ImageDraw

    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    current_neighbors = set(get_neighbors(neighbor_map, state.pano_id))

    visible_panos = _overview_visible_pano_ids(
        world,
        state,
        goal_pano_id=goal_pano_id,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
    )

    positions: dict[str, tuple[int, int]] = {}
    for pano_id in visible_panos:
        p = world.panos_by_id[pano_id]
        positions[pano_id] = _project(
            p.lat, p.lon, min_lat, min_lon, max_lat, max_lon
        )

    pano_list = sorted(visible_panos)
    pano_to_cluster = _cluster_panos_by_screen_px(
        pano_list, positions, threshold_px=OVERVIEW_CLUSTER_PX
    )

    clusters: dict[int, list[str]] = {}
    for pid, cid in pano_to_cluster.items():
        clusters.setdefault(cid, []).append(pid)

    centroids: dict[int, tuple[int, int]] = {}
    for cid, members in clusters.items():
        xs = [positions[m][0] for m in members]
        ys = [positions[m][1] for m in members]
        centroids[cid] = (sum(xs) // len(xs), sum(ys) // len(ys))

    you_cluster = pano_to_cluster[state.pano_id]
    goal_cluster = (
        pano_to_cluster[goal_pano_id] if goal_pano_id and goal_pano_id in pano_to_cluster else None
    )

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_map_font(12)

    cluster_edges = _build_cluster_graph(world, visible_panos, pano_to_cluster)
    for ca, cb in cluster_edges:
        ax, ay = centroids[ca]
        bx, by = centroids[cb]
        touches_you = you_cluster in {ca, cb}
        width = OVERVIEW_EDGE_WIDTH_FROM_YOU if touches_you else OVERVIEW_EDGE_WIDTH
        color = (148, 163, 184, 240) if touches_you else (100, 116, 139, 200)
        draw.line((ax, ay, bx, by), fill=color, width=width)

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
            radius = 10
        else:
            color = (34, 197, 94, 255)
            radius = 6
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        draw.text(
            (px + 10, py - 8),
            pole.pole_id.replace("POLE_", ""),
            fill=(226, 232, 240),
            font=font,
        )

    for cid, members in clusters.items():
        cx, cy = centroids[cid]
        count = len(members)
        is_you = cid == you_cluster
        is_goal = goal_cluster is not None and cid == goal_cluster
        if is_you:
            r = 14
            fill = (56, 189, 248, 255)
        elif is_goal:
            r = 10
            fill = (250, 204, 21, 255)
        else:
            r = 6 + min(8, count)
            fill = (71, 85, 105, 220)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=(255, 255, 255))
        if is_you:
            label = "YOU" if count == 1 else f"YOU ×{count}"
        elif is_goal:
            label = "GOAL" if count == 1 else f"GOAL ×{count}"
        elif count > 1:
            label = f"×{count}"
        else:
            continue
        draw.text((cx + r + 4, cy - 6), label, fill=(255, 255, 255), font=font)

    cx, cy = centroids[you_cluster]
    wedge_len = 70
    half_fov = 50
    points = [(cx, cy)]
    for offset in range(-half_fov, half_fov + 1, 10):
        angle = math.radians(view_yaw + offset - 90)
        wx = cx + int(math.cos(angle) * wedge_len)
        wy = cy + int(math.sin(angle) * wedge_len)
        points.append((wx, wy))
    draw.polygon(points, fill=(56, 189, 248, 70))

    legend = [
        "OVERVIEW (direction): nearby panos merged into one dot (×N = count)",
        "Thick gray lines = graph links between merged groups",
        "Use NODE ZOOM map (image 2) for exact move target_pano_id",
    ]
    y = 8
    for line in legend:
        draw.text((8, y), line, fill=(226, 232, 240), font=font)
        y += 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return out_path


def _render_node_zoom_map(
    world: World,
    state: AgentState,
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    goal_pano_id: str | None,
    out_path: Path,
) -> Path:
    from PIL import Image, ImageDraw

    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    current_neighbors = set(get_neighbors(neighbor_map, state.pano_id))

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), (15, 23, 42))
    draw = ImageDraw.Draw(image, "RGBA")

    visible_panos = {state.pano_id, *current_neighbors}

    node_radius = 8
    you_radius = 14
    wedge_len = 110
    neighbor_edge_width = 4

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

    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id in state.classified:
            color = (100, 116, 139, 255)
            radius = 6
        elif pole.track_id == target_track:
            color = (249, 115, 22, 255)
            radius = 12
        else:
            color = (34, 197, 94, 255)
            radius = 8
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        label = pole.pole_id.replace("POLE_", "")
        draw.text((px + 10, py - 8), label, fill=(226, 232, 240))

    cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
    font = _load_map_font(13)
    id_font = _load_map_font(12)

    pano_positions: dict[str, tuple[int, int, bool, bool]] = {}
    for pano_id in visible_panos:
        p = world.panos_by_id[pano_id]
        px, py = _project(p.lat, p.lon, min_lat, min_lon, max_lat, max_lon)
        if pano_id == state.pano_id:
            continue
        is_neighbor = pano_id in current_neighbors
        is_goal = pano_id == goal_pano_id
        pano_positions[pano_id] = (px, py, is_neighbor, is_goal)
        fill = (224, 242, 254, 255) if is_neighbor else (71, 85, 105, 200)
        r = node_radius
        draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)

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
    draw.text((cx + you_radius + 4, cy - 10), "YOU", fill=(255, 255, 255), font=font)

    for pano_id, (px, py, is_neighbor, is_goal) in pano_positions.items():
        if is_neighbor:
            id_lines = _pano_id_display_lines(pano_id)
            block_w, block_h = _text_block_size(
                draw, id_lines, id_font, title="MOVE"
            )
            tx, ty = _neighbor_label_anchor(cx, cy, px, py, block_w, block_h)
            tx = max(PADDING_PX, min(tx, MAP_SIZE - block_w - PADDING_PX))
            ty = max(PADDING_PX, min(ty, MAP_SIZE - block_h - PADDING_PX))
            _draw_text_block(
                draw,
                (tx, ty),
                id_lines,
                id_font,
                title="MOVE",
                fill=(30, 41, 59, 240),
                text_fill=(224, 242, 254),
            )
            draw.line(
                (px, py, tx + block_w // 2, ty + block_h // 2),
                fill=(125, 211, 252, 180),
                width=1,
            )
        elif is_goal:
            goal_lines = _pano_id_display_lines(pano_id)
            block_w, block_h = _text_block_size(draw, goal_lines, id_font, title="GOAL")
            tx = max(PADDING_PX, min(px - block_w // 2, MAP_SIZE - block_w - PADDING_PX))
            ty = max(PADDING_PX, py - block_h - 16)
            _draw_text_block(
                draw,
                (tx, ty),
                goal_lines,
                id_font,
                title="GOAL",
                fill=(30, 41, 59, 240),
                text_fill=(250, 204, 21),
            )

    legend = [
        "NODE ZOOM: MOVE boxes show full pano id to copy for move",
        "Use overview map for general direction only",
        "JSON target_pano_id must match MOVE box text exactly",
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
) -> Path:
    """Wide local map: target pole, graph nodes, and connections."""
    min_lat, min_lon, max_lat, max_lon = _overview_bounds(
        world, state, goal_pano_id=goal_pano_id
    )
    cache = _map_cache_dir(cache_dir)
    out_path = cache / f"map_overview_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"
    return _render_overview_clustered(
        world,
        state,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
    )


def render_map_node_zoom_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
) -> Path:
    """Tight map centered on the current pano and its immediate neighbors."""
    min_lat, min_lon, max_lat, max_lon = _node_zoom_bounds(world, state)
    cache = _map_cache_dir(cache_dir)
    out_path = cache / f"map_zoom_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"
    return _render_node_zoom_map(
        world,
        state,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
    )


def render_map_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
) -> Path:
    """Overview map (used by clear-view / classification prompts)."""
    return render_map_overview_image(
        world,
        state,
        cache_dir=cache_dir,
        goal_pano_id=goal_pano_id,
    )

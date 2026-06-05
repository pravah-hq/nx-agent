"""
PNG local maps for VLM (cached under .cache/agent_maps/).

Navigation: light-theme pano graph map (nodes, edges, MOVE ids, last-move vector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
# Node zoom: merge panos near GOAL for framing (same px rule as before).
OVERVIEW_CLUSTER_PX = 34
OVERVIEW_ROAD_WIDTH = 7
OVERVIEW_ROUTE_WIDTH = 10
# Pano graph dots (small so MOVE / pole labels do not cover them).
PANO_NODE_RADIUS = 4
PANO_NODE_CALLOUT_PAD = 10

# Light map theme — VLM maps use a pale background for readability.
MAP_BG = (248, 250, 252)
MAP_TEXT = (30, 41, 59)
MAP_TEXT_STROKE = (255, 255, 255)
MAP_EDGE_ACTIVE = (71, 85, 105, 255)
MAP_EDGE_DIM = (148, 163, 184, 200)
MAP_NODE_NEIGHBOR = (37, 99, 235, 255)
MAP_NODE_NEIGHBOR_VISITED = (124, 58, 237, 255)
MAP_NODE_GOAL = (217, 119, 6, 255)
MAP_NODE_OTHER = (100, 116, 139, 255)
MAP_NODE_VISITED = (148, 163, 184, 255)
MAP_YOU_FILL = (37, 99, 235, 255)
MAP_YOU_OUTLINE = (255, 255, 255)
MAP_YOU_WEDGE = (59, 130, 246, 55)
MAP_YOU_LABEL = (255, 255, 255)
MAP_POLE_CLASSIFIED = (148, 163, 184, 255)
MAP_POLE_TARGET = (234, 88, 12, 255)
MAP_POLE_OPEN = (22, 163, 74, 255)
MAP_MOVE_TEXT = (29, 78, 216)
MAP_MOVE_LINE = (59, 130, 246, 220)
MAP_GOAL_TEXT = (180, 83, 9)
MAP_GOAL_LINE = (217, 119, 6, 220)
MAP_LAST_MOVE = (219, 39, 119, 255)
MAP_ROAD = (100, 116, 139, 220)
MAP_ROUTE = (217, 119, 6, 245)


@dataclass(frozen=True)
class OverviewMapBounds:
    """Lat/lon extents used to project the overview map image (MAP_SIZE px)."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


def overview_bounds_for_state(
    world: World,
    state: AgentState,
    *,
    goal_pano_id: str | None = None,
) -> OverviewMapBounds:
    min_lat, min_lon, max_lat, max_lon = _overview_bounds(
        world, state, goal_pano_id=goal_pano_id
    )
    return OverviewMapBounds(min_lat, min_lon, max_lat, max_lon)


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


def _visited_cache_suffix(state: AgentState) -> str:
    """Disambiguate cached map PNGs when the visited-pano set changes."""
    tag = hash(frozenset(state.visited_pano_ids)) & 0xFFFFFFF
    return f"_v{tag:x}"


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
    background: bool = False,
    fill: tuple[int, int, int, int] = (*MAP_BG, 96),
    text_fill: tuple[int, int, int] = MAP_TEXT,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = MAP_TEXT_STROKE,
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
    if background:
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=fill)
    outline = stroke_width if stroke_width > 0 else (2 if not background else 0)
    y = y0 + pad_y
    for line in display:
        draw.text(
            (x0 + pad_x, y),
            line,
            fill=text_fill,
            font=font,
            stroke_width=outline,
            stroke_fill=stroke_fill,
        )
        box = draw.textbbox((0, 0), line, font=font)
        y += box[3] - box[1] + line_gap


def _rect_from_xywh(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return (x, y, x + w, y + h)


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    margin: int = 6,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + margin <= bx0
        or bx1 + margin <= ax0
        or ay1 + margin <= by0
        or by1 + margin <= ay0
    )


def _circle_rect_overlap(
    cx: int,
    cy: int,
    radius: int,
    rect: tuple[int, int, int, int],
    *,
    margin: int = 4,
) -> bool:
    x0, y0, x1, y1 = rect
    closest_x = min(max(cx, x0), x1)
    closest_y = min(max(cy, y0), y1)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < (radius + margin) ** 2


def _clamp_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    pad: int = PADDING_PX,
) -> tuple[int, int, int, int]:
    x = max(pad, min(x, MAP_SIZE - w - pad))
    y = max(pad, min(y, MAP_SIZE - h - pad))
    return _rect_from_xywh(x, y, w, h)


def _callout_candidate_positions(
    node_x: int,
    node_y: int,
    block_w: int,
    block_h: int,
    you_x: int,
    you_y: int,
) -> list[tuple[int, int]]:
    """Candidate top-left positions; primary push is away from YOU."""
    dx = node_x - you_x
    dy = node_y - you_y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / dist, dy / dist
    tangent_x, tangent_y = -uy, ux

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for outward in (22, 32, 44, 56, 70, 86):
        for tangential in (0, 1, -1, 2, -2, 3, -3):
            shift = tangential * (block_h + 10)
            cx = node_x + ux * outward + tangent_x * shift
            cy = node_y + uy * outward + tangent_y * shift
            x = int(cx - block_w / 2)
            y = int(cy - block_h / 2)
            key = (x // 4, y // 4)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((x, y))
    return candidates


def _layout_callout_rect(
    node_x: int,
    node_y: int,
    node_radius: int,
    block_w: int,
    block_h: int,
    you_x: int,
    you_y: int,
    you_radius: int,
    obstacles: list[tuple[str, tuple]],
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for x, y in _callout_candidate_positions(
        node_x, node_y, block_w, block_h, you_x, you_y
    ):
        rect = _clamp_rect(x, y, block_w, block_h)
        if _circle_rect_overlap(node_x, node_y, node_radius + 6, rect):
            continue
        if _circle_rect_overlap(you_x, you_y, you_radius + 12, rect):
            continue
        blocked = False
        for kind, obs in obstacles:
            if kind == "circle":
                ox, oy, orad = obs
                if _circle_rect_overlap(ox, oy, orad, rect):
                    blocked = True
                    break
            elif kind == "rect":
                if _rects_overlap(rect, obs):
                    blocked = True
                    break
        if blocked:
            continue
        for other in placed:
            if _rects_overlap(rect, other):
                blocked = True
                break
        if not blocked:
            placed.append(rect)
            return rect
    return None


def _leader_line_to_box(
    node_x: int,
    node_y: int,
    node_radius: int,
    rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    x0, y0, x1, y1 = rect
    bcx = (x0 + x1) // 2
    bcy = (y0 + y1) // 2
    angle = math.atan2(bcy - node_y, bcx - node_x)
    sx = node_x + int(math.cos(angle) * (node_radius + 3))
    sy = node_y + int(math.sin(angle) * (node_radius + 3))
    ex = bcx
    ey = bcy
    if abs(math.cos(angle)) > abs(math.sin(angle)):
        ex = x0 if bcx < node_x else x1
    else:
        ey = y0 if bcy < node_y else y1
    return (sx, sy), (ex, ey)


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


def _goal_cluster_pano_ids(
    world: World,
    state: AgentState,
    goal_pano_id: str | None,
) -> frozenset[str]:
    """
    Pano ids merged with GOAL on the overview map (same screen-pixel clustering).
    """
    if not goal_pano_id or goal_pano_id not in world.panos_by_id:
        return frozenset()

    pano = world.panos_by_id[state.pano_id]
    goal = world.panos_by_id[goal_pano_id]
    candidates: set[str] = {state.pano_id, goal_pano_id}
    candidates.update(get_neighbors(world.neighbor_map, state.pano_id))
    candidates.update(get_neighbors(world.neighbor_map, goal_pano_id))

    lo_lat = min(pano.lat, goal.lat)
    hi_lat = max(pano.lat, goal.lat)
    lo_lon = min(pano.lon, goal.lon)
    hi_lon = max(pano.lon, goal.lon)
    margin = NODE_ZOOM_PAD_DEG * 4
    for p in world.panos:
        if lo_lat - margin <= p.lat <= hi_lat + margin and lo_lon - margin <= p.lon <= hi_lon + margin:
            candidates.add(p.id)

    pre_pad = NODE_ZOOM_PAD_DEG * 2
    min_lat = min(world.panos_by_id[pid].lat for pid in candidates) - pre_pad
    max_lat = max(world.panos_by_id[pid].lat for pid in candidates) + pre_pad
    min_lon = min(world.panos_by_id[pid].lon for pid in candidates) - pre_pad
    max_lon = max(world.panos_by_id[pid].lon for pid in candidates) + pre_pad

    pano_list = sorted(candidates)
    positions = {
        pid: _project(
            world.panos_by_id[pid].lat,
            world.panos_by_id[pid].lon,
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        )
        for pid in pano_list
    }
    pano_to_cluster = _cluster_panos_by_screen_px(
        pano_list, positions, threshold_px=OVERVIEW_CLUSTER_PX
    )
    goal_cid = pano_to_cluster[goal_pano_id]
    return frozenset(pid for pid in pano_list if pano_to_cluster[pid] == goal_cid)


def _symmetric_bounds_between(
    anchor_a_lat: float,
    anchor_a_lon: float,
    anchor_b_lat: float,
    anchor_b_lon: float,
    points: list[tuple[float, float]],
    *,
    pad_deg: float,
    min_half_span_deg: float,
) -> tuple[float, float, float, float]:
    """Map bounds centered on the midpoint of two anchors, spanning all points."""
    center_lat = (anchor_a_lat + anchor_b_lat) / 2
    center_lon = (anchor_a_lon + anchor_b_lon) / 2
    half = min_half_span_deg
    for lat, lon in points:
        half = max(half, abs(lat - center_lat), abs(lon - center_lon))
    half += pad_deg
    return (
        center_lat - half,
        center_lon - half,
        center_lat + half,
        center_lon + half,
    )


def _node_zoom_bounds(
    world: World,
    state: AgentState,
    *,
    goal_pano_id: str | None = None,
) -> tuple[float, float, float, float]:
    """Bounds centered between YOU and the overview GOAL cluster (includes neighbors)."""
    pano = world.panos_by_id[state.pano_id]
    points: list[tuple[float, float]] = [(pano.lat, pano.lon)]

    for nid in get_neighbors(world.neighbor_map, state.pano_id):
        n = world.panos_by_id[nid]
        points.append((n.lat, n.lon))

    track = state.pole_in_consideration
    if track and track in world.poles_by_track:
        pole = world.poles_by_track[track]
        points.append((pole.lat, pole.lon))

    goal_cluster = _goal_cluster_pano_ids(world, state, goal_pano_id)
    if goal_cluster:
        goal_lats = [world.panos_by_id[pid].lat for pid in goal_cluster]
        goal_lons = [world.panos_by_id[pid].lon for pid in goal_cluster]
        goal_centroid_lat = sum(goal_lats) / len(goal_lats)
        goal_centroid_lon = sum(goal_lons) / len(goal_lons)
        for pid in goal_cluster:
            p = world.panos_by_id[pid]
            points.append((p.lat, p.lon))
        return _symmetric_bounds_between(
            pano.lat,
            pano.lon,
            goal_centroid_lat,
            goal_centroid_lon,
            points,
            pad_deg=NODE_ZOOM_PAD_DEG,
            min_half_span_deg=NODE_ZOOM_PAD_DEG * 2,
        )

    if goal_pano_id and goal_pano_id in world.panos_by_id:
        g = world.panos_by_id[goal_pano_id]
        points.append((g.lat, g.lon))
        return _symmetric_bounds_between(
            pano.lat,
            pano.lon,
            g.lat,
            g.lon,
            points,
            pad_deg=NODE_ZOOM_PAD_DEG,
            min_half_span_deg=NODE_ZOOM_PAD_DEG * 2,
        )

    pad = NODE_ZOOM_PAD_DEG
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad


def _facing_up_rotation_deg(view_yaw_deg: float) -> float:
    """
    PIL rotate (CCW, degrees) so world view_yaw points to the top of the image.

    North-up maps use the same wedge math as _draw_view_wedge (bearing from +x).
    """
    yaw_rad = math.radians(view_yaw_deg)
    phi_deg = math.degrees(math.atan2(-math.cos(yaw_rad), math.sin(yaw_rad)))
    return -90.0 - phi_deg


def _rotate_point_around(
    px: int,
    py: int,
    center: tuple[int, int],
    rot_deg: float,
) -> tuple[int, int]:
    """Map a pre-rotation pixel to its position after PIL CCW rotate by rot_deg."""
    if abs(rot_deg) < 0.05:
        return px, py
    cx, cy = center
    rad = math.radians(rot_deg)
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    dx = px - cx
    dy = py - cy
    return (
        int(round(cx + cos_t * dx - sin_t * dy)),
        int(round(cy + sin_t * dx + cos_t * dy)),
    )


def _rotate_image_heading_up(
    image,
    center: tuple[int, int],
    view_yaw_deg: float,
) -> tuple[object, float]:
    """Rotate map around agent position so facing is up; returns (image, rot_deg)."""
    from PIL import Image

    rot_deg = _facing_up_rotation_deg(view_yaw_deg)
    if abs(rot_deg) < 0.05:
        return image, rot_deg
    return (
        image.rotate(
            rot_deg,
            center=center,
            resample=Image.Resampling.BICUBIC,
            expand=False,
        ),
        rot_deg,
    )


def _draw_you_marker_facing_up(
    draw,
    cx: int,
    cy: int,
    *,
    you_radius: int,
    wedge_len: int,
    font,
    half_fov: int = 50,
    wedge_fill: tuple[int, int, int, int] = MAP_YOU_WEDGE,
) -> None:
    """YOU + view wedge with facing toward the top of the image (post-rotation)."""
    points = [(cx, cy)]
    for offset in range(-half_fov, half_fov + 1, 10):
        ang = math.radians(offset)
        wx = cx + int(math.sin(ang) * wedge_len)
        wy = cy - int(math.cos(ang) * wedge_len)
        points.append((wx, wy))
    draw.polygon(points, fill=wedge_fill)
    draw.ellipse(
        (cx - you_radius, cy - you_radius, cx + you_radius, cy + you_radius),
        fill=MAP_YOU_FILL,
        outline=MAP_YOU_OUTLINE,
    )
    draw.text(
        (cx + you_radius + 4, cy - 8),
        "YOU",
        fill=MAP_YOU_LABEL,
        font=font,
        stroke_width=2,
        stroke_fill=MAP_TEXT,
    )


def _draw_pano_graph_edges(
    draw,
    *,
    cx: int,
    cy: int,
    current_pano_id: str,
    pano_positions: dict[str, tuple[int, int, bool, bool]],
    neighbor_map: dict[str, list[str]],
    visible_panos: set[str],
    neighbor_edge_width: int,
) -> None:
    """Draw pano graph edges on top of labels (post-rotation coordinates)."""
    for pano_id in visible_panos:
        for nid in neighbor_map.get(pano_id, []):
            if nid not in visible_panos or pano_id > nid:
                continue
            is_from_current = pano_id == current_pano_id or nid == current_pano_id
            width = neighbor_edge_width if is_from_current else 1
            color = MAP_EDGE_ACTIVE if is_from_current else MAP_EDGE_DIM
            if pano_id == current_pano_id:
                ax, ay = cx, cy
            else:
                ax, ay = pano_positions[pano_id][0], pano_positions[pano_id][1]
            if nid == current_pano_id:
                bx, by = cx, cy
            else:
                bx, by = pano_positions[nid][0], pano_positions[nid][1]
            draw.line((ax, ay, bx, by), fill=color, width=width)


def _pano_node_fill(*, is_neighbor: bool, is_goal: bool, is_visited: bool) -> tuple[int, int, int, int]:
    if is_goal:
        return MAP_NODE_GOAL
    if is_neighbor:
        return MAP_NODE_NEIGHBOR_VISITED if is_visited else MAP_NODE_NEIGHBOR
    return MAP_NODE_VISITED if is_visited else MAP_NODE_OTHER


def _draw_pano_node_dots(
    draw,
    pano_positions: dict[str, tuple[int, int, bool, bool, bool]],
    *,
    node_radius: int,
) -> None:
    """Draw neighbor / goal pano dots on top of labels and leader lines."""
    for _pano_id, (px, py, is_neighbor, is_goal, is_visited) in pano_positions.items():
        fill = _pano_node_fill(
            is_neighbor=is_neighbor, is_goal=is_goal, is_visited=is_visited
        )
        r = node_radius
        draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)


def _draw_map_legend(draw, lines: list[str], *, font) -> None:
    y = 8
    for line in lines:
        draw.text(
            (8, y),
            line,
            fill=MAP_TEXT,
            font=font,
            stroke_width=2,
            stroke_fill=MAP_TEXT_STROKE,
        )
        y += 14


def _draw_last_move_vector(
    draw,
    cx: int,
    cy: int,
    view_yaw_deg: float,
    last_move_bearing_deg: float | None,
    *,
    font,
    length: int = 88,
) -> None:
    """Magenta arrow from YOU along previous move bearing (heading-up map)."""
    if last_move_bearing_deg is None:
        return
    delta = ((last_move_bearing_deg - view_yaw_deg + 540.0) % 360.0) - 180.0
    rad = math.radians(delta)
    ex = cx + int(math.sin(rad) * length)
    ey = cy - int(math.cos(rad) * length)
    color = MAP_LAST_MOVE
    draw.line((cx, cy, ex, ey), fill=color, width=4)
    head = 10
    left = math.radians(delta - 150)
    right = math.radians(delta + 150)
    draw.polygon(
        [
            (ex, ey),
            (ex + int(math.sin(left) * head), ey - int(math.cos(left) * head)),
            (ex + int(math.sin(right) * head), ey - int(math.cos(right) * head)),
        ],
        fill=color,
    )
    draw.text(
        (ex + 8, ey - 10),
        "last move",
        fill=color,
        font=font,
        stroke_width=2,
        stroke_fill=MAP_TEXT_STROKE,
    )


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


def _norm_pano_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


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


def _visible_graph_edges(world: World, visible_panos: set[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for pano_id in visible_panos:
        for nid in world.neighbor_map.get(pano_id, []):
            if nid in visible_panos and pano_id < nid:
                edges.append((pano_id, nid))
    return edges


def _path_edge_set(path: list[str] | None) -> set[tuple[str, str]]:
    if not path or len(path) < 2:
        return set()
    return {_norm_pano_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}


def _extract_road_polylines(
    world: World,
    visible_panos: set[str],
) -> list[list[str]]:
    """
    Chain 20 m graph edges into road polylines (merge straight runs through degree-2 nodes).
    """
    adj: dict[str, list[str]] = {p: [] for p in visible_panos}
    for a in visible_panos:
        for b in world.neighbor_map.get(a, []):
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
            e = _norm_pano_edge(tip, nxt)
            if e in used:
                break
            used.add(e)
            if backward:
                chain.insert(0, nxt)
            else:
                chain.append(nxt)

    for a, b in _visible_graph_edges(world, visible_panos):
        e = _norm_pano_edge(a, b)
        if e in used:
            continue
        used.add(e)
        chain = [a, b]
        extend_from_end(chain, backward=True)
        extend_from_end(chain, backward=False)
        if len(chain) >= 2:
            polylines.append(chain)

    return polylines


def _draw_road_polyline(
    draw,
    chain: list[str],
    positions: dict[str, tuple[int, int]],
    *,
    width: int,
    fill: tuple[int, int, int, int],
) -> None:
    for i in range(len(chain) - 1):
        ax, ay = positions[chain[i]]
        bx, by = positions[chain[i + 1]]
        draw.line((ax, ay, bx, by), fill=fill, width=width)


def _polyline_uses_route(
    chain: list[str],
    route_edges: set[tuple[str, str]],
) -> bool:
    return any(
        _norm_pano_edge(chain[i], chain[i + 1]) in route_edges
        for i in range(len(chain) - 1)
    )


def _render_overview_roads(
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
    """Overview: road network from pano graph; no pano nodes shown."""
    from PIL import Image, ImageDraw

    from agent.targeting import path_to_pano

    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)

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

    route_path = (
        path_to_pano(world, state.pano_id, goal_pano_id) if goal_pano_id else None
    )
    route_edges = _path_edge_set(route_path)
    polylines = _extract_road_polylines(world, visible_panos)

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), MAP_BG)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_map_font(12)

    road_color = MAP_ROAD
    route_color = MAP_ROUTE

    for chain in polylines:
        if _polyline_uses_route(chain, route_edges):
            continue
        _draw_road_polyline(
            draw,
            chain,
            positions,
            width=OVERVIEW_ROAD_WIDTH,
            fill=road_color,
        )

    for chain in polylines:
        if not _polyline_uses_route(chain, route_edges):
            continue
        _draw_road_polyline(
            draw,
            chain,
            positions,
            width=OVERVIEW_ROUTE_WIDTH,
            fill=route_color,
        )

    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id in state.classified:
            color = MAP_POLE_CLASSIFIED
            radius = 5
        elif pole.track_id == target_track:
            color = MAP_POLE_TARGET
            radius = 12
        else:
            color = MAP_POLE_OPEN
            radius = 5
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    cx, cy = positions[state.pano_id]
    image, rot_deg = _rotate_image_heading_up(image, (cx, cy), view_yaw)
    draw = ImageDraw.Draw(image, "RGBA")

    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id == target_track:
            rpx, rpy = _rotate_point_around(px, py, (cx, cy), rot_deg)
            draw.text(
                (rpx + 12, rpy - 10),
                pole.pole_id.replace("POLE_", ""),
                fill=MAP_TEXT,
                font=font,
                stroke_width=2,
                stroke_fill=MAP_TEXT_STROKE,
            )

    _draw_you_marker_facing_up(
        draw, cx, cy, you_radius=12, wedge_len=70, font=font, wedge_fill=MAP_YOU_WEDGE
    )
    _draw_map_legend(
        draw,
        [
            "Map up = your facing (matches street view)",
            "OVERVIEW: gray roads = walkable paths along pano graph",
            "Yellow road = suggested route toward orange target pole",
            "No pano nodes — pick exact move from NODE ZOOM (image 2)",
        ],
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return out_path


def _render_graph_map(
    world: World,
    state: AgentState,
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    goal_pano_id: str | None,
    out_path: Path,
    visible_panos: set[str] | None = None,
    last_move_bearing_deg: float | None = None,
    show_move_callouts: bool = True,
) -> Path:
    """Local pano graph: nodes, edges, poles; optional MOVE callouts and last-move arrow."""
    from PIL import Image, ImageDraw

    neighbor_map = world.neighbor_map
    pano = world.panos_by_id[state.pano_id]
    view_yaw = bin_center_world_yaw(pano, state.direction_bin)
    current_neighbors = set(get_neighbors(neighbor_map, state.pano_id))
    goal_cluster = _goal_cluster_pano_ids(world, state, goal_pano_id)

    image = Image.new("RGB", (MAP_SIZE, MAP_SIZE), MAP_BG)
    draw = ImageDraw.Draw(image, "RGBA")

    if visible_panos is None:
        visible_panos = {state.pano_id, *current_neighbors, *goal_cluster}
    else:
        visible_panos = set(visible_panos)
        visible_panos.add(state.pano_id)

    node_radius = PANO_NODE_RADIUS
    node_callout_radius = node_radius + PANO_NODE_CALLOUT_PAD
    you_radius = 14
    wedge_len = 110
    neighbor_edge_width = 4

    target_track = state.pole_in_consideration
    for pole in world.poles:
        px, py = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if px < PADDING_PX - 20 or py < PADDING_PX - 20:
            continue
        if px > MAP_SIZE or py > MAP_SIZE:
            continue
        if pole.track_id in state.classified:
            color = MAP_POLE_CLASSIFIED
            radius = 6
        elif pole.track_id == target_track:
            color = MAP_POLE_TARGET
            radius = 12
        else:
            color = MAP_POLE_OPEN
            radius = 8
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    cx, cy = _project(pano.lat, pano.lon, min_lat, min_lon, max_lat, max_lon)
    font = _load_map_font(13)
    id_font = _load_map_font(12)

    visited = state.visited_pano_ids
    pano_positions_raw: dict[str, tuple[int, int, bool, bool, bool]] = {}
    for pano_id in visible_panos:
        p = world.panos_by_id[pano_id]
        px, py = _project(p.lat, p.lon, min_lat, min_lon, max_lat, max_lon)
        if pano_id == state.pano_id:
            continue
        is_neighbor = pano_id in current_neighbors
        is_goal = pano_id in goal_cluster or pano_id == goal_pano_id
        is_visited = pano_id in visited
        pano_positions_raw[pano_id] = (px, py, is_neighbor, is_goal, is_visited)

    image, rot_deg = _rotate_image_heading_up(image, (cx, cy), view_yaw)
    draw = ImageDraw.Draw(image, "RGBA")

    pano_positions: dict[str, tuple[int, int, bool, bool, bool]] = {}
    for pano_id, (px, py, is_neighbor, is_goal, is_visited) in pano_positions_raw.items():
        rpx, rpy = _rotate_point_around(px, py, (cx, cy), rot_deg)
        pano_positions[pano_id] = (rpx, rpy, is_neighbor, is_goal, is_visited)

    for pole in world.poles:
        ppx, ppy = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if PADDING_PX - 20 <= ppx <= MAP_SIZE and PADDING_PX - 20 <= ppy <= MAP_SIZE:
            rpx, rpy = _rotate_point_around(ppx, ppy, (cx, cy), rot_deg)
            label = pole.pole_id.replace("POLE_", "")
            if pole.track_id in state.classified:
                label_dx, label_dy = 12, -7
            elif pole.track_id == target_track:
                label_dx, label_dy = 16, -8
            else:
                label_dx, label_dy = 14, -7
            draw.text(
                (rpx + label_dx, rpy + label_dy),
                label,
                fill=MAP_TEXT,
                stroke_width=2,
                stroke_fill=MAP_TEXT_STROKE,
            )

    obstacles: list[tuple[str, tuple]] = [
        ("rect", _rect_from_xywh(4, 4, 520, 52)),
        ("circle", (cx, cy, you_radius + wedge_len)),
    ]
    for pole in world.poles:
        ppx, ppy = _project(pole.lat, pole.lon, min_lat, min_lon, max_lat, max_lon)
        if PADDING_PX - 20 <= ppx <= MAP_SIZE and PADDING_PX - 20 <= ppy <= MAP_SIZE:
            rpx, rpy = _rotate_point_around(ppx, ppy, (cx, cy), rot_deg)
            pr = 14 if pole.track_id == target_track else 10
            obstacles.append(("circle", (rpx, rpy, pr)))
    for pano_id, (px, py, _is_neighbor, _is_goal, _is_visited) in pano_positions.items():
        obstacles.append(("circle", (px, py, node_callout_radius)))

    placed_boxes: list[tuple[int, int, int, int]] = []

    if show_move_callouts:
        move_items = [
            (pano_id, px, py)
            for pano_id, (px, py, is_neighbor, _is_goal, _is_visited) in pano_positions.items()
            if is_neighbor
        ]
        move_items.sort(key=lambda item: math.atan2(item[2] - cy, item[1] - cx))
    else:
        move_items = []

    for pano_id, px, py in move_items:
        id_lines = _pano_id_display_lines(pano_id)
        block_w, block_h = _text_block_size(
            draw, id_lines, id_font, title="MOVE", pad_x=2, pad_y=2
        )
        rect = _layout_callout_rect(
            px,
            py,
            node_callout_radius,
            block_w,
            block_h,
            cx,
            cy,
            you_radius,
            obstacles,
            placed_boxes,
        )
        if rect is None:
            rect = _clamp_rect(px + 20, py - block_h // 2, block_w, block_h)
            placed_boxes.append(rect)
        x0, y0, x1, y1 = rect
        _draw_text_block(
            draw,
            (x0, y0),
            id_lines,
            id_font,
            title="MOVE",
            pad_x=2,
            pad_y=2,
            text_fill=MAP_MOVE_TEXT,
        )
        start, end = _leader_line_to_box(px, py, node_radius, rect)
        draw.line((*start, *end), fill=MAP_MOVE_LINE, width=1)

    if show_move_callouts:
        goal_items = [
            (pano_id, px, py)
            for pano_id, (px, py, is_neighbor, is_goal, _is_visited) in pano_positions.items()
            if is_goal and not is_neighbor
        ]
        goal_items.sort(key=lambda item: math.atan2(item[2] - cy, item[1] - cx))
    else:
        goal_items = []

    for pano_id, px, py in goal_items:
        goal_lines = _pano_id_display_lines(pano_id)
        block_w, block_h = _text_block_size(
            draw, goal_lines, id_font, title="GOAL", pad_x=2, pad_y=2
        )
        rect = _layout_callout_rect(
            px,
            py,
            node_callout_radius,
            block_w,
            block_h,
            cx,
            cy,
            you_radius,
            obstacles,
            placed_boxes,
        )
        if rect is None:
            rect = _clamp_rect(px - block_w // 2, py - block_h - 20, block_w, block_h)
            placed_boxes.append(rect)
        x0, y0, x1, y1 = rect
        _draw_text_block(
            draw,
            (x0, y0),
            goal_lines,
            id_font,
            title="GOAL",
            pad_x=2,
            pad_y=2,
            text_fill=MAP_GOAL_TEXT,
        )
        start, end = _leader_line_to_box(px, py, node_radius, rect)
        draw.line((*start, *end), fill=MAP_GOAL_LINE, width=1)

    legend = [
        "Map up = your facing (matches street view)",
        "Gray lines = 20 m pano graph edges; blue dots = unvisited neighbors; purple = visited",
    ]
    if last_move_bearing_deg is not None:
        legend.append("Magenta arrow from YOU = direction you moved last step")
    if show_move_callouts:
        legend.append("MOVE boxes = copy target_pano_id for move")
    _draw_map_legend(draw, legend, font=font)

    _draw_pano_graph_edges(
        draw,
        cx=cx,
        cy=cy,
        current_pano_id=state.pano_id,
        pano_positions=pano_positions,
        neighbor_map=neighbor_map,
        visible_panos=visible_panos,
        neighbor_edge_width=neighbor_edge_width,
    )
    _draw_pano_node_dots(draw, pano_positions, node_radius=node_radius)
    _draw_last_move_vector(
        draw, cx, cy, view_yaw, last_move_bearing_deg, font=_load_map_font(11)
    )
    _draw_you_marker_facing_up(
        draw,
        cx,
        cy,
        you_radius=you_radius,
        wedge_len=wedge_len,
        font=font,
    )

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
    return _render_graph_map(
        world,
        state,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
        show_move_callouts=True,
        last_move_bearing_deg=state.last_move_bearing_deg,
    )


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
) -> tuple[Path, OverviewMapBounds]:
    """Overview: road network from pano graph; no pano nodes on map."""
    bounds = overview_bounds_for_state(world, state, goal_pano_id=goal_pano_id)
    cache = _map_cache_dir(cache_dir)
    out_path = cache / f"map_overview_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}.png"
    path = _render_overview_roads(
        world,
        state,
        min_lat=bounds.min_lat,
        min_lon=bounds.min_lon,
        max_lat=bounds.max_lat,
        max_lon=bounds.max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
    )
    return path, bounds


def render_map_node_zoom_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
) -> Path:
    """Map centered between YOU and the overview GOAL cluster; MOVE ids on neighbors."""
    min_lat, min_lon, max_lat, max_lon = _node_zoom_bounds(
        world, state, goal_pano_id=goal_pano_id
    )
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


def render_map_navigation_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
) -> Path:
    """Navigation map: pano nodes, edges, MOVE callouts, last-move vector."""
    bounds = overview_bounds_for_state(world, state, goal_pano_id=goal_pano_id)
    visible = _overview_visible_pano_ids(
        world,
        state,
        goal_pano_id=goal_pano_id,
        min_lat=bounds.min_lat,
        min_lon=bounds.min_lon,
        max_lat=bounds.max_lat,
        max_lon=bounds.max_lon,
    )
    cache = _map_cache_dir(cache_dir)
    out_path = (
        cache
        / f"map_graph_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}{_visited_cache_suffix(state)}.png"
    )
    return _render_graph_map(
        world,
        state,
        min_lat=bounds.min_lat,
        min_lon=bounds.min_lon,
        max_lat=bounds.max_lat,
        max_lon=bounds.max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
        visible_panos=visible,
        last_move_bearing_deg=state.last_move_bearing_deg,
        show_move_callouts=True,
    )


def render_map_image(
    world: World,
    state: AgentState,
    *,
    cache_dir: Path | None = None,
    goal_pano_id: str | None = None,
) -> Path:
    """Graph map for clear-view / classification (no MOVE callouts)."""
    bounds = overview_bounds_for_state(world, state, goal_pano_id=goal_pano_id)
    visible = _overview_visible_pano_ids(
        world,
        state,
        goal_pano_id=goal_pano_id,
        min_lat=bounds.min_lat,
        min_lon=bounds.min_lon,
        max_lat=bounds.max_lat,
        max_lon=bounds.max_lon,
    )
    cache = _map_cache_dir(cache_dir)
    out_path = (
        cache
        / f"map_observe_{state.pano_id.replace('/', '_')}_bin{state.direction_bin}{_visited_cache_suffix(state)}.png"
    )
    return _render_graph_map(
        world,
        state,
        min_lat=bounds.min_lat,
        min_lon=bounds.min_lon,
        max_lat=bounds.max_lat,
        max_lon=bounds.max_lon,
        goal_pano_id=goal_pano_id,
        out_path=out_path,
        visible_panos=visible,
        last_move_bearing_deg=state.last_move_bearing_deg,
        show_move_callouts=False,
    )

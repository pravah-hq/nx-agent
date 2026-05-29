from __future__ import annotations

import json
from pathlib import Path

from agent.types import Pano, Pole


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_panos(metadata_dir: Path | None = None) -> list[Pano]:
    root = metadata_dir or repo_root() / "data" / "metadata"
    path = root / "panoramas.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    panos: list[Pano] = []
    for row in payload["panoramas"]:
        panos.append(
            Pano(
                id=row["id"],
                image_path=row["imagePath"],
                session_id=row["sessionId"],
                order_in_session=int(row["orderInSession"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                heading_deg=float(row["headingDeg"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    panos.sort(key=lambda p: (p.session_id, p.order_in_session, p.id))
    return panos


def load_poles(metadata_dir: Path | None = None) -> list[Pole]:
    root = metadata_dir or repo_root() / "data" / "metadata"
    path = root / "poles.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    poles: list[Pole] = []
    for feature in payload["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        poles.append(
            Pole(
                track_id=props["track_id"],
                pole_id=props["pole_id"],
                lat=float(lat),
                lon=float(lon),
                quality=props.get("quality"),
                pole_material=props.get("pole_material"),
                n_sightings=props.get("n_sightings"),
                service_drop=props.get("service_drop"),
            )
        )
    poles.sort(key=lambda p: p.pole_id)
    return poles

import L from "leaflet";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Pano = {
  id: string;
  imagePath: string;
  sessionId: string;
  orderInSession: number;
  lat: number;
  lon: number;
  headingDeg: number;
  width: number;
  height: number;
};

type PanoManifest = {
  runId: string;
  imageRoot: string;
  panoramas: Pano[];
};

type PoleProperties = {
  track_id: string;
  pole_id: string;
  quality: string | null;
  pole_material: string | null;
  n_sightings: number | null;
  service_drop: boolean | null;
};

type PoleFeature = GeoJSON.Feature<GeoJSON.Point, PoleProperties>;
type PoleFeatureCollection = GeoJSON.FeatureCollection<GeoJSON.Point, PoleProperties>;

type LoadedData = {
  manifest: PanoManifest;
  poles: PoleFeatureCollection;
  area: GeoJSON.FeatureCollection;
};

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const STEP_ALIGNMENT_MAX_DEG = 45;
const LONG_SESSION_EDGES_TO_REWIRE = 2;
const LOOK_YAW_DEG_PER_SECOND = 340;
const LOOK_PITCH_DEG_PER_SECOND = 240;
const LOOK_SHIFT_MULTIPLIER = 2;
const LOOK_KEYS = new Set(["arrowleft", "arrowright", "arrowup", "arrowdown", "a", "d", "w", "s"]);
const DEFAULT_HFOV_DEG = 100;
const VIEW_SHED_RADIUS_M = 42;
const SELECTED_POLE_MARKER_COLOR = "#22c55e";
const POLE_MARKER_COLOR = "#64748b";
const ACTIVE_PANO_COLOR = "#38bdf8";

const MAP_TYPES = [
  {
    id: "dark",
    label: "Dark",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; OpenStreetMap contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 20,
  },
  {
    id: "streets",
    label: "Streets",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 20,
  },
  {
    id: "satellite",
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri",
    maxZoom: 19,
  },
  {
    id: "light",
    label: "Light",
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; OpenStreetMap contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 20,
  },
] as const;

type MapTypeId = (typeof MAP_TYPES)[number]["id"];

function normalizeDeg(value: number): number {
  return ((value % 360) + 360) % 360;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function angleDiffDeg(a: number, b: number): number {
  return Math.abs(((a - b + 540) % 360) - 180);
}

function bearingDeg(from: Pano, to: Pano): number {
  const lon1 = (from.lon * Math.PI) / 180;
  const lat1 = (from.lat * Math.PI) / 180;
  const lon2 = (to.lon * Math.PI) / 180;
  const lat2 = (to.lat * Math.PI) / 180;
  const dLon = lon2 - lon1;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return normalizeDeg((Math.atan2(y, x) * 180) / Math.PI);
}

function distanceM(a: Pano, b: Pano): number {
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const dLat = lat2 - lat1;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371000 * Math.asin(Math.sqrt(h));
}

function destination(lon: number, lat: number, bearing: number, meters: number): [number, number] {
  const radius = 6371000;
  const delta = meters / radius;
  const theta = (bearing * Math.PI) / 180;
  const phi1 = (lat * Math.PI) / 180;
  const lambda1 = (lon * Math.PI) / 180;
  const sinPhi2 =
    Math.sin(phi1) * Math.cos(delta) +
    Math.cos(phi1) * Math.sin(delta) * Math.cos(theta);
  const phi2 = Math.asin(sinPhi2);
  const y = Math.sin(theta) * Math.sin(delta) * Math.cos(phi1);
  const x = Math.cos(delta) - Math.sin(phi1) * Math.sin(phi2);
  const lambda2 = lambda1 + Math.atan2(y, x);
  return [(lambda2 * 180) / Math.PI, (phi2 * 180) / Math.PI];
}

function viewshedCoords(pano: Pano, yawDeg: number, hfovDeg: number): L.LatLngExpression[] {
  const halfArc = clamp(hfovDeg / 2, 8, 85);
  const coords: L.LatLngExpression[] = [[pano.lat, pano.lon]];
  for (let offset = -halfArc; offset <= halfArc; offset += 5) {
    const [lon, lat] = destination(pano.lon, pano.lat, yawDeg + offset, VIEW_SHED_RADIUS_M);
    coords.push([lat, lon]);
  }
  const [lon, lat] = destination(pano.lon, pano.lat, yawDeg + halfArc, VIEW_SHED_RADIUS_M);
  coords.push([lat, lon], [pano.lat, pano.lon]);
  return coords;
}

function imageUrl(pano: Pano): string {
  return `${API_BASE}/data/panoramas/${pano.imagePath.split("/").map(encodeURIComponent).join("/")}`;
}

function metadataUrl(path: string): string {
  return `${API_BASE}/metadata/${path}`;
}

function compactId(value: string): string {
  const parts = value.split("/");
  return parts.at(-1) ?? value;
}

function orderPanos(panos: Pano[]): Pano[] {
  return [...panos].sort((a, b) => {
    const session = a.sessionId.localeCompare(b.sessionId);
    return session || a.orderInSession - b.orderInSession || a.id.localeCompare(b.id);
  });
}

type SessionRoute = { sessionId: string; panos: Pano[] };
type EndpointSide = "first" | "last";
type PanoEdge = { from: Pano; to: Pano; kind: "session" | "connector" };

function sessionEndpoint(panos: Pano[], side: EndpointSide): Pano {
  return side === "first" ? panos[0] : panos[panos.length - 1];
}

function oppositeSide(side: EndpointSide): EndpointSide {
  return side === "first" ? "last" : "first";
}

function sessionRoutes(panos: Pano[]): SessionRoute[] {
  const sessions = new Map<string, Pano[]>();
  for (const pano of panos) {
    sessions.set(pano.sessionId, [...(sessions.get(pano.sessionId) ?? []), pano]);
  }
  return [...sessions.entries()].map(([sessionId, panos]) => ({ sessionId, panos }));
}

function sessionBridgeEdges(sessions: SessionRoute[]): PanoEdge[] {
  if (sessions.length <= 1) return [];
  const sides: EndpointSide[] = ["first", "last"];
  let bestPair:
    | {
        a: SessionRoute;
        b: SessionRoute;
        aSide: EndpointSide;
        bSide: EndpointSide;
        distance: number;
      }
    | null = null;

  for (let i = 0; i < sessions.length; i += 1) {
    for (let j = i + 1; j < sessions.length; j += 1) {
      for (const aSide of sides) {
        for (const bSide of sides) {
          const distance = distanceM(sessionEndpoint(sessions[i].panos, aSide), sessionEndpoint(sessions[j].panos, bSide));
          if (!bestPair || distance < bestPair.distance) {
            bestPair = { a: sessions[i], b: sessions[j], aSide, bSide, distance };
          }
        }
      }
    }
  }

  if (!bestPair) return [];

  const used = new Set([bestPair.a.sessionId, bestPair.b.sessionId]);
  const bridges: PanoEdge[] = [
    {
      from: sessionEndpoint(bestPair.a.panos, bestPair.aSide),
      to: sessionEndpoint(bestPair.b.panos, bestPair.bSide),
      kind: "connector",
    },
  ];
  let left = { session: bestPair.a, side: oppositeSide(bestPair.aSide) };
  let right = { session: bestPair.b, side: oppositeSide(bestPair.bSide) };

  while (used.size < sessions.length) {
    let bestAttach:
      | {
          session: SessionRoute;
          side: EndpointSide;
          at: "left" | "right";
          distance: number;
        }
      | null = null;

    for (const session of sessions) {
      if (used.has(session.sessionId)) continue;
      for (const side of sides) {
        const endpoint = sessionEndpoint(session.panos, side);
        const leftDistance = distanceM(endpoint, sessionEndpoint(left.session.panos, left.side));
        if (!bestAttach || leftDistance < bestAttach.distance) {
          bestAttach = { session, side, at: "left", distance: leftDistance };
        }
        const rightDistance = distanceM(sessionEndpoint(right.session.panos, right.side), endpoint);
        if (!bestAttach || rightDistance < bestAttach.distance) {
          bestAttach = { session, side, at: "right", distance: rightDistance };
        }
      }
    }

    if (!bestAttach) break;
    used.add(bestAttach.session.sessionId);
    if (bestAttach.at === "left") {
      bridges.push({
        from: sessionEndpoint(bestAttach.session.panos, bestAttach.side),
        to: sessionEndpoint(left.session.panos, left.side),
        kind: "connector",
      });
      left = { session: bestAttach.session, side: oppositeSide(bestAttach.side) };
    } else {
      bridges.push({
        from: sessionEndpoint(right.session.panos, right.side),
        to: sessionEndpoint(bestAttach.session.panos, bestAttach.side),
        kind: "connector",
      });
      right = { session: bestAttach.session, side: oppositeSide(bestAttach.side) };
    }
  }

  return bridges;
}

function addNeighbor(neighbors: Map<string, Pano[]>, a: Pano, b: Pano) {
  if (!neighbors.get(a.id)?.some((pano) => pano.id === b.id)) neighbors.get(a.id)?.push(b);
  if (!neighbors.get(b.id)?.some((pano) => pano.id === a.id)) neighbors.get(b.id)?.push(a);
}

function edgeKey(edge: PanoEdge): string {
  return [edge.from.id, edge.to.id].sort().join("|");
}

function edgeDistance(edge: PanoEdge): number {
  return distanceM(edge.from, edge.to);
}

function graphComponents(panos: Pano[], edges: PanoEdge[]): Array<{ panos: Pano[]; endpoints: Pano[] }> {
  const panoById = new Map(panos.map((pano) => [pano.id, pano]));
  const neighbors = new Map(panos.map((pano) => [pano.id, [] as string[]]));
  for (const edge of edges) {
    neighbors.get(edge.from.id)?.push(edge.to.id);
    neighbors.get(edge.to.id)?.push(edge.from.id);
  }

  const seen = new Set<string>();
  const components: Array<{ panos: Pano[]; endpoints: Pano[] }> = [];
  for (const start of panos) {
    if (seen.has(start.id)) continue;
    const stack = [start.id];
    const component: Pano[] = [];
    seen.add(start.id);

    while (stack.length) {
      const id = stack.pop();
      if (!id) continue;
      const pano = panoById.get(id);
      if (!pano) continue;
      component.push(pano);
      for (const neighborId of neighbors.get(id) ?? []) {
        if (!seen.has(neighborId)) {
          seen.add(neighborId);
          stack.push(neighborId);
        }
      }
    }

    components.push({
      panos: component,
      endpoints: component.filter((pano) => (neighbors.get(pano.id)?.length ?? 0) < 2),
    });
  }
  return components;
}

function reconnectComponents(panos: Pano[], edges: PanoEdge[]): PanoEdge[] {
  const connected = [...edges];
  let components = graphComponents(panos, connected);
  while (components.length > 1) {
    let best: { from: Pano; to: Pano; distance: number } | null = null;
    for (let i = 0; i < components.length; i += 1) {
      for (let j = i + 1; j < components.length; j += 1) {
        for (const from of components[i].endpoints) {
          for (const to of components[j].endpoints) {
            const distance = distanceM(from, to);
            if (!best || distance < best.distance) best = { from, to, distance };
          }
        }
      }
    }
    if (!best) break;
    connected.push({ from: best.from, to: best.to, kind: "connector" });
    components = graphComponents(panos, connected);
  }
  return connected;
}

function buildPanoEdges(panos: Pano[]): PanoEdge[] {
  const sessions = sessionRoutes(panos);
  const sessionEdges: PanoEdge[] = [];
  for (const session of sessions) {
    for (let i = 0; i < session.panos.length - 1; i += 1) {
      sessionEdges.push({ from: session.panos[i], to: session.panos[i + 1], kind: "session" });
    }
  }
  const rewiredKeys = new Set(
    [...sessionEdges]
      .sort((a, b) => edgeDistance(b) - edgeDistance(a))
      .slice(0, LONG_SESSION_EDGES_TO_REWIRE)
      .map(edgeKey),
  );
  const edges = [
    ...sessionEdges.filter((edge) => !rewiredKeys.has(edgeKey(edge))),
    ...sessionBridgeEdges(sessions),
  ];
  return reconnectComponents(panos, edges);
}

function buildPanoNeighbors(panos: Pano[]): Map<string, Pano[]> {
  const neighbors = new Map(panos.map((pano) => [pano.id, [] as Pano[]]));
  for (const edge of buildPanoEdges(panos)) {
    addNeighbor(neighbors, edge.from, edge.to);
  }
  return neighbors;
}

function featureCollection(input: GeoJSON.FeatureCollection | GeoJSON.Feature): GeoJSON.FeatureCollection {
  if (input.type === "FeatureCollection") return input;
  return { type: "FeatureCollection", features: [input] };
}

function PanoramaViewer({
  pano,
  yaw,
  onYawChange,
  onHfovChange,
}: {
  pano: Pano;
  yaw: number;
  onYawChange: (yaw: number) => void;
  onHfovChange: (hfov: number) => void;
}) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<PannellumViewer | null>(null);
  const onYawRef = useRef(onYawChange);
  const onHfovRef = useRef(onHfovChange);
  const [imageState, setImageState] = useState<"loading" | "ready" | "missing">("loading");
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const frameRef = useRef<number | null>(null);
  const lastFrameTsRef = useRef<number | null>(null);
  onYawRef.current = onYawChange;
  onHfovRef.current = onHfovChange;

  const publishView = useCallback((viewer: PannellumViewer) => {
    const nextYaw = viewer.getYaw?.();
    const nextHfov = viewer.getHfov?.();
    if (typeof nextYaw === "number") onYawRef.current(normalizeDeg(nextYaw));
    if (typeof nextHfov === "number") onHfovRef.current(nextHfov);
  }, []);

  const stopLookLoop = useCallback(() => {
    if (frameRef.current != null) window.cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    lastFrameTsRef.current = null;
    pressedKeysRef.current.clear();
  }, []);

  const startLookLoop = useCallback(() => {
    if (frameRef.current != null) return;
    const tick = (ts: number) => {
      const viewer = viewerRef.current;
      const keys = pressedKeysRef.current;
      if (!viewer || keys.size === 0) {
        stopLookLoop();
        return;
      }

      const dt = Math.min((ts - (lastFrameTsRef.current ?? ts)) / 1000, 0.05);
      lastFrameTsRef.current = ts;
      const multiplier = keys.has("shift") ? LOOK_SHIFT_MULTIPLIER : 1;
      const yawDir =
        (keys.has("arrowright") || keys.has("d") ? 1 : 0) -
        (keys.has("arrowleft") || keys.has("a") ? 1 : 0);
      const pitchDir =
        (keys.has("arrowup") || keys.has("w") ? 1 : 0) -
        (keys.has("arrowdown") || keys.has("s") ? 1 : 0);

      if (yawDir && viewer.setYaw) {
        const nextYaw = normalizeDeg((viewer.getYaw?.() ?? 0) + yawDir * LOOK_YAW_DEG_PER_SECOND * multiplier * dt);
        viewer.setYaw(nextYaw, false);
        onYawRef.current(nextYaw);
        const nextHfov = viewer.getHfov?.();
        if (typeof nextHfov === "number") onHfovRef.current(nextHfov);
      }
      if (pitchDir && viewer.setPitch) {
        const nextPitch = clamp((viewer.getPitch?.() ?? 0) + pitchDir * LOOK_PITCH_DEG_PER_SECOND * multiplier * dt, -90, 90);
        viewer.setPitch(nextPitch, false);
      }

      frameRef.current = window.requestAnimationFrame(tick);
    };
    frameRef.current = window.requestAnimationFrame(tick);
  }, [stopLookLoop]);

  useEffect(() => {
    setImageState("loading");
    const img = new Image();
    img.onload = () => setImageState("ready");
    img.onerror = () => setImageState("missing");
    img.src = imageUrl(pano);
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [pano]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (!LOOK_KEYS.has(key) && key !== "shift") return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest("input, textarea, select")) return;
      event.preventDefault();
      event.stopPropagation();
      pressedKeysRef.current.add(key);
      startLookLoop();
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      pressedKeysRef.current.delete(key);
      if (pressedKeysRef.current.size === 0) stopLookLoop();
    };

    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", stopLookLoop);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", stopLookLoop);
      stopLookLoop();
    };
  }, [startLookLoop, stopLookLoop]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || !window.pannellum || imageState !== "ready") return;

    const viewer = window.pannellum.viewer(element, {
      type: "equirectangular",
      panorama: imageUrl(pano),
      autoLoad: true,
      yaw,
      pitch: 0,
      hfov: DEFAULT_HFOV_DEG,
      showZoomCtrl: false,
    });
    viewerRef.current = viewer;
    publishView(viewer);

    const timer = window.setInterval(() => {
      if (viewerRef.current) publishView(viewerRef.current);
    }, 200);

    return () => {
      window.clearInterval(timer);
      viewerRef.current?.destroy?.();
      viewerRef.current = null;
    };
  }, [imageState, pano, publishView, yaw]);

  if (!window.pannellum) {
    return <div className="empty-state">Pannellum did not load.</div>;
  }
  if (imageState === "missing") {
    return <div className="empty-state">Missing panorama image: {pano.imagePath}</div>;
  }
  if (imageState === "loading") {
    return <div className="empty-state">Loading panorama.</div>;
  }

  return <div ref={elementRef} className="pano-viewer" />;
}

function MapPanel({
  data,
  activePano,
  activePole,
  yawDeg,
  hfovDeg,
  onPanoSelect,
}: {
  data: LoadedData;
  activePano: Pano | null;
  activePole: PoleFeature | null;
  yawDeg: number;
  hfovDeg: number;
  onPanoSelect: (id: string) => void;
}) {
  const mapElementRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const viewshedLayerRef = useRef<L.LayerGroup | null>(null);
  const fitOnceRef = useRef(false);
  const [mapType, setMapType] = useState<MapTypeId>("dark");
  const [mapReady, setMapReady] = useState(false);
  const onPanoSelectRef = useRef(onPanoSelect);
  const panoEdges = useMemo(() => buildPanoEdges(data.manifest.panoramas), [data.manifest.panoramas]);
  onPanoSelectRef.current = onPanoSelect;

  useEffect(() => {
    if (!mapElementRef.current || mapRef.current) return;

    const map = L.map(mapElementRef.current, { zoomControl: false });
    map.createPane("viewshed");
    const viewshedPane = map.getPane("viewshed");
    if (viewshedPane) viewshedPane.style.zIndex = "350";
    L.control.zoom({ position: "bottomright" }).addTo(map);
    mapRef.current = map;
    setMapReady(true);

    return () => {
      map.remove();
      mapRef.current = null;
      setMapReady(false);
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    tileLayerRef.current?.remove();
    const selected = MAP_TYPES.find((type) => type.id === mapType) ?? MAP_TYPES[0];
    tileLayerRef.current = L.tileLayer(selected.url, {
      attribution: selected.attribution,
      maxZoom: selected.maxZoom,
    }).addTo(map);
  }, [mapReady, mapType]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    layerRef.current?.remove();

    const layers = L.layerGroup().addTo(map);
    layerRef.current = layers;

    if (data.area.features.length) {
      L.geoJSON(data.area, {
        style: {
          color: "#0f766e",
          weight: 2,
          opacity: 0.9,
          fillColor: "#14b8a6",
          fillOpacity: 0.08,
        },
      }).addTo(layers);
    }

    for (const edge of panoEdges) {
      const isActiveEdge = activePano != null && (edge.from.id === activePano.id || edge.to.id === activePano.id);
      L.polyline(
        [
          [edge.from.lat, edge.from.lon],
          [edge.to.lat, edge.to.lon],
        ],
        {
          color: isActiveEdge ? ACTIVE_PANO_COLOR : "#cbd5e1",
          dashArray: edge.kind === "connector" && !isActiveEdge ? "4 5" : undefined,
          opacity: isActiveEdge ? 0.95 : edge.kind === "connector" ? 0.7 : 0.45,
          weight: isActiveEdge ? 3 : edge.kind === "connector" ? 2 : 1.4,
        },
      ).addTo(layers);
    }

    for (const pole of data.poles.features) {
      const isActive = pole.properties.track_id === activePole?.properties.track_id;
      const [lon, lat] = pole.geometry.coordinates;
      L.circleMarker([lat, lon], {
        radius: isActive ? 11 : 6,
        color: isActive ? "#ffffff" : POLE_MARKER_COLOR,
        weight: isActive ? 3 : 2,
        fillColor: isActive ? SELECTED_POLE_MARKER_COLOR : "#f8fafc",
        fillOpacity: isActive ? 0.95 : 0.78,
      })
        .bindTooltip(pole.properties.pole_id)
        .addTo(layers);
    }

    for (const pano of data.manifest.panoramas) {
      const isActive = pano.id === activePano?.id;
      L.circleMarker([pano.lat, pano.lon], {
        radius: isActive ? 7 : 4,
        color: isActive ? ACTIVE_PANO_COLOR : "#0f172a",
        weight: isActive ? 3 : 1,
        fillColor: isActive ? "#e0f2fe" : "#ffffff",
        fillOpacity: isActive ? 1 : 0.82,
      })
        .bindTooltip(compactId(pano.id))
        .on("click", () => onPanoSelectRef.current(pano.id))
        .addTo(layers);
    }

    if (!fitOnceRef.current) {
      const bounds = L.latLngBounds([]);
      for (const pano of data.manifest.panoramas) bounds.extend([pano.lat, pano.lon]);
      for (const pole of data.poles.features) {
        const [lon, lat] = pole.geometry.coordinates;
        bounds.extend([lat, lon]);
      }
      if (bounds.isValid()) map.fitBounds(bounds.pad(0.18), { maxZoom: 18 });
      fitOnceRef.current = true;
    }
  }, [activePano, activePole, data, mapReady, panoEdges]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    viewshedLayerRef.current?.remove();
    if (!activePano) return;

    const layers = L.layerGroup().addTo(map);
    viewshedLayerRef.current = layers;
    const origin: L.LatLngExpression = [activePano.lat, activePano.lon];
    const center = destination(activePano.lon, activePano.lat, yawDeg, VIEW_SHED_RADIUS_M);
    const halfArc = clamp(hfovDeg / 2, 8, 85);
    const left = destination(activePano.lon, activePano.lat, yawDeg - halfArc, VIEW_SHED_RADIUS_M);
    const right = destination(activePano.lon, activePano.lat, yawDeg + halfArc, VIEW_SHED_RADIUS_M);

    L.polygon(viewshedCoords(activePano, yawDeg, hfovDeg), {
      pane: "viewshed",
      stroke: false,
      fillColor: ACTIVE_PANO_COLOR,
      fillOpacity: 0.18,
    }).addTo(layers);
    L.polyline([origin, [left[1], left[0]]], {
      pane: "viewshed",
      color: ACTIVE_PANO_COLOR,
      weight: 2,
      opacity: 0.85,
    }).addTo(layers);
    L.polyline([origin, [right[1], right[0]]], {
      pane: "viewshed",
      color: ACTIVE_PANO_COLOR,
      weight: 2,
      opacity: 0.85,
    }).addTo(layers);
    L.polyline([origin, [center[1], center[0]]], {
      pane: "viewshed",
      color: "#ffffff",
      weight: 3,
      opacity: 0.9,
    }).addTo(layers);
  }, [activePano, hfovDeg, mapReady, yawDeg]);

  return (
    <>
      <div ref={mapElementRef} className="map" />
      <div className="map-type-control" role="group" aria-label="Map type">
        {MAP_TYPES.map((type) => (
          <button
            key={type.id}
            type="button"
            className={mapType === type.id ? "active" : ""}
            onClick={() => setMapType(type.id)}
          >
            {type.label}
          </button>
        ))}
      </div>
    </>
  );
}

export function App() {
  const [data, setData] = useState<LoadedData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePanoId, setActivePanoId] = useState<string | null>(null);
  const [activePole, setActivePole] = useState<PoleFeature | null>(null);
  const [viewerYaw, setViewerYaw] = useState(0);
  const [viewerHfov, setViewerHfov] = useState(DEFAULT_HFOV_DEG);
  const [initialYaw, setInitialYaw] = useState(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(metadataUrl("panoramas.json")).then((r) => r.json() as Promise<PanoManifest>),
      fetch(metadataUrl("poles.geojson")).then((r) => r.json() as Promise<PoleFeatureCollection>),
      fetch(metadataUrl("bhelupur-tiny.geojson")).then((r) => r.json() as Promise<GeoJSON.FeatureCollection | GeoJSON.Feature>),
    ])
      .then(([manifest, poles, area]) => {
        if (cancelled) return;
        const orderedManifest = { ...manifest, panoramas: orderPanos(manifest.panoramas) };
        setData({ manifest: orderedManifest, poles, area: featureCollection(area) });
        setActivePanoId(orderedManifest.panoramas[0]?.id ?? null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load metadata.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const panos = data?.manifest.panoramas ?? [];
  const poles = data?.poles.features ?? [];
  const panoById = useMemo(() => new Map(panos.map((pano) => [pano.id, pano])), [panos]);
  const panoNeighbors = useMemo(() => buildPanoNeighbors(panos), [panos]);
  const activePano = activePanoId ? panoById.get(activePanoId) ?? null : null;

  const worldYaw = activePano ? normalizeDeg(viewerYaw + activePano.headingDeg) : 0;
  const stepForBearing = useCallback(
    (bearing: number): Pano | null => {
      if (!activePano) return null;
      return (panoNeighbors.get(activePano.id) ?? [])
        .map((pano) => ({
          pano,
          angleDeg: angleDiffDeg(bearing, bearingDeg(activePano, pano)),
          distM: distanceM(activePano, pano),
        }))
        .filter((candidate) => candidate.angleDeg <= STEP_ALIGNMENT_MAX_DEG)
        .sort((a, b) => a.angleDeg - b.angleDeg || a.distM - b.distM)[0]?.pano ?? null;
    },
    [activePano, panoNeighbors],
  );
  const forwardStep = useMemo(() => stepForBearing(worldYaw), [stepForBearing, worldYaw]);
  const backwardStep = useMemo(() => stepForBearing(normalizeDeg(worldYaw + 180)), [stepForBearing, worldYaw]);

  const navigateTo = useCallback(
    (target: Pano | null) => {
      if (!target || !activePano) return;
      const nextYaw = normalizeDeg(worldYaw - target.headingDeg);
      setInitialYaw(nextYaw);
      setViewerYaw(nextYaw);
      setActivePanoId(target.id);
    },
    [activePano, worldYaw],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat) return;
      if (event.key !== " " && event.code !== "Space") return;
      event.preventDefault();
      navigateTo(event.shiftKey ? backwardStep : forwardStep);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [backwardStep, forwardStep, navigateTo]);

  if (error) {
    return <div className="empty-state">{error}</div>;
  }

  if (!data || !activePano) {
    return <div className="empty-state">Loading Bhelupur.</div>;
  }

  return (
    <main className="app">
      <section className="map-pane" aria-label="Bhelupur map">
        <MapPanel
          data={data}
          activePano={activePano}
          activePole={activePole}
          yawDeg={worldYaw}
          hfovDeg={viewerHfov}
          onPanoSelect={(id) => {
            const pano = panoById.get(id);
            if (!pano) return;
            setInitialYaw(0);
            setViewerYaw(0);
            setActivePanoId(id);
          }}
        />
        <div className="map-summary">
          <strong>Bhelupur</strong>
          <span>{panos.length} panos</span>
          <span>{data.poles.features.length} poles</span>
        </div>
      </section>

      <section className="viewer-pane" aria-label="Panorama viewer">
        <header className="viewer-header">
          <div>
            <p className="eyebrow">Panorama</p>
            <h1>{compactId(activePano.id)}</h1>
          </div>
          <div className="viewer-meta">
            <span>{activePano.sessionId}</span>
            <span>#{activePano.orderInSession}</span>
          </div>
        </header>

        <div className="viewer-shell">
          <PanoramaViewer
            key={activePano.id}
            pano={activePano}
            yaw={initialYaw}
            onYawChange={setViewerYaw}
            onHfovChange={setViewerHfov}
          />
          <div className="step-controls" aria-label="Panorama navigation">
            <button type="button" onClick={() => navigateTo(backwardStep)} disabled={!backwardStep} title="Step backward">
              ↓
            </button>
            <button type="button" onClick={() => navigateTo(forwardStep)} disabled={!forwardStep} title="Step forward">
              ↑
            </button>
          </div>
        </div>

        <footer className="detail-row">
          <label className="pole-picker">
            <span className="eyebrow">Pole</span>
            <select
              value={activePole?.properties.track_id ?? ""}
              onChange={(event) => {
                const trackId = event.target.value;
                setActivePole(poles.find((pole) => pole.properties.track_id === trackId) ?? null);
              }}
            >
              <option value="">None selected</option>
              {poles.map((pole) => (
                <option key={pole.properties.track_id} value={pole.properties.track_id}>
                  {pole.properties.pole_id}
                </option>
              ))}
            </select>
          </label>
          <div>
            <p className="eyebrow">Selected</p>
            <strong>{activePole?.properties.pole_id ?? "None selected"}</strong>
          </div>
          {activePole && (
            <>
              <div>
                <p className="eyebrow">Quality</p>
                <span>{activePole.properties.quality ?? "n/a"}</span>
              </div>
              <div>
                <p className="eyebrow">Sightings</p>
                <span>{activePole.properties.n_sightings ?? "n/a"}</span>
              </div>
            </>
          )}
        </footer>
      </section>
    </main>
  );
}

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from ..core.scenario import Scenario, ZonePolygon
from . import zone_definitions as zd


# Wider bbox covering Valle de Aburrá so all 6 CVRPTW zones are populated.
PLACE = "Medellin, Colombia"
BBOX = dict(north=6.435, south=6.030, east=-75.330, west=-75.680)
ORIGIN_LONLAT = (-75.58683269546758, 6.243283526198216)  # UPB
# TODO: Restrain further to only metro area polygon to avoid nearby towns or mountain corregimientos
DEFAULT_BUS_CAPACITY = 20

CACHE_DIR = Path("cache")
GRAPH_CACHE = CACHE_DIR / "medellin_valle_drive.graphml"
GEOJSON_DIR = Path(__file__).parent / "geojson"


def _load_graph() -> nx.MultiDiGraph:
    """Download once, then reuse from disk. All scenarios share this graph
    so road distances are directly comparable across runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GRAPH_CACHE.exists():
        return ox.load_graphml(GRAPH_CACHE)

    print(f"Downloading {PLACE} road network … (one-time, cached at {GRAPH_CACHE})")
    G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)

    # Crop to bbox
    nodes_in_bbox = [
        n for n in G.nodes
        if BBOX["west"] <= G.nodes[n]["x"] <= BBOX["east"]
        and BBOX["south"] <= G.nodes[n]["y"] <= BBOX["north"]
    ]
    G = G.subgraph(nodes_in_bbox).copy()

    # Largest strongly-connected component for full reachability
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    # Pre-compute travel_time edge attribute for CVRPTW
    G = ox.add_edge_speeds(G, fallback=zd.SPEED_FALLBACK_KPH)
    G = ox.add_edge_travel_times(G)

    ox.save_graphml(G, GRAPH_CACHE)
    print(f"  Graph cached: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _load_zones() -> list[ZonePolygon]:
    """Build the 6 CVRPTW routing zones from naomi's GeoJSON files."""
    commune_geoms: dict[str, object] = {}
    with open(GEOJSON_DIR / "comunas_medellin.geojson", encoding="utf-8") as f:
        for feat in json.load(f)["features"]:
            name = str(feat["properties"].get("name", "")).strip()
            if name:
                commune_geoms[name] = shape(feat["geometry"])

    muni_geoms: dict[str, object] = {}
    with open(GEOJSON_DIR / "valle_aburra_urbano.geojson", encoding="utf-8") as f:
        for feat in json.load(f)["features"]:
            p = feat["properties"]
            if p.get("municipio") != "Medellín":
                muni_geoms[p["municipio"]] = shape(feat["geometry"])

    zones: list[ZonePolygon] = []
    for zdef in zd.ZONE_DEFINITIONS:
        polys = []
        for member in zdef["members"]:
            if member in commune_geoms:
                polys.append(commune_geoms[member])
            elif member in muni_geoms:
                polys.append(muni_geoms[member])
        if not polys:
            continue
        merged = unary_union(polys)
        zones.append(ZonePolygon(
            name=zdef["zone_name"],
            color=zdef["color"],
            members=zdef["members"],
            geometry=merged,
            centroid=(merged.centroid.x, merged.centroid.y),
        ))
    return zones


def _tag_node_zones(node_xy: list[tuple[float, float]],
                    zones: list[ZonePolygon]) -> np.ndarray:
    """Return zone-name-per-row using STRtree with centroid fallback."""
    geoms = [z.geometry for z in zones]
    tree = STRtree(geoms)
    centroids = np.array([z.centroid for z in zones])  # (k, 2)
    out = np.empty(len(node_xy), dtype=object)
    for i, (lon, lat) in enumerate(node_xy):
        pt = Point(lon, lat)
        hits = tree.query(pt, predicate="covers")
        if len(hits):
            out[i] = zones[int(hits[0])].name
        else:
            d = (centroids[:, 0] - lon) ** 2 + (centroids[:, 1] - lat) ** 2
            out[i] = zones[int(np.argmin(d))].name
    return out


_GRAPH_SINGLETON: nx.MultiDiGraph | None = None
_ZONES_SINGLETON: list[ZonePolygon] | None = None
_METRO_POLY_SINGLETON = None
_METRO_NODES_SINGLETON: list | None = None


def _shared_graph() -> nx.MultiDiGraph:
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is None:
        _GRAPH_SINGLETON = _load_graph()
    return _GRAPH_SINGLETON


def _shared_zones() -> list[ZonePolygon]:
    global _ZONES_SINGLETON
    if _ZONES_SINGLETON is None:
        _ZONES_SINGLETON = _load_zones()
    return _ZONES_SINGLETON


def _shared_metro_polygon():
    """Union of all CVRPTW zone polygons = Valle de Aburrá urban footprint."""
    global _METRO_POLY_SINGLETON
    if _METRO_POLY_SINGLETON is None:
        _METRO_POLY_SINGLETON = unary_union([z.geometry for z in _shared_zones()])
    return _METRO_POLY_SINGLETON


def _shared_metro_nodes() -> list:
    """Graph nodes whose (lon, lat) falls inside the Valle de Aburrá polygon.
    Children are sampled from this set so no student lands in mountain
    corregimientos or outside the metro area."""
    global _METRO_NODES_SINGLETON
    if _METRO_NODES_SINGLETON is None:
        G = _shared_graph()
        prepared = prep(_shared_metro_polygon())
        _METRO_NODES_SINGLETON = [
            n for n in G.nodes
            if prepared.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"]))
        ]
        if len(_METRO_NODES_SINGLETON) < 1000:
            raise RuntimeError(
                f"Only {len(_METRO_NODES_SINGLETON)} graph nodes inside Valle "
                "de Aburrá polygon — check GeoJSON loading."
            )
        print(f"  Metro polygon contains {len(_METRO_NODES_SINGLETON)} of "
              f"{G.number_of_nodes()} graph nodes")
    return _METRO_NODES_SINGLETON


def make_scenario(n_children: int,
                  seed: int,
                  bus_capacity: int = DEFAULT_BUS_CAPACITY) -> Scenario:
    """Build a Scenario with `n_children` randomly-placed students.

    The road network is shared across all calls (single download, cached
    on disk). Distance and time matrices are computed via Dijkstra with
    `length` and `travel_time` edge weights.
    """
    G = _shared_graph()
    zones = _shared_zones()
    metro_nodes = _shared_metro_nodes()
    rng = np.random.default_rng(seed)

    all_nodes = list(G.nodes)
    children = list(rng.choice(metro_nodes, size=n_children, replace=False))

    # Snap depot to the closest graph node
    x_all = np.array([G.nodes[n]["x"] for n in all_nodes])
    y_all = np.array([G.nodes[n]["y"] for n in all_nodes])
    d2 = (x_all - ORIGIN_LONLAT[0]) ** 2 + (y_all - ORIGIN_LONLAT[1]) ** 2
    origin_node = all_nodes[int(np.argmin(d2))]

    ni = list(children) + [origin_node]
    xi = np.array([G.nodes[n]["x"] for n in ni])
    yi = np.array([G.nodes[n]["y"] for n in ni])
    n_pts = len(ni)

    # Pairwise shortest path distance + travel time
    dist_m = np.full((n_pts, n_pts), np.inf)
    time_s = np.full((n_pts, n_pts), np.inf)
    for i, src in enumerate(ni):
        d_lengths = nx.single_source_dijkstra_path_length(G, src, weight="length")
        t_lengths = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
        for j, dst in enumerate(ni):
            if dst in d_lengths:
                dist_m[i, j] = float(d_lengths[dst])
            if dst in t_lengths:
                time_s[i, j] = float(t_lengths[dst])

    # Replace inf with a large finite value (paths should exist on largest SCC)
    finite_max = np.nanmax(dist_m[np.isfinite(dist_m)]) if np.any(np.isfinite(dist_m)) else 1.0
    dist_m = np.where(np.isfinite(dist_m), dist_m, finite_max * 10)
    time_finite_max = np.nanmax(time_s[np.isfinite(time_s)]) if np.any(np.isfinite(time_s)) else 1.0
    time_s = np.where(np.isfinite(time_s), time_s, time_finite_max * 10)

    node_zones = _tag_node_zones(list(zip(xi.tolist(), yi.tolist())), zones)

    return Scenario(
        node_ids=np.array(ni, dtype=np.int64),
        x=xi,
        y=yi,
        dist_matrix=dist_m,
        time_matrix=time_s,
        origin_index=n_pts - 1,
        bus_capacity=bus_capacity,
        graph=G,
        n_children=n_children,
        seed=seed,
        zones=zones,
        node_zones=node_zones,
    )

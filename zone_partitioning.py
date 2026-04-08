"""
zone_partitioning.py
====================
Stage 2b of the school bus routing pipeline.

Splits the sampled child nodes from scenario_data.npz into geographic zones
covering the full Valle de Aburrá (Medellín comunas + surrounding municipalities),
then runs capacitated k-medoids clustering and OR-Tools TSP independently for
each zone.  Results are saved per zone and as a combined summary.

Run order:
    python src/data_generation.py      # produces scenario_data.npz
    python src/zone_partitioning.py    # produces zone_*_result.npz + summary
    # (clustering.py and tsp_solver.py are now called internally per zone)
"""

import json
import math
import numpy as np
import networkx as nx
import kmedoids
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from pathlib import Path
from shapely.geometry import Point, shape
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# ---------------------------------------------------------------------------
# 0.  Zone definitions — Valle de Aburrá
#     Each zone is a dict with a display name and a list of polygon dicts.
#     Polygons come from two sources:
#       - Medellín comunas: loaded from the GeoJSON file (exact boundaries)
#       - Surrounding municipalities: defined inline as bbox rectangles
#     The COMUNAS_GEOJSON path can be overridden via the constant below.
# ---------------------------------------------------------------------------

# Un solo GeoJSON con todo: 16 comunas de Medellín + municipios del valle
VALLE_GEOJSON = "valle_aburra_urbano.geojson"   # comunas + municipios (polígonos reales)
SCENARIO_NPZ  = "scenario_data.npz"
OUTPUT_DIR    = Path(".")                        # donde se guardan los zone_*.npz

# Zone definitions: each entry maps a zone name to the set of area names it
# contains.  Names must match either:
#   - a "name" field in comunas_medellin.geojson  (substring match on commune number)
#   - a "municipio" field in valle_aburra_urbano.geojson  (exact match)
# Nombres EXACTOS según el campo "comuna" del GeoJSON valle_aburra_urbano.geojson
# y el campo "municipio" para los municipios del valle.
ZONE_DEFINITIONS = [
    {
        "name": "Norte",
        "areas": ["Bello"],
        "color": "#378ADD",
    },
    {
        "name": "Nororiental",
        "areas": [
            "Comuna 1 - Popular",
            "Comuna 2 - Santa Cruz",
            "Comuna 3 - Manrique",
        ],
        "color": "#639922",
    },
    {
        "name": "Noroccidental",
        "areas": [
            "Comuna 5 - Castilla",
            "Comuna 6 - Doce de Octubre",
            "Comuna 7 - Robledo",
        ],
        "color": "#7F77DD",
    },
    {
        "name": "Centro",
        "areas": [
            "Comuna 4 - Aranjuez",
            "Comuna 10 - La Candelaria",
            "Comuna 11 - Laureles-Estadio",
            "Comuna 12 - La América",
            "Comuna 13 - San Javier",
        ],
        "color": "#EF9F27",
    },
    {
        "name": "Suroriental",
        "areas": [
            "Comuna 8 - Villa Hermosa",
            "Comuna 9 - Buenos Aires",
            "Comuna 14 - El Poblado",
            "Envigado",
        ],
        "color": "#D85A30",
    },
    {
        "name": "Sur",
        "areas": [
            "Comuna 15 - Guayabal",
            "Comuna 16 - Belén",
            "Itagüí",
            "Sabaneta",
            "La Estrella",
        ],
        "color": "#1D9E75",
    },
]

# ---------------------------------------------------------------------------
# 1.  Build zone polygons
# ---------------------------------------------------------------------------

def build_zone_polygons(valle_path: str) -> list[dict]:
    """
    Carga un único GeoJSON que contiene:
      - Comunas de Medellín  (properties: municipio="Medellín", comuna="Comuna X - Nombre")
      - Municipios del valle (properties: municipio="Bello" etc.,  comuna=None)

    Retorna lista de zonas, cada una con:
        name    : str
        color   : str (hex)
        polygons: list of Shapely Polygon
        centroid: (lon, lat)
    """
    with open(valle_path, encoding="utf-8") as f:
        gj = json.load(f)

    # Separar comunas y municipios
    commune_polys = {}   # "Comuna 11" → Shapely polygon
    muni_polys    = {}   # "Bello"     → Shapely polygon

    for feat in gj["features"]:
        p    = feat["properties"]
        geom = shape(feat["geometry"])
        if p.get("comuna"):                      # es una comuna de Medellín
            commune_polys[p["comuna"]] = geom
        else:                                    # es un municipio del valle
            muni_polys[p["municipio"]] = geom

    print(f"  Cargadas {len(commune_polys)} comunas + {len(muni_polys)} municipios")

    zones = []
    for zdef in ZONE_DEFINITIONS:
        polys = []
        for area in zdef["areas"]:
            if area in muni_polys:               # municipio exacto
                polys.append(muni_polys[area])
                continue
            # Exact match first, then substring fallback
            if area in commune_polys:
                matched = [commune_polys[area]]
            else:
                matched = [p for name, p in commune_polys.items() if area in name]
            if matched:
                polys.extend(matched)
            else:
                print(f"  WARNING: '{area}' no encontrado en el GeoJSON")

        if not polys:
            raise ValueError(f"Zona '{zdef['name']}' sin polígonos — revisa ZONE_DEFINITIONS")

        cx = np.mean([p.centroid.x for p in polys])
        cy = np.mean([p.centroid.y for p in polys])
        zones.append({"name": zdef["name"], "color": zdef["color"],
                      "polygons": polys, "centroid": (cx, cy)})

    return zones


# ---------------------------------------------------------------------------
# 2.  Assign each child node to a zone
# ---------------------------------------------------------------------------

def assign_zones(x: np.ndarray, y: np.ndarray,
                 origin_idx: int, zones: list[dict]) -> np.ndarray:
    """
    Returns zone_labels array of shape (n_points,) where zone_labels[i] is the
    zone index for child node i.  The origin/depot (origin_idx) is excluded and
    gets label -1.

    Assignment logic:
      1. Point-in-polygon check against all zone polygons.
      2. If a point falls in no polygon (outside all defined areas), assign it
         to the nearest zone centroid — so no child is ever left unassigned.
    """
    n = len(x)
    zone_labels = np.full(n, -1, dtype=int)

    for i in range(n):
        if i == origin_idx:
            continue
        pt = Point(x[i], y[i])

        assigned = False
        for z_idx, zone in enumerate(zones):
            if any(poly.contains(pt) for poly in zone["polygons"]):
                zone_labels[i] = z_idx
                assigned = True
                break

        # Fallback: nearest centroid
        if not assigned:
            dists = [
                (x[i] - zone["centroid"][0])**2 + (y[i] - zone["centroid"][1])**2
                for zone in zones
            ]
            zone_labels[i] = int(np.argmin(dists))

    # Report
    for z_idx, zone in enumerate(zones):
        n_zone = int(np.sum(zone_labels == z_idx))
        print(f"  Zone {z_idx} {zone['name']:15s}: {n_zone:3d} children")

    n_fallback = 0
    for i in range(n):
        if i == origin_idx or zone_labels[i] < 0:
            continue
        pt = Point(x[i], y[i])
        if not any(poly.contains(pt) for poly in zones[zone_labels[i]]["polygons"]):
            n_fallback += 1
    if n_fallback:
        print(f"  ({n_fallback} children assigned via nearest-centroid fallback)")

    return zone_labels


# ---------------------------------------------------------------------------
# 3.  Clustering (capacitated k-medoids) — extracted from clustering.py
# ---------------------------------------------------------------------------

def _capacitated_assign(D: np.ndarray, medoids: list[int],
                         capacity: int) -> tuple[np.ndarray, float]:
    """Min-cost flow assignment of children to medoids respecting bus capacity."""
    n = D.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge("s", ("c", i), capacity=1, weight=0)
    for i in range(n):
        for m_idx, m in enumerate(medoids):
            cost = int(D[i, m] * 1000)
            G.add_edge(("c", i), ("m", m_idx), capacity=1, weight=cost)
    for m_idx in range(len(medoids)):
        G.add_edge(("m", m_idx), "t", capacity=capacity, weight=0)
    G.nodes["s"]["demand"] = -n
    G.nodes["t"]["demand"] = n

    flow_dict = nx.min_cost_flow(G)

    labels = np.empty(n, dtype=int)
    total_cost = 0.0
    for i in range(n):
        for m_idx in range(len(medoids)):
            if flow_dict[("c", i)].get(("m", m_idx), 0) == 1:
                labels[i] = m_idx
                total_cost += D[i, medoids[m_idx]]
                break
    return labels, total_cost


def _recompute_medoids(D: np.ndarray, labels: np.ndarray, k: int) -> list[int]:
    medoids = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        sub = D[np.ix_(members, members)]
        best = members[np.argmin(sub.sum(axis=1))]
        medoids.append(int(best))
    return medoids


def run_clustering(D: np.ndarray, bus_capacity: int,
                   zone_name: str) -> tuple[np.ndarray, list[int], int, float]:
    """
    Run capacitated k-medoids on distance matrix D.
    Returns (labels, medoid_indices, k, total_cost).
    labels and medoid_indices are local indices into D (0 … n_children-1).
    """
    n = D.shape[0]
    k = max(1, math.ceil(n / bus_capacity))
    print(f"    [{zone_name}] {n} children  |  capacity {bus_capacity}  |  k={k} buses")

    if n <= 1:
        return np.zeros(n, dtype=int), [0], 1, 0.0

    # Handle edge case where k >= n (very small zone)
    k = min(k, n)

    result = kmedoids.fasterpam(D, k, random_state=42)
    medoids = list(result.medoids)

    max_iter = 50
    labels, total_cost = None, 0.0
    for iteration in range(1, max_iter + 1):
        labels, total_cost = _capacitated_assign(D, medoids, bus_capacity)
        new_medoids = _recompute_medoids(D, labels, k)
        if new_medoids == medoids:
            print(f"    [{zone_name}] Converged in {iteration} iter  |  cost {total_cost:,.0f} m")
            break
        medoids = new_medoids
    else:
        print(f"    [{zone_name}] Max iter reached  |  cost {total_cost:,.0f} m")

    return labels, medoids, k, total_cost


# ---------------------------------------------------------------------------
# 4.  TSP solver — extracted from tsp_solver.py
# ---------------------------------------------------------------------------

def _solve_tsp(dist_sub: np.ndarray, depot: int) -> list[int]:
    """Solve TSP on sub-distance matrix. Returns local node indices."""
    n = dist_sub.shape[0]
    if n <= 2:
        return list(range(n))

    int_dist = (dist_sub * 1000).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(int_dist[manager.IndexToNode(from_index),
                             manager.IndexToNode(to_index)])

    cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools: no solution found")

    route, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    return route


def run_tsp_for_zone(cluster_labels: np.ndarray, cluster_medoids: list[int],
                     k: int, zone_scenario_idx: np.ndarray,
                     full_dist: np.ndarray, origin_idx: int,
                     zone_name: str) -> tuple[list[list[int]], float]:
    """
    Solve TSP for every cluster in a zone.

    Parameters
    ----------
    cluster_labels     : per-child cluster assignment (local to zone)
    cluster_medoids    : medoid indices (local to zone)
    k                  : number of clusters
    zone_scenario_idx  : array mapping zone-local child index → full scenario index
    full_dist          : full N×N distance matrix from scenario_data.npz
    origin_idx         : index of depot/school in the full matrix

    Returns
    -------
    routes      : list of k routes, each a list of full-scenario indices
    zone_dist   : total route distance for this zone (metres)
    """
    routes = []
    zone_dist = 0.0

    for c in range(k):
        child_mask = cluster_labels == c
        local_children = np.where(child_mask)[0]
        scenario_children = zone_scenario_idx[local_children]

        sub_indices = np.append(scenario_children, origin_idx)
        depot_pos = len(sub_indices) - 1
        dist_sub = full_dist[np.ix_(sub_indices, sub_indices)]

        route_local = _solve_tsp(dist_sub, depot_pos)
        route_full = [int(sub_indices[i]) for i in route_local]
        routes.append(route_full)

        route_dist = sum(
            full_dist[route_full[i], route_full[i + 1]]
            for i in range(len(route_full) - 1)
        )
        zone_dist += route_dist
        print(f"    [{zone_name}] Cluster {c:2d}: {len(scenario_children):3d} children  "
              f"|  route {route_dist:,.0f} m")

    return routes, zone_dist


# ---------------------------------------------------------------------------
# 5.  Visualisation
# ---------------------------------------------------------------------------

def plot_zone_overview(x: np.ndarray, y: np.ndarray,
                       zone_labels: np.ndarray, origin_idx: int,
                       zones: list[dict], out_path: str) -> None:
    """Scatter plot of all children coloured by zone, with basemap."""
    fig, ax = plt.subplots(figsize=(14, 14))

    for z_idx, zone in enumerate(zones):
        mask = zone_labels == z_idx
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], color=zone["color"], s=18, alpha=0.8,
                   label=f"Zone {z_idx} {zone['name']} ({mask.sum()})")

    ax.scatter(x[origin_idx], y[origin_idx],
               marker="*", color="red", s=200, zorder=6, label="School / depot")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=13, crs="EPSG:4326")
    except Exception:
        pass  # basemap is cosmetic; don't fail the pipeline if tiles are unavailable

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Zone overview map saved → {out_path}")


def plot_zone_routes(x: np.ndarray, y: np.ndarray,
                     zone_idx: int, zone: dict,
                     routes: list[list[int]], cluster_labels: np.ndarray,
                     zone_scenario_idx: np.ndarray, cluster_medoids: list[int],
                     origin_idx: int, out_path: str) -> None:
    """Per-zone route map showing TSP tours for every cluster."""
    k = len(routes)
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(k, 2))

    for c in range(k):
        color = cmap(c)
        child_mask = cluster_labels == c
        sc_idx = zone_scenario_idx[child_mask]

        ax.scatter(x[sc_idx], y[sc_idx], color=color, s=22,
                   label=f"Bus {c} ({child_mask.sum()})")

        route = routes[c]
        ax.plot([x[i] for i in route], [y[i] for i in route],
                color=color, linewidth=1.2, alpha=0.7)

        for stop_num, node_idx in enumerate(route):
            ax.annotate(str(stop_num), (x[node_idx], y[node_idx]),
                        fontsize=5, fontweight="bold", color=color,
                        ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")

        mi = cluster_medoids[c]
        ax.scatter(x[zone_scenario_idx[mi]], y[zone_scenario_idx[mi]],
                   color=color, s=120, marker="D",
                   edgecolors="black", linewidths=1.0)

    ax.scatter(x[origin_idx], y[origin_idx],
               marker="*", color="red", s=200, zorder=6, label="School")

    ax.set_title(f"Zone {zone_idx} — {zone['name']}", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=14, crs="EPSG:4326")
    except Exception:
        pass

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Route map saved → {out_path}")


# ---------------------------------------------------------------------------
# 6.  Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load scenario data
    # ------------------------------------------------------------------
    print("Loading scenario data …")
    scenario = np.load(SCENARIO_NPZ)
    full_dist  = scenario["dist_matrix"]
    x          = scenario["x"]
    y          = scenario["y"]
    origin_idx = int(scenario["origin_index"])
    bus_cap    = int(scenario["bus_capacity"])
    n_children = full_dist.shape[0] - 1

    print(f"  {n_children} children  |  bus capacity {bus_cap}  |  depot index {origin_idx}")

    # ------------------------------------------------------------------
    # Build zone polygons
    # ------------------------------------------------------------------
    print("\nBuilding zone polygons …")
    zones = build_zone_polygons(VALLE_GEOJSON)
    print(f"  {len(zones)} zones defined")

    # ------------------------------------------------------------------
    # Assign children to zones
    # ------------------------------------------------------------------
    print("\nAssigning children to zones …")
    zone_labels = assign_zones(x, y, origin_idx, zones)

    # ------------------------------------------------------------------
    # Plot zone overview
    # ------------------------------------------------------------------
    plot_zone_overview(x, y, zone_labels, origin_idx, zones,
                       str(OUTPUT_DIR / "zones_overview_map.png"))

    # ------------------------------------------------------------------
    # Per-zone: clustering + TSP
    # ------------------------------------------------------------------
    all_routes       = {}   # zone_idx → list of routes
    zone_distances   = {}   # zone_idx → total distance (m)
    zone_cluster_info = {}  # zone_idx → {labels, medoids, k}

    total_distance = 0.0

    for z_idx, zone in enumerate(zones):
        print(f"\n{'='*60}")
        print(f"Zone {z_idx} — {zone['name']}")
        print(f"{'='*60}")

        # Indices of children in this zone (indices into full scenario arrays)
        child_mask = zone_labels == z_idx
        zone_scenario_idx = np.where(child_mask)[0]   # full scenario indices

        n_zone = len(zone_scenario_idx)
        if n_zone == 0:
            print(f"  No children in zone {z_idx} — skipping")
            continue

        # Build zone-local distance sub-matrix (children only, no depot)
        D = full_dist[np.ix_(zone_scenario_idx, zone_scenario_idx)]
        D = (D + D.T) / 2.0   # symmetrise

        # --- Clustering ---
        print("  Clustering …")
        labels, medoids, k, clust_cost = run_clustering(D, bus_cap, zone["name"])

        # --- TSP ---
        print("  Routing …")
        routes, zone_dist = run_tsp_for_zone(
            labels, medoids, k,
            zone_scenario_idx, full_dist, origin_idx, zone["name"]
        )

        total_distance += zone_dist
        all_routes[z_idx] = routes
        zone_distances[z_idx] = zone_dist
        zone_cluster_info[z_idx] = {
            "labels":  labels,
            "medoids": medoids,
            "k":       k,
        }

        print(f"  Zone {z_idx} total distance: {zone_dist:,.0f} m  ({k} buses)")

        # --- Save per-zone result ---
        zone_npz = OUTPUT_DIR / f"zone_{z_idx}_result.npz"
        np.savez(
            str(zone_npz),
            zone_idx=z_idx,
            zone_name=zone["name"],
            zone_scenario_idx=zone_scenario_idx,
            labels=labels,
            medoids=np.array(medoids),
            k=k,
            clustering_cost=clust_cost,
            zone_distance=zone_dist,
            **{f"route_{c}": np.array(r) for c, r in enumerate(routes)},
        )
        print(f"  Saved → {zone_npz}")

        # --- Per-zone route map ---
        plot_zone_routes(
            x, y, z_idx, zone, routes,
            labels, zone_scenario_idx, medoids, origin_idx,
            str(OUTPUT_DIR / f"zone_{z_idx}_{zone['name'].lower()}_routes.png")
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_buses = 0
    for z_idx, zone in enumerate(zones):
        if z_idx not in zone_distances:
            print(f"  Zone {z_idx} {zone['name']:15s}:  no children")
            continue
        k   = zone_cluster_info[z_idx]["k"]
        dist = zone_distances[z_idx]
        n_z  = int(np.sum(zone_labels == z_idx))
        total_buses += k
        print(f"  Zone {z_idx} {zone['name']:15s}: {n_z:3d} children  "
              f"{k:2d} buses  {dist:>10,.0f} m")

    print(f"  {'─'*52}")
    print(f"  {'TOTAL':20s}: {n_children:3d} children  "
          f"{total_buses:2d} buses  {total_distance:>10,.0f} m")

    # Save combined summary
    summary_npz = OUTPUT_DIR / "zones_summary.npz"
    np.savez(
        str(summary_npz),
        n_zones=len(zones),
        zone_names=np.array([z["name"] for z in zones]),
        zone_labels=zone_labels,
        total_distance=total_distance,
        total_buses=total_buses,
    )
    print(f"\nSummary saved → {summary_npz}")
    print("Done.")


if __name__ == "__main__":
    main()

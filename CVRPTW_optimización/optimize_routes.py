"""
optimize_routes.py  —  Stage 2: Multi-objective CVRPTW per zone.

For every zone (16 Medellín comunas + 5 municipalities):
  1. Identifies which children belong to the zone.
  2. Determines k = ⌈n_children / BUS_CAPACITY⌉ buses.
  3. Solves a Capacitated VRP with Time Windows (CVRPTW) via OR-Tools:
       PRIMARY objective   → minimise total travel time (seconds).
       SECONDARY objective → minimise route-time imbalance (SetGlobalSpanCost).
       HARD constraint     → each route ≤ MAX_ROUTE_MINUTES including boarding.
  4. Reports route distances, times, and worst-case route per zone.

Outputs:
  outputs/route_results.json  — full results consumed by visualize_results.py

Run:
    python optimize_routes.py
"""

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

import config as cfg


# =============================================================================
# 1.  Rebuild zone geometries (needed for point-in-polygon)
# =============================================================================

def load_zones() -> list[dict]:
    """
    Returns same structure as generate_data.build_zones() but geometry only
    (no road-network work needed here).
    """
    zones = []

    # Medellín comunas
    with open(cfg.COMUNAS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        p    = feat["properties"]
        ref  = str(p.get("ref", "")).strip().zfill(2)
        name = str(p.get("name", f"Comuna {ref}")).strip()
        geom = shape(feat["geometry"])
        zones.append({"name": name, "municipality": "Medellín",
                      "geometry": geom,
                      "centroid": (geom.centroid.x, geom.centroid.y)})

    zones.sort(key=lambda z: int(z["name"].split()[1])
               if z["municipality"] == "Medellín" else 99)

    # Other municipalities
    with open(cfg.VALLE_GEOJSON, encoding="utf-8") as f:
        gj2 = json.load(f)
    for feat in gj2["features"]:
        p = feat["properties"]
        if p.get("municipio") != "Medellín":
            geom = shape(feat["geometry"])
            zones.append({"name": p["municipio"],
                          "municipality": p["municipio"],
                          "geometry": geom,
                          "centroid": (geom.centroid.x, geom.centroid.y)})
    return zones


# =============================================================================
# 2.  Assign children to zones
# =============================================================================

def assign_zones(x: np.ndarray, y: np.ndarray,
                 origin_idx: int, zones: list[dict],
                 saved_zone_names: np.ndarray) -> np.ndarray:
    """
    Primary: use the zone names saved during data generation (exact match).
    Fallback: point-in-polygon, then nearest centroid.
    Returns zone_labels[i] = zone index, -1 for depot.
    """
    zone_name_to_idx = {z["name"]: i for i, z in enumerate(zones)}
    n      = len(x)
    labels = np.full(n, -1, dtype=int)

    for i in range(n - 1):   # last entry is depot
        saved = saved_zone_names[i] if i < len(saved_zone_names) else ""
        if saved in zone_name_to_idx:
            labels[i] = zone_name_to_idx[saved]
            continue
        # Fallback: point-in-polygon
        pt       = Point(x[i], y[i])
        assigned = False
        for z_idx, z in enumerate(zones):
            if z["geometry"].covers(pt):
                labels[i] = z_idx
                assigned   = True
                break
        if not assigned:
            dists      = [(x[i] - z["centroid"][0])**2 +
                          (y[i] - z["centroid"][1])**2 for z in zones]
            labels[i]  = int(np.argmin(dists))

    return labels


# =============================================================================
# 3.  OR-Tools CVRPTW
# =============================================================================

def solve_cvrptw(time_sub: np.ndarray,
                 n_children: int,
                 depot_local: int,
                 k: int,
                 zone_name: str) -> tuple[list[list[int]], list[float], list[float]]:
    """
    Solve CVRPTW for one zone.

    Returns
    -------
    routes       : k lists of LOCAL child indices in visit order
    route_times  : route duration in minutes per bus (incl. boarding)
    route_dists  : approximate route distance proxy (travel time, seconds)
    """
    n              = time_sub.shape[0]
    max_ms         = int(cfg.MAX_ROUTE_MINUTES * 60 * 1000)
    boarding_ms    = int(cfg.BOARDING_SECONDS * 1000)

    int_time = (time_sub * 1000).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, k, depot_local)
    routing = pywrapcp.RoutingModel(manager)

    # Arc-cost: pure travel time (objective 1 — minimise total travel time)
    def travel_cb(fi, ti):
        return int(int_time[manager.IndexToNode(fi), manager.IndexToNode(ti)])

    arc_cb_idx = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(arc_cb_idx)

    # Time dimension: travel + boarding at each pickup stop
    def time_with_boarding(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        b  = 0 if fn == depot_local else boarding_ms
        return int(int_time[fn, tn]) + b

    time_cb_idx = routing.RegisterTransitCallback(time_with_boarding)
    routing.AddDimension(
        time_cb_idx,
        0,        # no slack
        max_ms,   # hard upper bound on total route time
        True,     # cumulative variable starts at 0
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Objective 2: penalise imbalance between longest and shortest route
    time_dim.SetGlobalSpanCostCoefficient(cfg.SPAN_BALANCE_COEFF)

    # Capacity dimension: every child stop has demand = 1
    def demand_cb(fi):
        return 0 if manager.IndexToNode(fi) == depot_local else 1

    dem_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_cb_idx, 0, [cfg.BUS_CAPACITY] * k, True, "Capacity"
    )

    # High penalty for unserved children (forces full coverage)
    penalty = max_ms * 200
    for node in range(n):
        if node != depot_local:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = cfg.OR_TOOLS_TIME_LIMIT_SEC

    solution = routing.SolveWithParameters(params)

    if solution is None:
        print(f"    [{zone_name}] WARNING: no solution found — empty routes.")
        return [[] for _ in range(k)], [0.0] * k, [0.0] * k

    routes, route_times, route_dists = [], [], []
    for v in range(k):
        route = []
        idx   = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot_local:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

        end_ms = solution.Min(time_dim.CumulVar(routing.End(v)))
        route_times.append(round(end_ms / 60_000, 2))   # minutes
        route_dists.append(round(end_ms / 1000, 1))     # seconds (proxy)

    served = sum(len(r) for r in routes)
    if served < n_children:
        print(f"    [{zone_name}] WARNING: {n_children - served} child(ren) "
              "unserved — raise MAX_ROUTE_MINUTES or lower N_CHILDREN.")

    return routes, route_times, route_dists


# =============================================================================
# 4.  Per-zone driver
# =============================================================================

def process_zone(z_idx: int, zone: dict,
                 zone_labels: np.ndarray,
                 full_time: np.ndarray,
                 full_dist: np.ndarray,
                 x: np.ndarray, y: np.ndarray,
                 origin_idx: int) -> dict | None:

    child_mask        = zone_labels == z_idx
    zone_scenario_idx = np.where(child_mask)[0]
    n_zone            = len(zone_scenario_idx)
    if n_zone == 0:
        return None

    k = max(1, math.ceil(n_zone / cfg.BUS_CAPACITY))
    print(f"\n  {zone['name']}: {n_zone} children → {k} bus(es)")

    sub_indices = np.append(zone_scenario_idx, origin_idx)
    depot_local = len(sub_indices) - 1
    time_sub    = full_time[np.ix_(sub_indices, sub_indices)]
    dist_sub    = full_dist[np.ix_(sub_indices, sub_indices)]

    local_routes, rt_min, _ = solve_cvrptw(
        time_sub, n_zone, depot_local, k, zone["name"])

    scenario_routes  = []
    route_distances  = []
    route_times_min  = []

    for r_local, rt in zip(local_routes, rt_min):
        r_scenario = [int(zone_scenario_idx[i]) for i in r_local]
        scenario_routes.append(r_scenario)

        stops = [origin_idx] + r_scenario + [origin_idx]
        dist  = sum(full_dist[stops[i], stops[i+1]] for i in range(len(stops)-1))
        time_ = sum(full_time[stops[i], stops[i+1]] for i in range(len(stops)-1))
        time_ += len(r_scenario) * cfg.BOARDING_SECONDS
        route_distances.append(round(dist, 1))
        route_times_min.append(round(time_ / 60, 2))

    max_min = max(route_times_min) if route_times_min else 0
    tot_km  = sum(route_distances) / 1000

    print(f"    Total: {tot_km:.2f} km  |  longest route: {max_min:.1f} min")
    for c, (rt, rd) in enumerate(zip(route_times_min, route_distances)):
        nc = len(scenario_routes[c])
        flag = "  ⚠ OVER LIMIT" if rt > cfg.MAX_ROUTE_MINUTES else ""
        print(f"    Bus {c}: {nc:2d} children  {rt:.1f} min  "
              f"{rd/1000:.2f} km{flag}")

    return {
        "zone_idx":          z_idx,
        "zone_name":         zone["name"],
        "municipality":      zone["municipality"],
        "n_children":        n_zone,
        "k_buses":           k,
        "scenario_indices":  zone_scenario_idx.tolist(),
        "routes":            scenario_routes,
        "route_distances_m": route_distances,
        "route_times_min":   route_times_min,
        "total_distance_m":  round(sum(route_distances), 1),
        "total_time_min":    round(sum(route_times_min), 1),
        "max_route_min":     round(max_min, 2),
    }

# =============================================================================
# 5.  Zone merging — absorb zones with too few children
# =============================================================================

def merge_small_zones(zone_labels: np.ndarray,
                      zones: list[dict],
                      x: np.ndarray,
                      y: np.ndarray,
                      origin_idx: int) -> np.ndarray:
    """
    Iteratively merges the smallest zone into its geographically nearest
    neighbour until every active zone has at least MIN_ZONE_CHILDREN children.

    'Nearest' is measured centroid-to-centroid.

    Returns a new zone_labels array with updated assignments.
    """
    labels = zone_labels.copy()
    threshold = cfg.MIN_ZONE_CHILDREN

    # Build centroid lookup
    centroids = {i: z["centroid"] for i, z in enumerate(zones)}

    while True:
        # Count children per zone (exclude depot at origin_idx)
        counts = {}
        for i in range(len(x)):
            if i == origin_idx or labels[i] < 0:
                continue
            z = int(labels[i])
            counts[z] = counts.get(z, 0) + 1

        # Find the zone with fewest children (if below threshold)
        small = [(z, n) for z, n in counts.items() if n < threshold]
        if not small:
            break  # all zones meet the threshold — done

        # Pick the smallest zone
        victim_idx, victim_count = min(small, key=lambda t: t[1])

        # Find its nearest neighbour zone (by centroid distance)
        cx, cy = centroids[victim_idx]
        best_neighbour = min(
            (z for z in counts if z != victim_idx),
            key=lambda z: (centroids[z][0] - cx)**2 + (centroids[z][1] - cy)**2,
            default=None,
        )

        if best_neighbour is None:
            break  # only one zone left — nothing to merge into

        # Reassign all children from victim → neighbour
        labels[labels == victim_idx] = best_neighbour
        print(f"  Merged zone {zones[victim_idx]['name']!r} "
              f"({victim_count} children) → {zones[best_neighbour]['name']!r}")

    # Report final counts
    final_counts = {}
    for i in range(len(x)):
        if i == origin_idx or labels[i] < 0:
            continue
        z = int(labels[i])
        final_counts[z] = final_counts.get(z, 0) + 1

    print(f"\n  Active zones after merging: {len(final_counts)}")
    total_buses = sum(
        max(1, math.ceil(n / cfg.BUS_CAPACITY))
        for n in final_counts.values()
    )
    print(f"  Estimated total buses: {total_buses}")
    for z_idx in sorted(final_counts):
        k_est = max(1, math.ceil(final_counts[z_idx] / cfg.BUS_CAPACITY))
        print(f"    {zones[z_idx]['name']:<35s}: "
              f"{final_counts[z_idx]:3d} children → {k_est} bus(es)")
    return labels

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 62)
    print("Stage 2 — Multi-Objective CVRPTW  (Valle de Aburrá)")
    print("=" * 62)

    scenario    = np.load(cfg.SCENARIO_NPZ, allow_pickle=True)
    full_time   = scenario["time_matrix"]
    full_dist   = scenario["dist_matrix"]
    x           = scenario["x"]
    y           = scenario["y"]
    origin_idx  = int(scenario["origin_index"])
    node_zones  = scenario["node_zones"]
    n_children  = full_time.shape[0] - 1
    print(f"\n  {n_children} children  |  bus capacity {cfg.BUS_CAPACITY}  "
          f"|  max route {cfg.MAX_ROUTE_MINUTES} min  "
          f"|  min zone size {cfg.MIN_ZONE_CHILDREN}")

    print("\nLoading zone polygons …")
    zones = load_zones()

    print("\nAssigning children to zones …")
    zone_labels = assign_zones(x, y, origin_idx, zones, node_zones)

    # ── NEW: merge small zones before optimising ──────────────────────────
    print(f"\nMerging zones with fewer than {cfg.MIN_ZONE_CHILDREN} children …")
    zone_labels = merge_small_zones(zone_labels, zones, x, y, origin_idx)

    print(f"\nOptimising per zone …")
    results = []
    for z_idx, zone in enumerate(zones):
        res = process_zone(z_idx, zone, zone_labels,
                           full_time, full_dist, x, y, origin_idx)
        if res is not None:
            results.append(res)

    total_buses = sum(r["k_buses"] for r in results)
    total_km    = sum(r["total_distance_m"] for r in results) / 1000
    total_min   = sum(r["total_time_min"] for r in results)
    worst_min   = max(r["max_route_min"] for r in results)

    print(f"\n{'='*62}")
    print("GLOBAL SUMMARY")
    print(f"{'='*62}")
    print(f"  Active zones     : {len(results)}")
    print(f"  Total buses      : {total_buses}")
    print(f"  Total distance   : {total_km:.1f} km")
    print(f"  Cumulative time  : {total_min:.0f} min")
    print(f"  Longest route    : {worst_min:.1f} min  "
          f"(limit: {cfg.MAX_ROUTE_MINUTES} min)")

    output = {
        "config": {
            "n_children":          n_children,
            "bus_capacity":        cfg.BUS_CAPACITY,
            "max_route_minutes":   cfg.MAX_ROUTE_MINUTES,
            "boarding_seconds":    cfg.BOARDING_SECONDS,
            "span_balance_coeff":  cfg.SPAN_BALANCE_COEFF,
            "min_zone_children":   cfg.MIN_ZONE_CHILDREN,
        },
        "summary": {
            "total_buses":        total_buses,
            "total_distance_km":  round(total_km, 2),
            "cumulative_min":     round(total_min, 1),
            "worst_route_min":    round(worst_min, 2),
            "active_zones":       len(results),
        },
        "zone_labels":  zone_labels.tolist(),
        "zones_meta":   [{"idx": i, "name": z["name"],
                          "municipality": z["municipality"],
                          "centroid": list(z["centroid"])}
                         for i, z in enumerate(zones)],
        "results":      results,
    }

    Path(cfg.RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {cfg.RESULTS_JSON}")
    print("Stage 2 complete.\n")
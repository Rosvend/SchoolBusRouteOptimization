import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

import config as cfg


# =============================================================================
# 1.  Build the 6 routing zones
# =============================================================================

def build_routing_zones() -> list[dict]:
    """
    Returns a list of 6 zone dicts, each with:
        zone_name  : str
        color      : str  (hex, from config)
        members    : list[str]  individual commune/municipality names
        geometry   : merged Shapely geometry of all member polygons
        centroid   : (lon, lat) of the merged geometry
    """
    from shapely.ops import unary_union

    # Load individual commune geometries
    commune_geoms: dict[str, object] = {}
    with open(cfg.COMUNAS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        p    = feat["properties"]
        name = str(p.get("name", "")).strip()
        if name:
            commune_geoms[name] = shape(feat["geometry"])

    # Load municipality geometries
    muni_geoms: dict[str, object] = {}
    with open(cfg.VALLE_GEOJSON, encoding="utf-8") as f:
        gj2 = json.load(f)
    for feat in gj2["features"]:
        p = feat["properties"]
        if p.get("municipio") != "Medellín":
            muni_geoms[p["municipio"]] = shape(feat["geometry"])

    zones = []
    for zdef in cfg.ZONE_DEFINITIONS:
        polys = []
        for member in zdef["members"]:
            if member in commune_geoms:
                polys.append(commune_geoms[member])
            elif member in muni_geoms:
                polys.append(muni_geoms[member])
            else:
                print(f"  WARNING: '{member}' not found in any GeoJSON — "
                      "check spelling in config.ZONE_DEFINITIONS")

        if not polys:
            raise ValueError(
                f"Zone '{zdef['zone_name']}' has no valid polygons. "
                "Fix the 'members' list in config.ZONE_DEFINITIONS."
            )

        merged   = unary_union(polys)
        centroid = (merged.centroid.x, merged.centroid.y)
        zones.append({
            "zone_name": zdef["zone_name"],
            "color":     zdef["color"],
            "members":   zdef["members"],
            "geometry":  merged,
            "centroid":  centroid,
        })
        print(f"  Zone '{zdef['zone_name']:15s}': "
              f"{len(polys)} sub-polygon(s) merged")

    return zones


# =============================================================================
# 2.  Assign each child to one of the 6 zones
# =============================================================================

def assign_to_routing_zones(x: np.ndarray,
                             y: np.ndarray,
                             origin_idx: int,
                             zones: list[dict],
                             node_zones: np.ndarray) -> np.ndarray:
    
    # Build lookup: individual commune/municipality name → routing zone index
    member_to_zone: dict[str, int] = {}
    for z_idx, zone in enumerate(zones):
        for member in zone["members"]:
            member_to_zone[member] = z_idx

    n      = len(x)
    labels = np.full(n, -1, dtype=int)
    n_fallback_pip = 0
    n_fallback_centroid = 0

    from shapely.geometry import Point

    for i in range(n):
        if i == origin_idx:
            continue

        # Primary: lookup via saved fine-grained zone name
        saved = str(node_zones[i]) if i < len(node_zones) else ""
        if saved in member_to_zone:
            labels[i] = member_to_zone[saved]
            continue

        # Fallback 1: point-in-polygon
        pt       = Point(x[i], y[i])
        assigned = False
        for z_idx, zone in enumerate(zones):
            if zone["geometry"].covers(pt):
                labels[i] = z_idx
                assigned   = True
                n_fallback_pip += 1
                break

        # Fallback 2: nearest centroid
        if not assigned:
            dists      = [(x[i] - z["centroid"][0])**2 +
                          (y[i] - z["centroid"][1])**2 for z in zones]
            labels[i]  = int(np.argmin(dists))
            n_fallback_centroid += 1

    print(f"  Assignment method breakdown:")
    for z_idx, zone in enumerate(zones):
        n_z = int(np.sum(labels == z_idx))
        k_z = max(1, math.ceil(n_z / cfg.BUS_CAPACITY))
        print(f"    {zone['zone_name']:15s}: {n_z:3d} children → {k_z} bus(es)")
    if n_fallback_pip:
        print(f"  ({n_fallback_pip} children via point-in-polygon fallback)")
    if n_fallback_centroid:
        print(f"  ({n_fallback_centroid} children via nearest-centroid fallback)")

    return labels


# =============================================================================
# 3.  CVRPTW solver (one call per zone)
# =============================================================================

def solve_zone_cvrptw(time_sub: np.ndarray,
                      n_children: int,
                      depot_local: int,
                      k: int,
                      zone_name: str) -> tuple[list[list[int]], list[float]]:
    
    n           = time_sub.shape[0]
    max_ms      = int(cfg.MAX_ROUTE_MINUTES * 60 * 1000)
    boarding_ms = int(cfg.BOARDING_SECONDS * 1000)
    int_time    = (time_sub * 1000).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, k, depot_local)
    routing = pywrapcp.RoutingModel(manager)

    # Arc-cost: pure travel time (primary objective)
    def travel_cb(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        return int(int_time[fn, tn])

    arc_cb = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(arc_cb)

    # Time dimension: travel + boarding at every pickup stop
    def time_with_boarding(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        b  = 0 if fn == depot_local else boarding_ms
        return int(int_time[fn, tn]) + b

    time_cb = routing.RegisterTransitCallback(time_with_boarding)
    routing.AddDimension(
        time_cb,
        0,        
        max_ms,   
        True,     
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Secondary objective: balance route durations across buses
    time_dim.SetGlobalSpanCostCoefficient(cfg.SPAN_BALANCE_COEFF)

    # Capacity: demand = 1 per child stop
    def demand_cb(fi):
        return 0 if manager.IndexToNode(fi) == depot_local else 1

    dem_cb = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_cb, 0, [cfg.BUS_CAPACITY] * k, True, "Capacity"
    )

    # Very high drop penalty → forces OR-Tools to serve every child
    penalty = max_ms * 500
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
        print(f"    [{zone_name}] WARNING: no solution found. "
              "Try raising MAX_ROUTE_MINUTES or OR_TOOLS_TIME_LIMIT_SEC.")
        return [[] for _ in range(k)], [0.0] * k

    routes, route_times = [], []
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
        route_times.append(round(end_ms / 60_000, 2))

    served  = sum(len(r) for r in routes)
    if served < n_children:
        print(f"    [{zone_name}] WARNING: {n_children - served} child(ren) "
              "unserved — raise MAX_ROUTE_MINUTES.")

    return routes, route_times


# =============================================================================
# 4.  Per-zone driver
# =============================================================================

def process_zone(z_idx: int,
                 zone: dict,
                 routing_labels: np.ndarray,
                 full_time: np.ndarray,
                 full_dist: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 origin_idx: int,
                 node_zones: np.ndarray) -> dict | None:
    """Run CVRPTW for one of the 6 zones. Returns result dict or None."""

    child_mask        = routing_labels == z_idx
    zone_scenario_idx = np.where(child_mask)[0]   
    n_zone            = len(zone_scenario_idx)

    if n_zone == 0:
        print(f"  {zone['zone_name']}: 0 children — skipped")
        return None

    k = max(1, math.ceil(n_zone / cfg.BUS_CAPACITY))
    print(f"\n  {zone['zone_name']}: {n_zone} children → {k} bus(es)")

    # Build sub-matrices: zone children + depot
    sub_indices = np.append(zone_scenario_idx, origin_idx)
    depot_local = len(sub_indices) - 1
    time_sub    = full_time[np.ix_(sub_indices, sub_indices)]
    dist_sub    = full_dist[np.ix_(sub_indices, sub_indices)]

    # Solve CVRPTW (routes contain LOCAL indices into zone_scenario_idx)
    local_routes, rt_min = solve_zone_cvrptw(
        time_sub, n_zone, depot_local, k, zone["zone_name"])

    # Convert local indices → full scenario indices and compute metrics
    scenario_routes  = []
    route_distances  = []
    route_times_min  = []

    for r_local, rt in zip(local_routes, rt_min):
        r_scenario = [int(zone_scenario_idx[i]) for i in r_local]
        scenario_routes.append(r_scenario)

        stops = [origin_idx] + r_scenario + [origin_idx]
        dist  = sum(full_dist[stops[i], stops[i+1]]
                    for i in range(len(stops) - 1))
        time_ = sum(full_time[stops[i], stops[i+1]]
                    for i in range(len(stops) - 1))
        time_ += len(r_scenario) * cfg.BOARDING_SECONDS
        route_distances.append(round(dist, 1))
        route_times_min.append(round(time_ / 60, 2))

    max_min = max(route_times_min) if route_times_min else 0
    tot_km  = sum(route_distances) / 1000

    print(f"    Total: {tot_km:.2f} km  |  longest route: {max_min:.1f} min")
    for c, (rt, rd) in enumerate(zip(route_times_min, route_distances)):
        nc   = len(scenario_routes[c])
        flag = "  ⚠ OVER LIMIT" if rt > cfg.MAX_ROUTE_MINUTES else ""
        print(f"    Bus {c+1}: {nc:2d} children  {rt:.1f} min  "
              f"{rd/1000:.2f} km{flag}")

    return {
        "zone_idx":          z_idx,
        "zone_name":         zone["zone_name"],
        "zone_color":        zone["color"],
        "members":           zone["members"],
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
# 5.  Main
# =============================================================================

def main():
    print("=" * 62)
    print("Stage 2 — Zone-Based Multi-Objective CVRPTW")
    print("=" * 62)

    scenario   = np.load(cfg.SCENARIO_NPZ, allow_pickle=True)
    full_time  = scenario["time_matrix"]
    full_dist  = scenario["dist_matrix"]
    x          = scenario["x"]
    y          = scenario["y"]
    origin_idx = int(scenario["origin_index"])
    node_zones = scenario["node_zones"]
    n_children = full_time.shape[0] - 1

    print(f"\n  {n_children} children  |  capacity {cfg.BUS_CAPACITY}  "
          f"|  max route {cfg.MAX_ROUTE_MINUTES} min")
    print(f"  Theoretical minimum buses: "
          f"ceil({n_children}/{cfg.BUS_CAPACITY}) = "
          f"{math.ceil(n_children/cfg.BUS_CAPACITY)}")

    # Build the 6 routing zones
    print("\nBuilding 6 routing zones …")
    routing_zones = build_routing_zones()

    # Assign children to one of the 6 zones
    print("\nAssigning children to routing zones …")
    routing_labels = assign_to_routing_zones(
        x, y, origin_idx, routing_zones, node_zones)

    # Solve per zone
    print("\nOptimising routes per zone …")
    results = []
    for z_idx, zone in enumerate(routing_zones):
        res = process_zone(
            z_idx, zone, routing_labels,
            full_time, full_dist, x, y, origin_idx, node_zones)
        if res is not None:
            results.append(res)

    # Global summary
    total_buses = sum(r["k_buses"]           for r in results)
    total_km    = sum(r["total_distance_m"]  for r in results) / 1000
    total_min   = sum(r["total_time_min"]    for r in results)
    worst_min   = max(r["max_route_min"]     for r in results)

    print(f"\n{'='*62}")
    print("GLOBAL SUMMARY")
    print(f"{'='*62}")
    print(f"  Active zones     : {len(results)} / {len(routing_zones)}")
    print(f"  Total buses      : {total_buses}")
    print(f"  Total distance   : {total_km:.1f} km")
    print(f"  Cumulative time  : {total_min:.0f} min")
    print(f"  Longest route    : {worst_min:.1f} min "
          f"(limit: {cfg.MAX_ROUTE_MINUTES} min)")

    output = {
        "config": {
            "n_children":          n_children,
            "bus_capacity":        cfg.BUS_CAPACITY,
            "max_route_minutes":   cfg.MAX_ROUTE_MINUTES,
            "boarding_seconds":    cfg.BOARDING_SECONDS,
            "span_balance_coeff":  cfg.SPAN_BALANCE_COEFF,
        },
        "summary": {
            "total_buses":        total_buses,
            "total_distance_km":  round(total_km, 2),
            "cumulative_min":     round(total_min, 1),
            "worst_route_min":    round(worst_min, 2),
            "active_zones":       len(results),
        },
        "x":             x.tolist(),
        "y":             y.tolist(),
        "origin_idx":    origin_idx,
        "node_zones":    node_zones.tolist(),
        "routing_labels":routing_labels.tolist(),
        "zone_defs":     [{"zone_name": z["zone_name"],
                           "color":     z["color"],
                           "members":   z["members"],
                           "centroid":  list(z["centroid"])}
                          for z in routing_zones],
        "results":       results,
    }

    Path(cfg.RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {cfg.RESULTS_JSON}")
    print("Stage 2 complete.\n")


if __name__ == "__main__":
    main()
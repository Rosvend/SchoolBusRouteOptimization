"""CVRPTW with 6 fixed geographic zones (ported from naomi).

Each child is assigned to one of 6 zones (point-in-polygon, then nearest-
centroid fallback). Each zone is solved independently with OR-Tools CVRPTW
under capacity, time-window, and route-duration constraints.
"""
from __future__ import annotations

import math

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from shapely.geometry import Point

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver
from ..data import zone_definitions as zd


def _assign_children_to_zones(scenario: Scenario) -> np.ndarray:
    """Returns labels[i] = zone index for each child (children-only order)."""
    if not scenario.zones:
        raise RuntimeError("CVRPTW adapter requires scenario.zones to be set")

    member_to_zone = {m: i for i, z in enumerate(scenario.zones) for m in z.members}
    ci = scenario.children_idx
    labels = np.full(len(ci), -1, dtype=int)

    for k, idx in enumerate(ci):
        # Primary: cached zone-name from the generator
        if scenario.node_zones is not None:
            saved = str(scenario.node_zones[idx])
            if saved in member_to_zone:
                labels[k] = member_to_zone[saved]
                continue

        pt = Point(scenario.x[idx], scenario.y[idx])
        for z_idx, zone in enumerate(scenario.zones):
            if zone.geometry.covers(pt):
                labels[k] = z_idx
                break

        if labels[k] == -1:
            cents = np.array([z.centroid for z in scenario.zones])
            d = (cents[:, 0] - scenario.x[idx]) ** 2 + (cents[:, 1] - scenario.y[idx]) ** 2
            labels[k] = int(np.argmin(d))

    return labels


def _solve_zone_cvrptw(time_sub: np.ndarray,
                       depot_local: int,
                       k_buses: int,
                       capacity: int) -> list[list[int]]:
    n = time_sub.shape[0]
    max_ms = int(zd.MAX_ROUTE_MINUTES * 60 * 1000)
    boarding_ms = int(zd.BOARDING_SECONDS * 1000)
    int_time = (time_sub * 1000).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, k_buses, depot_local)
    routing = pywrapcp.RoutingModel(manager)

    def travel_cb(fi, ti):
        return int(int_time[manager.IndexToNode(fi), manager.IndexToNode(ti)])
    arc_cb = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(arc_cb)

    def time_with_boarding(fi, ti):
        fn = manager.IndexToNode(fi); tn = manager.IndexToNode(ti)
        b = 0 if fn == depot_local else boarding_ms
        return int(int_time[fn, tn]) + b
    time_cb = routing.RegisterTransitCallback(time_with_boarding)
    routing.AddDimension(time_cb, 0, max_ms, True, "Time")
    routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(zd.SPAN_BALANCE_COEFF)

    def demand_cb(fi):
        return 0 if manager.IndexToNode(fi) == depot_local else 1
    dem_cb = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(dem_cb, 0, [capacity] * k_buses, True, "Capacity")

    penalty = max_ms * 500
    for node in range(n):
        if node != depot_local:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = zd.OR_TOOLS_TIME_LIMIT_SEC

    sol = routing.SolveWithParameters(p)
    if sol is None:
        return [[] for _ in range(k_buses)]

    routes = []
    for v in range(k_buses):
        r = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot_local:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r)
    return routes


class CVRPTWZonesSolver(Solver):
    name = "cvrptw_zones"

    def solve(self, scenario: Scenario) -> Solution:
        ci = scenario.children_idx
        zone_labels = _assign_children_to_zones(scenario)
        n_zones = len(scenario.zones)

        all_routes: list[list[int]] = []
        served: set[int] = set()

        for z_idx in range(n_zones):
            mask = zone_labels == z_idx
            zone_scenario_idx = ci[mask]
            n_zone = len(zone_scenario_idx)
            if n_zone == 0:
                continue
            k_buses = max(1, math.ceil(n_zone / scenario.bus_capacity))

            sub_indices = np.append(zone_scenario_idx, scenario.origin_index)
            depot_local = len(sub_indices) - 1
            time_sub = scenario.time_matrix[np.ix_(sub_indices, sub_indices)]

            local_routes = _solve_zone_cvrptw(time_sub, depot_local, k_buses,
                                              scenario.bus_capacity)

            for lr in local_routes:
                if not lr:
                    continue
                scenario_route = [int(zone_scenario_idx[i]) for i in lr]
                served.update(scenario_route)
                all_routes.append(
                    [scenario.origin_index] + scenario_route + [scenario.origin_index]
                )

        return Solution(
            routes=all_routes,
            cluster_labels=zone_labels,
            served_indices=served,
            extra={"n_zones": n_zones},
        )

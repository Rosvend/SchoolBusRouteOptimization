from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import silhouette_score

from .scenario import Scenario
from .solution import Solution


def fleet_distance(scenario: Scenario, solution: Solution) -> float:
    D = scenario.dist_matrix
    total = 0.0
    for route in solution.routes:
        for u, v in zip(route[:-1], route[1:]):
            total += float(D[u, v])
    return total


def buses_used(solution: Solution) -> int:
    return sum(1 for r in solution.routes if len(r) > 2)


def served_set(scenario: Scenario, solution: Solution) -> set[int]:
    if solution.served_indices is not None:
        return set(int(i) for i in solution.served_indices)
    served: set[int] = set()
    for r in solution.routes:
        for i in r:
            if i != scenario.origin_index:
                served.add(int(i))
    return served


def coverage(scenario: Scenario, solution: Solution) -> float:
    if scenario.n_children == 0:
        return 1.0
    return len(served_set(scenario, solution)) / scenario.n_children


def _route_load(scenario: Scenario, route: list[int]) -> float:
    """Passengers on a route. Depot appears at start AND end, hence the -2.

    With per-node demands (external CVRP instances) the load is the summed
    demand instead of the stop count.
    """
    if len(route) <= 2:
        return 0.0
    if scenario.demands is None:
        return float(len(route) - 2)
    return float(sum(scenario.demands[i] for i in route[1:-1]))


def capacity_violations(scenario: Scenario, solution: Solution) -> int:
    cap = scenario.bus_capacity
    n = 0
    for r in solution.routes:
        if _route_load(scenario, r) > cap + 1e-9:
            n += 1
    return n


def silhouette(scenario: Scenario, solution: Solution) -> float:
    if solution.cluster_labels is None:
        return float("nan")
    labels = np.asarray(solution.cluster_labels)
    if labels.shape[0] != scenario.n_children:
        return float("nan")
    unique = np.unique(labels[labels >= 0])
    if unique.size < 2:
        return float("nan")
    D_sym = scenario.dist_sym_children
    valid = labels >= 0
    if valid.sum() < 2:
        return float("nan")
    try:
        return float(silhouette_score(
            D_sym[np.ix_(valid, valid)],
            labels[valid],
            metric="precomputed",
        ))
    except Exception:
        return float("nan")


def max_route_distance(scenario: Scenario, solution: Solution) -> float:
    D = scenario.dist_matrix
    best = 0.0
    for r in solution.routes:
        d = sum(float(D[u, v]) for u, v in zip(r[:-1], r[1:]))
        if d > best:
            best = d
    return best


def mean_route_load(scenario: Scenario, solution: Solution) -> float:
    loads = [_route_load(scenario, r) for r in solution.routes if len(r) > 2]
    return float(np.mean(loads)) if loads else 0.0


def compute_all(scenario: Scenario, solution: Solution) -> dict:
    return {
        "fleet_distance_m": fleet_distance(scenario, solution),
        "buses_used": buses_used(solution),
        "coverage": coverage(scenario, solution),
        "capacity_violations": capacity_violations(scenario, solution),
        "silhouette": silhouette(scenario, solution),
        "max_route_distance_m": max_route_distance(scenario, solution),
        "mean_route_load": mean_route_load(scenario, solution),
    }

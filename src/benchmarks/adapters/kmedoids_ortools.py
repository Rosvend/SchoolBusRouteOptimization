"""K-medoids capacitated clustering + OR-Tools TSP per cluster.

Ported verbatim from src/clustering.py and src/tsp_solver.py (the main-branch
pipeline). Wrapped behind the Solver interface; no file I/O.
"""
from __future__ import annotations

import kmedoids
import networkx as nx
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver


def capacitated_assign(D: np.ndarray, medoids: list[int], capacity: int):
    """Min-cost flow assignment of children to medoids respecting capacity.
    Source → child (cap 1) → medoid (cap 1, cost=int(D*1000)) → sink (cap=C).
    """
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
    flow = nx.min_cost_flow(G)

    labels = np.empty(n, dtype=int)
    total = 0.0
    for i in range(n):
        for m_idx in range(len(medoids)):
            if flow[("c", i)].get(("m", m_idx), 0) == 1:
                labels[i] = m_idx
                total += D[i, medoids[m_idx]]
                break
    return labels, total


def recompute_medoids(D: np.ndarray, labels: np.ndarray, k: int) -> list[int]:
    medoids = []
    for c in range(k):
        members = np.where(labels == c)[0]
        sub = D[np.ix_(members, members)]
        medoids.append(int(members[np.argmin(sub.sum(axis=1))]))
    return medoids


def solve_tsp(dist_sub: np.ndarray, depot: int, time_limit_s: int = 5) -> list[int]:
    n = dist_sub.shape[0]
    if n <= 2:
        return list(range(n))
    int_dist = (dist_sub * 1000).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def cb(fi, ti):
        return int(int_dist[manager.IndexToNode(fi), manager.IndexToNode(ti)])

    cb_idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit_s

    sol = routing.SolveWithParameters(p)
    if sol is None:
        raise RuntimeError("OR-Tools could not solve TSP sub-problem")

    route = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    return route


class KmedoidsORToolsSolver(Solver):
    name = "kmedoids_ortools"

    def __init__(self, max_iter: int = 50, tsp_time_limit_s: int = 5,
                 random_state: int = 42):
        self.max_iter = max_iter
        self.tsp_time_limit_s = tsp_time_limit_s
        self.random_state = random_state

    def solve(self, scenario: Scenario) -> Solution:
        ci = scenario.children_idx
        D = scenario.dist_sym_children
        n_children = scenario.n_children
        k = int(np.ceil(n_children / scenario.bus_capacity))

        # Initial PAM
        result = kmedoids.fasterpam(D, k, random_state=self.random_state)
        medoids = list(int(m) for m in result.medoids)

        # Iterative capacitated refinement
        labels = np.zeros(n_children, dtype=int)
        for _ in range(self.max_iter):
            labels, _ = capacitated_assign(D, medoids, scenario.bus_capacity)
            new = recompute_medoids(D, labels, k)
            if new == medoids:
                break
            medoids = new

        # Per-cluster TSP on directed full distance matrix
        full_D = scenario.dist_matrix
        routes: list[list[int]] = []
        for c in range(k):
            child_local = np.where(labels == c)[0]
            scenario_indices = ci[child_local]
            sub_indices = np.append(scenario_indices, scenario.origin_index)
            depot_pos = len(sub_indices) - 1
            sub = full_D[np.ix_(sub_indices, sub_indices)]
            local = solve_tsp(sub, depot_pos, self.tsp_time_limit_s)
            routes.append([int(sub_indices[i]) for i in local])

        return Solution(
            routes=routes,
            cluster_labels=labels,
            extra={"k": k, "medoids": medoids},
        )

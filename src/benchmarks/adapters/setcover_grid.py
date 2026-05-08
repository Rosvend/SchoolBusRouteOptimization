"""Grid-seeded set-cover (ported from roberto's notebook).

Lay a Cartesian grid of seed points across the bbox; for each seed, take
the `cap` Euclidean-nearest children, then approximate a tour via
NetworkX greedy_tsp using the actual road distance matrix as edge weight.
PuLP set-cover with a configurable coverage threshold (default 1.0;
roberto's notebook used 0.95) minimizes total tour distance.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pulp
from networkx.algorithms.approximation import traveling_salesman_problem

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver


def _nn_tour_for_route(scenario_members: list[int],
                       full_D: np.ndarray, depot: int) -> list[int]:
    """Greedy NN tour starting and ending at depot, visiting each member once."""
    remaining = list(scenario_members)
    tour = [depot]
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda n: full_D[cur, n])
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    tour.append(depot)
    return tour


def _cluster_around_point(xi: np.ndarray, yi: np.ndarray,
                          xo: float, yo: float,
                          ni: list[int], origin_idx: int,
                          full_D: np.ndarray, cap: int):
    """Pick `cap` Euclidean-nearest children to (xo, yo), then greedy-TSP
    over {cluster ∪ origin} using full road-distance weights.
    Returns (tour_scenario_indices, cost) or (None, inf) if any leg is non-finite.
    """
    n_children_in_xi = len(xi) - 1  # last is origin
    distances = np.sqrt((xi[:n_children_in_xi] - xo) ** 2 + (yi[:n_children_in_xi] - yo) ** 2)
    nearest_local = list(np.argsort(distances)[:cap])
    nearest_scenario = [int(ni[i]) for i in nearest_local]
    nearest_scenario.append(origin_idx)

    G = nx.DiGraph()
    for i in range(len(nearest_scenario)):
        for j in range(len(nearest_scenario)):
            if i == j:
                continue
            w = full_D[nearest_scenario[i], nearest_scenario[j]]
            if not np.isfinite(w):
                return None, float("inf")
            G.add_edge(i, j, weight=float(w))

    tour = traveling_salesman_problem(G, cycle=True, weight="weight",
                                       method=nx.algorithms.approximation.greedy_tsp)
    cost = sum(full_D[nearest_scenario[tour[i]], nearest_scenario[tour[i + 1]]]
               for i in range(len(tour) - 1))
    tour_scenario = [nearest_scenario[i] for i in tour]
    return (frozenset(nearest_scenario[:-1]), cost, tour_scenario)


class SetCoverGridSolver(Solver):
    name = "setcover_grid"

    def __init__(self, coverage_threshold: float = 1.0):
        # 1.0 = full coverage (apples-to-apples with other algos)
        # 0.95 = roberto's relaxation
        self.coverage_threshold = coverage_threshold

    def solve(self, scenario: Scenario) -> Solution:
        ci = scenario.children_idx
        n = scenario.n_children
        cap = scenario.bus_capacity

        # Children-only x,y plus origin appended (mirrors roberto's data layout)
        xi = np.concatenate([scenario.x[ci], [scenario.x[scenario.origin_index]]])
        yi = np.concatenate([scenario.y[ci], [scenario.y[scenario.origin_index]]])
        ni = list(int(i) for i in ci) + [int(scenario.origin_index)]

        xmin, xmax = float(np.min(xi)), float(np.max(xi))
        ymin, ymax = float(np.min(yi)), float(np.max(yi))
        # Roberto's grid resolution heuristic (slightly higher density than N seeds)
        d = float(np.sqrt(((xmax - xmin) * (ymax - ymin)) / max(n, 1)))
        if d <= 0:
            d = 1e-4
        xs, ys = np.meshgrid(np.arange(xmin, xmax + d, d),
                             np.arange(ymin, ymax + d, d))

        candidates = []  # (frozenset of scenario indices, cost, tour scenario indices)
        for xo, yo in zip(xs.flatten(), ys.flatten()):
            r = _cluster_around_point(xi, yi, xo, yo, ni, scenario.origin_index,
                                      scenario.dist_matrix, cap)
            if r is None or r[1] == float("inf"):
                continue
            members, cost, tour_scen = r
            candidates.append((members, cost, tour_scen))

        # Dedup by member set; keep the lowest-cost candidate per set
        best: dict[frozenset, tuple[float, list[int]]] = {}
        for members, cost, tour_scen in candidates:
            if members not in best or cost < best[members][0]:
                best[members] = (cost, tour_scen)
        candidates = [(m, c, t) for m, (c, t) in best.items()]

        # ILP: choose minimum-total-distance subset that covers ≥ threshold·n
        prob = pulp.LpProblem("SetCover_Grid", pulp.LpMinimize)
        x = [pulp.LpVariable(f"r_{j}", cat="Binary") for j in range(len(candidates))]
        kid = [pulp.LpVariable(f"k_{i}", cat="Binary") for i in range(n)]

        # Children: covered_i ≥ kid_i
        for i, child_scenario in enumerate(ci):
            covering = [x[j] for j, (m, _, _) in enumerate(candidates) if int(child_scenario) in m]
            if covering:
                prob += pulp.lpSum(covering) >= kid[i], f"Cover_{i}"
            else:
                # No candidate covers this child → kid_i must be 0
                prob += kid[i] == 0, f"Uncoverable_{i}"

        prob += pulp.lpSum(x[j] * candidates[j][1] for j in range(len(candidates))), "TotalDistance"
        prob += pulp.lpSum(kid) >= self.coverage_threshold * n, "Coverage"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [candidates[j] for j, v in enumerate(x) if pulp.value(v) == 1]

        routes: list[list[int]] = []
        served: set[int] = set()
        cluster_labels = np.full(n, -1, dtype=int)
        scenario_to_local = {int(s): k for k, s in enumerate(ci)}

        for c, (members, _, _tour_scen) in enumerate(selected):
            # greedy_tsp on directed graphs can return non-Hamiltonian tours
            # (revisits nodes). It's used for the ILP cost only; for the
            # final route emit a clean NN tour over the candidate's members.
            kids = list(int(m) for m in members)
            t = _nn_tour_for_route(kids, scenario.dist_matrix, scenario.origin_index)
            routes.append(t)
            for s in members:
                served.add(int(s))
                local = scenario_to_local.get(int(s))
                if local is not None and cluster_labels[local] == -1:
                    cluster_labels[local] = c

        return Solution(
            routes=routes,
            cluster_labels=cluster_labels,
            served_indices=served,
            extra={
                "n_candidates": len(candidates),
                "n_buses": len(selected),
                "coverage_threshold": self.coverage_threshold,
            },
        )

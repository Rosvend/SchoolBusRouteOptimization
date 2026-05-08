"""Per-child set-cover (ported from mariana:src/setcover_rutas.ipynb).

For each child, build a candidate route via NN packing in road-distance.
Dedupe (frozensets), then PuLP/CBC set cover minimizing #buses with
full coverage. Order each selected route by NN tour for distance.
"""
from __future__ import annotations

import numpy as np
import pulp

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver


def _build_candidate_route(center: int, D: np.ndarray, capacity: int) -> frozenset:
    dists = sorted((D[center, j], j) for j in range(D.shape[0]) if j != center)
    members = {center}
    for _, j in dists:
        if len(members) >= capacity:
            break
        members.add(int(j))
    return frozenset(members)


def _nn_tour(member_scenario_idx: list[int],
             full_D: np.ndarray,
             origin: int) -> list[int]:
    """Greedy nearest-neighbor tour starting and ending at the depot."""
    remaining = list(member_scenario_idx)
    tour = [origin]
    current = origin
    while remaining:
        nxt = min(remaining, key=lambda n: full_D[current, n])
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    tour.append(origin)
    return tour


class SetCoverPerChildSolver(Solver):
    name = "setcover_perchild"

    def solve(self, scenario: Scenario) -> Solution:
        ci = scenario.children_idx
        n = scenario.n_children
        D = scenario.dist_sym_children

        # Generate one candidate per child seed (deduped)
        seen: set[frozenset] = set()
        candidates: list[frozenset] = []
        for i in range(n):
            r = _build_candidate_route(i, D, scenario.bus_capacity)
            if r not in seen:
                seen.add(r)
                candidates.append(r)

        # Coverage map: child_local → list of candidate indices
        coverage = {i: [] for i in range(n)}
        for j, route in enumerate(candidates):
            for s in route:
                coverage[s].append(j)

        prob = pulp.LpProblem("SetCover_BusRoutes", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(len(candidates))]
        prob += pulp.lpSum(x)
        for i in range(n):
            covering = [x[j] for j in coverage[i]]
            if covering:
                prob += pulp.lpSum(covering) >= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [candidates[j] for j, var in enumerate(x) if pulp.value(var) == 1]

        full_D = scenario.dist_matrix
        routes: list[list[int]] = []
        cluster_labels = np.full(n, -1, dtype=int)
        served: set[int] = set()
        for c, route in enumerate(selected):
            members_scenario = [int(ci[i]) for i in route]
            tour = _nn_tour(members_scenario, full_D, scenario.origin_index)
            routes.append(tour)
            served.update(members_scenario)
            for child_local in route:
                # routes overlap → label child by first selected route covering it
                if cluster_labels[child_local] == -1:
                    cluster_labels[child_local] = c

        return Solution(
            routes=routes,
            cluster_labels=cluster_labels,
            served_indices=served,
            extra={"n_candidates": len(candidates), "n_buses": len(selected)},
        )

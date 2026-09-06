"""Adaptive large-neighbourhood search adapter.

Wraps the SBRP search in ``algorithms/alns_sbrp.py`` behind the benchmark's
``Solver`` interface. The search is budget-matched to the clustering pipelines:
a run gets the same ``k * tsp_time_limit_s`` wall clock that K-medoids and
Sectorial spend inside OR-Tools, so any quality difference is attributable to
the method rather than to compute.

The initial solution reuses the baseline's capacitated k-medoids partition, but
routes each cluster with a nearest-neighbour tour plus 2-opt instead of calling
OR-Tools: a full per-cluster TSP would consume the entire budget before the
destroy-repair search had run a single iteration.
"""
from __future__ import annotations

import time

import kmedoids
import numpy as np

from ..algorithms.alns_sbrp import (
    AlnsContext,
    AlnsParams,
    greedy_initial_routes,
    nearest_neighbour_route,
    polish_routes,
    run_alns,
)
from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver
from .kmedoids_ortools import capacitated_assign, recompute_medoids


class AlnsSolver(Solver):
    name = "alns"

    def __init__(self, tsp_time_limit_s: int = 5, time_budget_s: float | None = None,
                 init_max_iter: int = 20,
                 destroy_min_frac: float = 0.10, destroy_max_frac: float = 0.25,
                 destroy_cap: int = 60, worst_p: float = 3.0, shaw_p: float = 3.0,
                 regret_k: int = 2, op_decay: float = 0.8,
                 sa_worse: float = 0.05, sa_accept_prob: float = 0.50,
                 sa_horizon_iters: int = 25_000, random_state: int = 42):
        # tsp_time_limit_s is not a TSP budget here: it is the per-bus wall clock
        # the clustering pipelines are allowed, reused so the latency columns of
        # ALNS, K-medoids and Sectorial are directly comparable.
        self.tsp_time_limit_s = tsp_time_limit_s
        # Fixed per-instance budget, overriding the k-proportional one. Used for
        # external CVRP instances, where there is no per-bus budget to match.
        self.time_budget_s = time_budget_s
        self.init_max_iter = init_max_iter
        self.random_state = random_state
        self.params = AlnsParams(
            destroy_min_frac=destroy_min_frac,
            destroy_max_frac=destroy_max_frac,
            destroy_cap=destroy_cap,
            worst_p=worst_p,
            shaw_p=shaw_p,
            regret_k=regret_k,
            op_decay=op_decay,
            sa_worse=sa_worse,
            sa_accept_prob=sa_accept_prob,
            sa_horizon_iters=sa_horizon_iters,
        )

    def _initial_routes(self, scenario: Scenario, ctx: AlnsContext, k: int):
        """Capacitated k-medoids partition, each cluster toured by NN + 2-opt."""
        demands = ctx.demands[ctx.customers]
        if not np.allclose(demands, 1.0):
            # External CVRP instances carry non-unit demands, for which the
            # min-cost-flow assignment is no longer exact.
            return polish_routes(greedy_initial_routes(ctx, k), ctx)

        ci = scenario.children_idx
        D = scenario.dist_sym_children
        result = kmedoids.fasterpam(D, k, random_state=self.random_state)
        medoids = [int(m) for m in result.medoids]

        labels = np.zeros(scenario.n_children, dtype=int)
        for _ in range(self.init_max_iter):
            labels, _ = capacitated_assign(D, medoids, scenario.bus_capacity)
            new = recompute_medoids(D, labels, k)
            if new == medoids:
                break
            medoids = new

        routes = []
        for c in range(k):
            members = ci[np.where(labels == c)[0]]
            routes.append(nearest_neighbour_route(members, ctx))
        return polish_routes(routes, ctx)

    def solve(self, scenario: Scenario) -> Solution:
        t_start = time.perf_counter()
        ctx = AlnsContext(
            dist=scenario.dist_matrix,
            depot=scenario.origin_index,
            customers=scenario.children_idx,
            capacity=scenario.bus_capacity,
            demands=getattr(scenario, "demands", None),
            params=self.params,
        )
        k = ctx.min_routes()
        rng = np.random.default_rng(self.random_state + scenario.seed)

        initial = self._initial_routes(scenario, ctx, k)
        init_s = time.perf_counter() - t_start

        total_budget = (self.time_budget_s if self.time_budget_s is not None
                        else k * self.tsp_time_limit_s)
        budget = total_budget - init_s
        outcome = run_alns(ctx, initial, budget, rng)

        # Bus index per child, in children-local order.
        pos = np.full(len(scenario.node_ids), -1, dtype=int)
        pos[scenario.children_idx] = np.arange(scenario.n_children)
        labels = np.full(scenario.n_children, -1, dtype=int)
        for bus, route in enumerate(outcome.routes):
            for c in route[1:-1]:
                labels[pos[c]] = bus

        return Solution(
            routes=outcome.routes,
            cluster_labels=labels,
            extra={
                "k": k,
                "iterations": outcome.iterations,
                "init_objective_m": outcome.init_objective,
                "best_objective_m": outcome.objective,
                "improvement_pct": 100.0 * (outcome.init_objective - outcome.objective)
                / outcome.init_objective if outcome.init_objective else 0.0,
                "init_time_s": init_s,
                "search_budget_s": budget,
                "stage_times_s": outcome.timers,
                "repair_dead_ends": outcome.repair_dead_ends,
                "destroy_counts": outcome.destroy_counts,
                "repair_counts": outcome.repair_counts,
            },
        )

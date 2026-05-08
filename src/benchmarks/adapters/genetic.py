"""Genetic Algorithm for capacitated VRP.

Chromosome: permutation of children (children-only local indices).
Decoder:    greedy split into chunks of size ≤ bus_capacity (each chunk = one bus).
Tour cost:  per-chunk nearest-neighbor + single 2-opt pass on the directed
            road-distance matrix (matches the cost the rest of the harness reports).
Operators:  order crossover (OX), swap mutation, tournament selection, elitism.
"""
from __future__ import annotations

import numpy as np

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver


def _nn_tour(scenario_indices: list[int], full_D: np.ndarray, depot: int) -> list[int]:
    remaining = list(scenario_indices)
    tour = [depot]
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda n: full_D[cur, n])
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    tour.append(depot)
    return tour


def _two_opt(tour: list[int], full_D: np.ndarray, max_passes: int = 1) -> list[int]:
    """One pass of 2-opt; depot is fixed at endpoints."""
    n = len(tour)
    if n <= 4:
        return tour
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[j + 1]
                delta = (full_D[a, c] + full_D[b, d]) - (full_D[a, b] + full_D[c, d])
                if delta < -1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        # one full pass is the contract; spec says "single 2-opt pass"
    return tour


def _decode(perm: np.ndarray,
            children_scenario_idx: np.ndarray,
            full_D: np.ndarray,
            depot: int,
            capacity: int):
    """Split permutation into capacity-respecting buses, build tours, return
    (routes, fleet_distance)."""
    routes: list[list[int]] = []
    total = 0.0
    for start in range(0, len(perm), capacity):
        chunk_local = perm[start:start + capacity]
        chunk_scen = [int(children_scenario_idx[i]) for i in chunk_local]
        tour = _nn_tour(chunk_scen, full_D, depot)
        tour = _two_opt(tour, full_D, max_passes=1)
        d = sum(full_D[u, v] for u, v in zip(tour[:-1], tour[1:]))
        routes.append(tour)
        total += float(d)
    return routes, total


def _ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in set(p1[a:b + 1].tolist())]
    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1
    return child


def _swap_mutation(chrom: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(chrom)
    if n < 2:
        return chrom
    i, j = rng.choice(n, size=2, replace=False)
    out = chrom.copy()
    out[i], out[j] = out[j], out[i]
    return out


class GeneticSolver(Solver):
    name = "genetic"

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 200,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism: int = 2,
                 stagnation_limit: int = 30,
                 random_state: int = 42):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.stagnation_limit = stagnation_limit
        self.random_state = random_state

    def solve(self, scenario: Scenario) -> Solution:
        rng = np.random.default_rng(self.random_state + scenario.seed)
        ci = scenario.children_idx
        n = scenario.n_children
        D = scenario.dist_matrix
        depot = scenario.origin_index
        cap = scenario.bus_capacity

        # Initial population: random permutations
        pop = np.array([rng.permutation(n) for _ in range(self.population_size)])
        fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            _, fitness[i] = _decode(pop[i], ci, D, depot, cap)

        best_idx = int(np.argmin(fitness))
        best_perm = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])
        stagnant = 0

        for gen in range(self.generations):
            # Elitism: carry top-k unchanged
            order = np.argsort(fitness)
            new_pop = [pop[i].copy() for i in order[:self.elitism]]

            while len(new_pop) < self.population_size:
                # Tournament selection
                def pick():
                    cand = rng.choice(self.population_size, size=self.tournament_size, replace=False)
                    return pop[cand[np.argmin(fitness[cand])]]

                p1, p2 = pick(), pick()
                child = _ox_crossover(p1, p2, rng)
                if rng.random() < self.mutation_rate:
                    child = _swap_mutation(child, rng)
                new_pop.append(child)

            pop = np.array(new_pop)
            for i in range(self.population_size):
                _, fitness[i] = _decode(pop[i], ci, D, depot, cap)

            cur_best = int(np.argmin(fitness))
            if fitness[cur_best] < best_fit - 1e-6:
                best_fit = float(fitness[cur_best])
                best_perm = pop[cur_best].copy()
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= self.stagnation_limit:
                    break

        routes, _ = _decode(best_perm, ci, D, depot, cap)

        # Cluster labels: assign each child to its bus index
        labels = np.full(n, -1, dtype=int)
        bus = 0
        for start in range(0, len(best_perm), cap):
            for local in best_perm[start:start + cap]:
                labels[int(local)] = bus
            bus += 1

        return Solution(
            routes=routes,
            cluster_labels=labels,
            extra={"best_fitness": best_fit, "generations_run": gen + 1},
        )

"""Multi-objective angular/sectorial clustering + OR-Tools TSP per cluster.

Algorithm by Sara Ruaga (branch `Sara`, PR #6), adapted to the unified
benchmark harness. Children are scored by a weighted combination of their
angular position and radial distance relative to the school, swept into
capacity-bounded sectors, then refined with a capacity-constrained min-cost
flow assignment around per-cluster medoids. Each resulting cluster is ordered
into a route with the shared OR-Tools TSP solver (reused from
``kmedoids_ortools``), so route costs are directly comparable across algorithms.

Only the clustering core is ported here; the original module's standalone
visualization/CLI helpers are dropped in favour of the harness's uniform
plotter and runner.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..core.scenario import Scenario
from ..core.solution import Solution
from ..core.solver import Solver
from .kmedoids_ortools import solve_tsp


@dataclass
class ClusteringConfig:
    """Configuration for the sectorial clustering algorithm."""
    capacity: int                       # max children per cluster (= bus capacity)
    alpha: float = 0.6                  # weight of the angular component [0, 1]
    beta: float = 0.4                   # weight of the radial component [0, 1]
    distance_normalization: str = "minmax"   # 'minmax' or 'zscore'
    max_iter: int = 5                   # medoid/flow refinement iterations
    sector_penalty: int = 0             # cost added when flow breaks the initial sweep (>=0)

    def __post_init__(self):
        if abs(self.alpha + self.beta - 1.0) >= 1e-6:
            raise ValueError("alpha + beta must equal 1.0")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative")
        if self.sector_penalty < 0:
            raise ValueError("sector_penalty must be non-negative")


class GeographicClusterer:
    """Angular + radial multi-objective clustering with capacity constraints.

    Operates in the caller's index space: ``school_idx`` indexes the depot in
    ``coords``/``distance_matrix`` and is excluded from every cluster
    (``labels_[school_idx] == -1``).
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.node_ids = None
        self.coords = None
        self.school_idx = None
        self.distance_matrix = None
        self.distances_to_school = None
        self.labels_ = None
        self.n_clusters_ = None

    def fit(self, node_ids, coords, school_idx, distance_matrix,
            distances_to_school) -> "GeographicClusterer":
        self.node_ids = node_ids
        self.coords = coords
        self.school_idx = school_idx
        self.distance_matrix = distance_matrix
        self.distances_to_school = distances_to_school

        # 1. Angular position + normalized radial distance per node.
        angles = self._compute_angles()
        normalized_distances = self._normalize_distances()

        # 2. Multi-objective score → initial capacity-bounded angular sweep.
        scores = self._compute_multiobjective_score(angles, normalized_distances)
        initial_labels = self._angular_sweep_assignment(angles, scores)

        n_nodes = len(node_ids) - 1  # exclude school
        k = int(np.ceil(n_nodes / self.config.capacity))

        # 3. Lloyd-style loop: capacity min-cost-flow assignment + medoid recompute.
        medoids = self._recompute_medoids(initial_labels, k)
        labels = initial_labels.copy()
        for _ in range(self.config.max_iter):
            labels = self._assign_with_min_cost_flow(medoids, initial_labels)
            new_medoids = self._recompute_medoids(labels, k)
            if new_medoids == medoids:
                break
            medoids = new_medoids

        self.labels_ = labels
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
        return self

    def _compute_angles(self) -> np.ndarray:
        """Angle (rad, [-pi, pi]) of each node relative to the school."""
        school_coords = self.coords[self.school_idx]
        vectors = self.coords - school_coords
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles[self.school_idx] = 0.0
        return angles

    def _normalize_distances(self) -> np.ndarray:
        """Normalize school distances to ~[0, 1]."""
        distances = self.distances_to_school.copy()
        distances[self.school_idx] = 0.0

        if self.config.distance_normalization == "minmax":
            d_min = distances.min()
            d_max = distances.max()
            if d_max - d_min > 0:
                return (distances - d_min) / (d_max - d_min)
            return np.zeros_like(distances)

        if self.config.distance_normalization == "zscore":
            mean = distances.mean()
            std = distances.std()
            if std > 0:
                normalized = (distances - mean) / std
                return (normalized - normalized.min()) / (normalized.max() - normalized.min())
            return np.zeros_like(distances)

        raise ValueError(
            f"Unknown normalization method: {self.config.distance_normalization}"
        )

    def _compute_multiobjective_score(self, angles: np.ndarray,
                                      normalized_distances: np.ndarray) -> np.ndarray:
        """Score(i) = alpha * (theta_i + pi)/(2pi) + beta * d_norm(i)."""
        alpha = self.config.alpha
        beta = self.config.beta
        angular_component = np.arctan2(np.sin(angles), np.cos(angles))
        angular_component = (angular_component + np.pi) / (2 * np.pi)
        return alpha * angular_component + beta * normalized_distances

    def _angular_sweep_assignment(self, angles: np.ndarray,
                                  scores: np.ndarray) -> np.ndarray:
        """Sort by score, sweep into clusters of at most ``capacity`` nodes."""
        n_nodes = len(scores)
        labels = np.full(n_nodes, -1, dtype=int)

        active_nodes = np.arange(n_nodes)
        active_nodes = active_nodes[active_nodes != self.school_idx]
        sorted_indices = active_nodes[np.argsort(scores[active_nodes])]

        current_cluster = 0
        current_count = 0
        for idx in sorted_indices:
            if current_count >= self.config.capacity:
                current_cluster += 1
                current_count = 0
            labels[idx] = current_cluster
            current_count += 1

        labels[self.school_idx] = -1
        return labels

    def _recompute_medoids(self, labels: np.ndarray, k: int) -> list[int]:
        """Pick each cluster's medoid by blending internal cost and school distance."""
        medoids = []
        lambda_acc = 0.5
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            sub_dist_matrix = self.distance_matrix[np.ix_(members, members)]
            dist_to_school = self.distances_to_school[members]
            internal_costs = sub_dist_matrix.sum(axis=1)
            combined_score = (1 - lambda_acc) * internal_costs + lambda_acc * dist_to_school
            best_idx = members[int(np.argmin(combined_score))]
            medoids.append(int(best_idx))
        return medoids

    def _assign_with_min_cost_flow(self, medoid_indices, initial_labels=None) -> np.ndarray:
        """Assign nodes to medoids respecting capacity via min-cost flow.

        When ``sector_penalty > 0`` and ``initial_labels`` is given, breaking the
        initial angular sweep is penalized, biasing the solution toward compact sectors.
        """
        n = len(self.coords)
        capacity = self.config.capacity
        k = len(medoid_indices)

        G = nx.DiGraph()
        for i in range(n):
            if i == self.school_idx:
                continue
            G.add_edge("s", ("n", i), capacity=1, weight=0)
        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx, m in enumerate(medoid_indices):
                cost = int(self.distance_matrix[i, m])
                if initial_labels is not None and self.config.sector_penalty > 0:
                    if initial_labels[i] != c_idx:
                        cost += int(self.config.sector_penalty)
                G.add_edge(("n", i), ("c", c_idx), capacity=1, weight=cost)
        for c_idx in range(k):
            G.add_edge(("c", c_idx), "t", capacity=capacity, weight=0)

        G.nodes["s"]["demand"] = -(n - 1)  # all nodes except school
        G.nodes["t"]["demand"] = (n - 1)

        flow = nx.min_cost_flow(G)

        labels = np.full(n, -1, dtype=int)
        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx in range(k):
                if flow[("n", i)].get(("c", c_idx), 0) == 1:
                    labels[i] = c_idx
                    break
        return labels


class SectorialSolver(Solver):
    name = "sectorial"

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, max_iter: int = 5,
                 sector_penalty: int = 0, tsp_time_limit_s: int = 5,
                 distance_normalization: str = "minmax"):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.sector_penalty = sector_penalty
        self.tsp_time_limit_s = tsp_time_limit_s
        self.distance_normalization = distance_normalization

    def solve(self, scenario: Scenario) -> Solution:
        coords = np.column_stack([scenario.x, scenario.y])
        full_D = scenario.dist_matrix
        origin = scenario.origin_index

        config = ClusteringConfig(
            capacity=scenario.bus_capacity,
            alpha=self.alpha,
            beta=self.beta,
            distance_normalization=self.distance_normalization,
            max_iter=self.max_iter,
            sector_penalty=self.sector_penalty,
        )
        clusterer = GeographicClusterer(config).fit(
            node_ids=list(scenario.node_ids),
            coords=coords,
            school_idx=origin,
            distance_matrix=full_D,
            distances_to_school=full_D[origin, :],
        )

        labels_full = clusterer.labels_                      # length n+1, school = -1
        cluster_labels = labels_full[scenario.children_idx]  # children-only order

        # Per-cluster TSP on the directed full distance matrix (matches kmedoids).
        routes: list[list[int]] = []
        for c in np.unique(labels_full[labels_full >= 0]):
            member_idx = np.where(labels_full == c)[0]       # scenario indices (children)
            if len(member_idx) == 1:
                # Singleton: solve_tsp's n<=2 shortcut would yield [child, depot];
                # build the depot-anchored round trip explicitly to honour the
                # Solution.routes contract ([depot, c1, ..., depot]).
                routes.append([origin, int(member_idx[0]), origin])
                continue
            sub_indices = np.append(member_idx, origin)
            depot_pos = len(sub_indices) - 1
            sub = full_D[np.ix_(sub_indices, sub_indices)]
            local = solve_tsp(sub, depot_pos, self.tsp_time_limit_s)
            routes.append([int(sub_indices[i]) for i in local])

        return Solution(
            routes=routes,
            cluster_labels=cluster_labels.astype(int),
            extra={"k": int(clusterer.n_clusters_),
                   "alpha": self.alpha, "beta": self.beta},
        )

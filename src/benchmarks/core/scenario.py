from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class ZonePolygon:
    name: str
    color: str
    members: list[str]
    geometry: Any
    centroid: tuple[float, float]


@dataclass
class Scenario:
    """A single benchmarking scenario.

    All adapters consume the same Scenario for a given (n_children, seed),
    so route distances and clusterings are directly comparable.
    """
    node_ids: np.ndarray            # int, length n+1, last = depot
    x: np.ndarray                   # lon, length n+1
    y: np.ndarray                   # lat, length n+1
    dist_matrix: np.ndarray         # directed road distance (m), (n+1, n+1)
    time_matrix: np.ndarray         # directed travel time (s), (n+1, n+1)
    origin_index: int               # always n in this build, but explicit
    bus_capacity: int
    graph: nx.MultiDiGraph
    n_children: int
    seed: int
    zones: list[ZonePolygon] | None = None
    node_zones: np.ndarray | None = None  # zone-name string per row, length n+1
    # Per-node demand, length n+1, depot 0. None means unit demand (one student
    # per stop), which is the SBRP case; external CVRP instances set it.
    demands: np.ndarray | None = None

    @property
    def children_idx(self) -> np.ndarray:
        return np.array([i for i in range(len(self.node_ids))
                         if i != self.origin_index], dtype=int)

    @property
    def dist_sym_children(self) -> np.ndarray:
        ci = self.children_idx
        D = self.dist_matrix[np.ix_(ci, ci)]
        return (D + D.T) / 2.0

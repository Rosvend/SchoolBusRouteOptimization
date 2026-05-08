from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    """Output of a Solver.solve(scenario) call.

    routes:           list of tours, each [depot, c1, c2, ..., depot]
                      indices reference the full Scenario index space
    cluster_labels:   length n_children, child→cluster mapping (children-only
                      order, i.e. scenario.children_idx). None when the solver
                      doesn't produce a partition (e.g. set-cover variants).
    served_indices:   set of full-scenario indices actually visited.
                      Defaults to "everyone" but Roberto's grid set-cover may
                      legitimately leave students out under relaxed coverage.
    extra:            free-form per-solver diagnostics (iterations, sub-times)
    """
    routes: list[list[int]]
    cluster_labels: np.ndarray | None = None
    served_indices: set[int] | None = None
    extra: dict = field(default_factory=dict)

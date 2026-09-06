"""Invariant checks for the ALNS solver.

The repo has no test harness, so this is the re-runnable correctness check for
the ALNS adapter. It uses synthetic asymmetric distance matrices rather than the
OSM scenario generator, so it runs in seconds and needs no cached road graph.

What it asserts, per instance:
  * routes are depot-anchored at both ends (the harness computes route load as
    len(route) - 2, so a missing endpoint silently corrupts load and bus count)
  * every customer is served exactly once
  * no route exceeds capacity
  * cluster_labels covers every child, in children-local order
  * the harness's fleet_distance equals the search's own objective, which is
    what catches an index-space or depot-placement mistake
  * the search never returns a solution worse than the one it started from

Usage:
    python -m scripts.check_alns_invariants [--budget 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarks.adapters.alns import AlnsSolver           # noqa: E402
from src.benchmarks.core.metrics import compute_all, fleet_distance  # noqa: E402
from src.benchmarks.core.scenario import Scenario             # noqa: E402


def synthetic(n: int, seed: int, capacity: int = 20) -> Scenario:
    """A scenario with directed, asymmetric distances and no road graph."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n + 1, 2)) * 20_000.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d *= 1.0 + 0.15 * rng.random((n + 1, n + 1))   # break symmetry
    np.fill_diagonal(d, 0.0)
    return Scenario(
        node_ids=np.arange(n + 1), x=pts[:, 0], y=pts[:, 1],
        dist_matrix=d, time_matrix=d / 8.0, origin_index=n,
        bus_capacity=capacity, graph=nx.MultiDiGraph(), n_children=n,
        seed=seed,
    )


def check(n: int, seed: int, budget: float) -> list[str]:
    sc = synthetic(n, seed)
    sol = AlnsSolver(tsp_time_limit_s=budget).solve(sc)
    m, e = compute_all(sc, sol), sol.extra
    fails = []

    for r in sol.routes:
        if r[0] != sc.origin_index or r[-1] != sc.origin_index:
            fails.append("route is not depot-anchored at both ends")
        if len(r) - 2 > sc.bus_capacity:
            fails.append(f"route load {len(r) - 2} exceeds capacity {sc.bus_capacity}")

    visited = [c for r in sol.routes for c in r[1:-1]]
    if sorted(visited) != list(range(n)):
        fails.append(f"coverage: {len(visited)} visits / {len(set(visited))} unique, want {n}")
    if sol.cluster_labels is None or len(sol.cluster_labels) != n:
        fails.append("cluster_labels missing or wrong length")
    elif (sol.cluster_labels < 0).any():
        fails.append("some child left unlabelled")
    if abs(fleet_distance(sc, sol) - e["best_objective_m"]) > 1e-6:
        fails.append(f"objective mismatch: harness {fleet_distance(sc, sol):.4f} "
                     f"vs search {e['best_objective_m']:.4f}")
    if e["best_objective_m"] > e["init_objective_m"] + 1e-9:
        fails.append("search returned a solution worse than its initial one")
    if m["capacity_violations"]:
        fails.append(f"{m['capacity_violations']} capacity violations")
    if m["coverage"] != 1.0:
        fails.append(f"coverage {m['coverage']}")

    status = "FAIL" if fails else "ok"
    print(f"  n={n:<4} seed={seed} k={e['k']:<3} {status:4} | "
          f"init {e['init_objective_m']/1000:7.1f} -> {e['best_objective_m']/1000:7.1f} km "
          f"({e['improvement_pct']:5.2f}%) | {e['iterations']:6d} iters | "
          f"buses {m['buses_used']:2d}")
    for f in fails:
        print(f"       -> {f}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2.0,
                    help="per-bus wall-clock seconds (total = k * budget)")
    args = ap.parse_args()

    # n=19/20/21 straddle the capacity boundary, where k flips from 1 to 2 and
    # single-stop routes appear.
    cases = [(1, 1), (2, 1), (19, 1), (20, 1), (21, 1), (50, 1), (50, 2), (100, 1)]
    print(f"ALNS invariants ({len(cases)} cases, {args.budget:g}s per bus)")
    failed = sum(bool(check(n, seed, args.budget)) for n, seed in cases)
    print("ALL CHECKS PASS" if not failed else f"{failed}/{len(cases)} CASES FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

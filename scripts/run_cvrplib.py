"""Calibrate the ALNS solver against published CVRP optima (Augerat set A).

The Medellin benchmark establishes *relative* standings between our own
implementations. This script adds the external reference point: the same ALNS
solver, unchanged, run on classical instances whose optimal values are known,
so its absolute solution quality can be stated rather than assumed.

Usage:
    python -m scripts.run_cvrplib [--budget 60] [--out results/cvrplib.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarks.adapters.alns import AlnsSolver          # noqa: E402
from src.benchmarks.core.metrics import compute_all          # noqa: E402
from src.benchmarks.data.cvrplib_loader import load_directory  # noqa: E402

INSTANCE_DIR = Path("src/benchmarks/data/cvrplib")
COLS = ["instance", "n", "capacity", "trucks_declared", "routes_used",
        "optimal", "alns_cost", "gap_pct", "latency_s", "iterations",
        "coverage", "capacity_violations"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, default=INSTANCE_DIR)
    ap.add_argument("--budget", type=float, default=60.0,
                    help="ALNS wall-clock seconds per instance")
    ap.add_argument("--out", type=Path, default=Path("results/cvrplib.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    instances = load_directory(args.instances)
    if not instances:
        print(f"ERROR: no .vrp files in {args.instances}")
        return 2
    print(f"{len(instances)} instances, {args.budget:g}s each "
          f"(~{len(instances) * args.budget / 60:.0f} min)\n")

    rows = []
    for inst in instances:
        solver = AlnsSolver(time_budget_s=args.budget, random_state=args.seed)
        t0 = time.perf_counter()
        sol = solver.solve(inst.scenario)
        latency = time.perf_counter() - t0

        m = compute_all(inst.scenario, sol)
        cost = m["fleet_distance_m"]
        gap = (100.0 * (cost - inst.optimal) / inst.optimal
               if inst.optimal else float("nan"))
        rows.append({
            "instance": inst.name, "n": inst.scenario.n_children,
            "capacity": inst.capacity, "trucks_declared": inst.n_trucks,
            "routes_used": m["buses_used"], "optimal": inst.optimal,
            "alns_cost": cost, "gap_pct": gap, "latency_s": latency,
            "iterations": sol.extra["iterations"], "coverage": m["coverage"],
            "capacity_violations": m["capacity_violations"],
        })
        flag = "" if m["capacity_violations"] == 0 and m["coverage"] == 1.0 else "  <-- INFEASIBLE"
        print(f"  {inst.name:<12} n={inst.scenario.n_children:<3} "
              f"opt={inst.optimal:>7.0f}  alns={cost:>7.0f}  "
              f"gap={gap:+6.2f}%  routes={m['buses_used']}/{inst.n_trucks}"
              f"  iters={sol.extra['iterations']:>6}{flag}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    gaps = [r["gap_pct"] for r in rows]
    optimal_hits = sum(1 for g in gaps if g < 1e-9)
    print(f"\nmean gap   : {st.mean(gaps):+.3f}%")
    print(f"median gap : {st.median(gaps):+.3f}%")
    print(f"max gap    : {max(gaps):+.3f}%   ({rows[gaps.index(max(gaps))]['instance']})")
    print(f"optimal    : {optimal_hits}/{len(rows)} instances solved to proven optimality")
    print(f"feasible   : {sum(1 for r in rows if r['capacity_violations'] == 0 and r['coverage'] == 1.0)}/{len(rows)}")
    print(f"\nwrote {args.out}")

    print("\n% ---- Table: ALNS vs published optima, Augerat set A ----")
    print(f"% budget {args.budget:g}s per instance, seed {args.seed}")
    for r in rows:
        print(f"{r['instance']} & {r['n']} & {r['trucks_declared']} & "
              f"{r['optimal']:.0f} & {r['alns_cost']:.0f} & {r['gap_pct']:+.2f} \\\\")
    print(f"\\midrule\nMean & & & & & {st.mean(gaps):+.2f} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

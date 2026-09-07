"""Emit the manuscript's result-table bodies as LaTeX, straight from runs.csv.

Generating the rows mechanically removes the risk of a table drifting from the
data through hand transcription. Bolding follows the manuscript's convention:
the distance minimum among full-coverage methods, and the best silhouette /
shortest max route.

Usage:
    python -m scripts.emit_tables [--runs results/runs.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ORDER = [
    ("kmedoids_ortools", "K-medoids + TSP"),
    ("sectorial", "Sectorial"),
    ("cvrptw_zones", "CVRPTW (zonal)$^\\ast$"),
    ("setcover_perchild", "Set Cover (per-child)"),
    ("setcover_grid", "Set Cover (grid)"),
    ("alns", "ALNS"),
]
# CVRPTW is excluded from distance bolding: it may leave students unserved.
FULL_COVERAGE = {"kmedoids_ortools", "sectorial", "setcover_perchild",
                 "setcover_grid", "alns"}


def load(path: Path):
    return [r for r in csv.DictReader(path.open()) if r["status"] == "ok"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    args = ap.parse_args()
    rows = load(args.runs)
    Ns = sorted({int(r["N"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    def v(algo, N, col):
        return [float(r[col]) for r in rows
                if r["algo"] == algo and int(r["N"]) == N]

    def mean(algo, N, col):
        return st.mean(v(algo, N, col))

    print(f"% Generated from {args.runs}: {len(seeds)} seeds "
          f"({len(Ns) * len(seeds)} blocks, {len(rows)} runs)")
    print(f"% densities={Ns} seeds={seeds}\n")

    # ---- combined distance + latency table ----
    best_dist = {N: min(mean(a, N, "fleet_distance_m")
                        for a in FULL_COVERAGE) for N in Ns}
    print("% ---- Table: fleet distance (km) | latency (s) ----")
    for algo, label in ORDER:
        cells = []
        for N in Ns:
            vals = v(algo, N, "fleet_distance_m")
            m = st.mean(vals) / 1000
            sd = (st.stdev(vals) / 1000) if len(vals) > 1 else 0.0
            cell = f"{m:.0f}$\\pm${sd:.0f}"
            if algo in FULL_COVERAGE and \
               abs(st.mean(vals) - best_dist[N]) < 1e-9:
                cell = f"\\textbf{{{m:.0f}$\\pm${sd:.0f}}}"
            cells.append(cell)
        # Match the manuscript's precision: 2 decimals below 10 s, 1 above.
        lat = [f"{mean(algo, N, 'latency_s'):.2f}"
               if mean(algo, N, "latency_s") < 10
               else f"{mean(algo, N, 'latency_s'):.1f}" for N in Ns]
        print(f"{label:24s} & " + " & ".join(f"{c:>18s}" for c in cells)
              + "\n  & " + " & ".join(lat) + "\\\\")

    # ---- quality table at the largest density ----
    N = max(Ns)
    best_sil = max(mean(a, N, "silhouette") for a, _ in ORDER)
    best_max = min(mean(a, N, "max_route_distance_m") for a, _ in ORDER)
    print(f"\n% ---- Table: quality metrics at N={N} ----")
    for algo, label in ORDER:
        label = label.replace("$^\\ast$", "")
        sil = mean(algo, N, "silhouette")
        mx = mean(algo, N, "max_route_distance_m") / 1000
        sil_s = f"\\textbf{{{sil:.3f}}}" if abs(sil - best_sil) < 1e-9 else (
            f"${sil:.3f}$" if sil < 0 else f"{sil:.3f}")
        mx_s = f"\\textbf{{{mx:.1f}}}" if abs(
            mean(algo, N, "max_route_distance_m") - best_max) < 1e-9 else f"{mx:.1f}"
        print(f"{label:22s} & {mean(algo, N, 'buses_used'):.1f} & "
              f"{mean(algo, N, 'coverage'):.3f} & {sil_s} & {mx_s}\\\\")

    # ---- prose numbers that appear in the running text ----
    print("\n% ---- numbers quoted in the text ----")
    dists = {a: mean(a, N, "fleet_distance_m") / 1000 for a, _ in ORDER}
    worst = max(dists.values()); best = min(dists.values())
    print(f"%   VKT at N={N}: worst {worst:.0f} km vs best {best:.0f} km "
          f"= {worst / best:.2f}x")
    cv = mean("cvrptw_zones", N, "max_route_distance_m")
    others = [mean(a, N, "max_route_distance_m") for a, _ in ORDER
              if a != "cvrptw_zones"]
    print(f"%   CVRPTW max route is {100 * (1 - cv / max(others)):.0f}-"
          f"{100 * (1 - cv / min(others)):.0f}% shorter than the others")
    pc = [float(r["latency_s"]) for r in rows if r["algo"] == "setcover_perchild"]
    print(f"%   set cover (child) latency: max over all runs = {max(pc):.3f} s")
    g = v("setcover_grid", N, "latency_s")
    print(f"%   set cover (grid) at N={N}: {st.mean(g):.1f}"
          f"$\\pm${st.stdev(g) if len(g) > 1 else 0:.1f} s")
    b = mean("setcover_perchild", N, "buses_used")
    print(f"%   set cover (child): {b:.1f} routes x cap 20 = {b * 20:.0f} seats "
          f"for {N} students -> {100 * (b * 20 - N) / (b * 20):.0f}% redundant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

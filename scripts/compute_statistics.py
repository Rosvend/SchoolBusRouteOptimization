"""Reproduce every statistic reported in the paper from results/runs.csv.

Emits the Friedman test, the mean ranks, the Wilcoxon signed-rank test between
the two leading strategies, and the Nemenyi post-hoc critical difference, for
both fleet distance and latency.

Usage:
    python -m scripts.compute_statistics [--runs results/runs.csv]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Display order used in the paper's tables.
STRATEGY_ORDER = [
    ("kmedoids_ortools", "K-medoids + TSP"),
    ("sectorial", "Sectorial"),
    ("cvrptw_zones", "CVRPTW (zonal)"),
    ("setcover_grid", "Set Cover (grid)"),
    ("alns", "ALNS"),
    ("setcover_perchild", "Set Cover (per-child)"),
]

# Studentized range statistic q_alpha at alpha=0.05 for infinite df, indexed by
# the number of compared methods k (Demsar 2006, Table 5).
Q_ALPHA_05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
              7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load(runs_csv: Path):
    """Return (blocks, {algo: {block: value}}) for each metric."""
    rows = [r for r in csv.DictReader(runs_csv.open()) if r["status"] == "ok"]
    if not rows:
        raise SystemExit(f"no successful runs found in {runs_csv}")
    blocks = sorted({(int(r["N"]), int(r["seed"])) for r in rows})
    data: dict[str, dict] = defaultdict(dict)
    for r in rows:
        key = (int(r["N"]), int(r["seed"]))
        data["fleet_distance_m"][(r["algo"], key)] = float(r["fleet_distance_m"])
        data["latency_s"][(r["algo"], key)] = float(r["latency_s"])
    return blocks, data


def matrix(data, metric, blocks, algos):
    """blocks x algos matrix of the metric (lower is better for both metrics)."""
    return np.array([[data[metric][(a, b)] for a in algos] for b in blocks])


def analyse(label, M, algos, names):
    k = len(algos)
    n_blocks = M.shape[0]

    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.array([stats.rankdata(M[i, :]) for i in range(n_blocks)])
    mean_ranks = ranks.mean(axis=0)

    cd = Q_ALPHA_05[k] * np.sqrt(k * (k + 1) / (6 * n_blocks))

    print(f"\n=== {label} ===")
    print(f"Friedman: chi2={chi2:.4f}  df={k - 1}  p={p:.4e}  "
          f"(blocks={n_blocks}, strategies={k})")
    print("\nMean ranks (1 = best):")
    for name, r in sorted(zip(names, mean_ranks), key=lambda t: t[1]):
        print(f"  {name:24s} {r:.4f}")

    print(f"\nNemenyi post-hoc: q_0.05={Q_ALPHA_05[k]}, CD={cd:.4f}")
    sig, not_sig = [], []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            bucket = sig if diff > cd else not_sig
            bucket.append((names[i], names[j], diff))

    print(f"  significant pairs: {len(sig)} of {k * (k - 1) // 2}")
    for a, b, d in sorted(sig, key=lambda t: -t[2]):
        print(f"    {a} vs {b}: delta={d:.3f}")
    print("  not separable:")
    for a, b, d in sorted(not_sig, key=lambda t: -t[2]):
        print(f"    {a} vs {b}: delta={d:.3f}")
    return mean_ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    args = ap.parse_args()

    blocks, data = load(args.runs)
    algos = [a for a, _ in STRATEGY_ORDER]
    names = [n for _, n in STRATEGY_ORDER]

    M_dist = matrix(data, "fleet_distance_m", blocks, algos)
    analyse("FLEET DISTANCE", M_dist, algos, names)
    analyse("LATENCY", matrix(data, "latency_s", blocks, algos), algos, names)

    # Pairwise Wilcoxon between the two leaders on fleet distance.
    i, j = algos.index("kmedoids_ortools"), algos.index("sectorial")
    w, p = stats.wilcoxon(M_dist[:, i], M_dist[:, j])
    print(f"\n=== WILCOXON (fleet distance) ===")
    print(f"K-medoids vs Sectorial over {len(blocks)} blocks: W={w:.1f}, p={p:.4f}")

    # Per-density means, as reported in the results tables.
    print("\n=== MEAN FLEET DISTANCE (km) AND LATENCY (s) BY DENSITY ===")
    densities = sorted({N for N, _ in blocks})
    for algo, name in STRATEGY_ORDER:
        cells = []
        for N in densities:
            vals = [data["fleet_distance_m"][(algo, b)] for b in blocks if b[0] == N]
            lat = [data["latency_s"][(algo, b)] for b in blocks if b[0] == N]
            cells.append(f"N={N}: {np.mean(vals) / 1000:7.1f}+-{np.std(vals, ddof=1) / 1000:4.1f} km "
                         f"/ {np.mean(lat):7.2f} s")
        print(f"  {name:24s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

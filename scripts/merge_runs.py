"""Merge per-seed benchmark output directories into a single runs.csv.

The harness writes only its own rows to <output-dir>/runs.csv, so extending the
sweep with additional seeds means several output directories. This collects them
into one file, refusing to merge an unbalanced design by default.

Usage:
    python -m scripts.merge_runs results/runs.csv results/extra_seeds/seed*/runs.csv \
        -o results/runs_merged.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

COLS = ["algo", "N", "seed", "status", "fleet_distance_m", "latency_s",
        "buses_used", "coverage", "silhouette", "capacity_violations",
        "max_route_distance_m", "mean_route_load", "error_msg"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--allow-unbalanced", action="store_true",
                    help="merge even if some (algo, N) cells have unequal seed counts")
    args = ap.parse_args()

    rows: list[dict] = []
    seen: set[tuple] = set()
    for path in args.inputs:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        n_before = len(rows)
        for r in csv.DictReader(path.open()):
            key = (r["algo"], r["N"], r["seed"])
            if key in seen:
                print(f"  duplicate {key} in {path} - keeping first occurrence")
                continue
            seen.add(key)
            rows.append(r)
        print(f"  + {len(rows) - n_before:3d} rows from {path}")

    ok = [r for r in rows if r["status"] == "ok"]
    failed = len(rows) - len(ok)
    algos = sorted({r["algo"] for r in ok})
    Ns = sorted({int(r["N"]) for r in ok})
    seeds = sorted({int(r["seed"]) for r in ok})

    # A Friedman/Nemenyi design needs every strategy present in every block.
    cell = Counter((r["algo"], int(r["N"])) for r in ok)
    counts = set(cell.values())
    complete_seeds = []
    for s in seeds:
        if all(any(r["algo"] == a and int(r["N"]) == N and int(r["seed"]) == s
                   for r in ok) for a in algos for N in Ns):
            complete_seeds.append(s)

    print(f"\n  total rows      : {len(rows)} ({len(ok)} ok, {failed} failed)")
    print(f"  strategies      : {len(algos)} {algos}")
    print(f"  densities       : {Ns}")
    print(f"  seeds present   : {seeds}")
    print(f"  seeds COMPLETE  : {complete_seeds}  -> {len(Ns)*len(complete_seeds)} usable blocks")
    if len(counts) != 1:
        print(f"  !! UNBALANCED: per-(algo,N) seed counts = {sorted(counts)}")
        if not args.allow_unbalanced:
            print("  refusing to write; pass --allow-unbalanced to override, or")
            print("  restrict to the complete seeds listed above.")
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (int(r["N"]), int(r["seed"]), r["algo"])):
            w.writerow({k: r.get(k, "") for k in COLS})
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

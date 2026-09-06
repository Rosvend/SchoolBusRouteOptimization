"""Regenerate the qualitative route-map figure with fewer, larger panels.

The six-panel version is included at ~0.55\\textwidth, which leaves each panel
under an inch wide -- too small to actually show the structural contrast its
caption claims. Three panels in a single row occupy the same panel size as six
panels at full text width, in half the vertical space, so this plots one
representative of each solution paradigm:

    Sectorial          - clustering        (compact wedges radiating from the depot)
    Set Cover (grid)   - covering ILP      (overlapping, capacity-filled routes)
    ALNS               - destroy-repair search (locality-aware metaheuristic)

Usage:
    python -m scripts.plot_qualitative_figure [--n 200] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarks.adapters.sectorial import SectorialSolver
from src.benchmarks.adapters.setcover_grid import SetCoverGridSolver
from src.benchmarks.adapters.alns import AlnsSolver
from src.benchmarks.data.scenario_generator import make_scenario
from src.benchmarks.viz.plotter import plot_comparison_grid

PANELS = [
    ("Sectorial (clustering)", SectorialSolver),
    ("Set Cover, grid (covering ILP)", SetCoverGridSolver),
    ("ALNS (destroy-repair search)", AlnsSolver),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/latex/figures/comparison_row_N200.png"))
    args = ap.parse_args()

    print(f"building scenario N={args.n} seed={args.seed} ...")
    scenario = make_scenario(args.n, args.seed)

    solutions = {}
    for label, cls in PANELS:
        print(f"  solving {label} ...", flush=True)
        solutions[label] = cls().solve(scenario)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison_grid(scenario, solutions, out_path=args.out,
                         draw_roads=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

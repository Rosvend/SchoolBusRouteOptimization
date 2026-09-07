"""Unified benchmark runner.

Runs every registered Solver on every (n_children, seed) scenario,
records metrics + latency to CSV, renders per-scenario maps and a
comparison grid, and aggregates a paper-ready summary table.

Usage (from repo root):
    python -m src.benchmarks.run_benchmark
    python -m src.benchmarks.run_benchmark --densities 50 --seeds 1 \
        --algos kmedoids_ortools alns
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .adapters.alns import AlnsSolver
from .adapters.cvrptw_zones import CVRPTWZonesSolver
from .adapters.genetic import GeneticSolver
from .adapters.kmedoids_ortools import KmedoidsORToolsSolver
from .adapters.sectorial import SectorialSolver
from .adapters.setcover_grid import SetCoverGridSolver
from .adapters.setcover_perchild import SetCoverPerChildSolver
from .core.metrics import compute_all
from .core.scenario import Scenario
from .core.solver import Solver
from .data.scenario_generator import make_scenario
from .viz.plotter import plot_comparison_grid, plot_single


SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "kmedoids_ortools":  KmedoidsORToolsSolver,
    "sectorial":         SectorialSolver,
    "cvrptw_zones":      CVRPTWZonesSolver,
    "setcover_perchild": SetCoverPerChildSolver,
    "setcover_grid":     SetCoverGridSolver,
    "alns":              AlnsSolver,
    # Superseded by ALNS as the paper's metaheuristic baseline. Kept registered
    # so the historical `genetic` rows in results/ stay reproducible, but out of
    # the default sweep.
    "genetic":           GeneticSolver,
}

# The six strategies the paper reports. `--algos` defaults to these rather than
# to the whole registry.
PAPER_ALGOS = ["kmedoids_ortools", "sectorial", "cvrptw_zones",
               "setcover_perchild", "setcover_grid", "alns"]

DEFAULT_DENSITIES = [50, 100, 200, 400]
DEFAULT_SEEDS = [1, 2, 3]
CSV_COLS = ["algo", "N", "seed", "status",
            "fleet_distance_m", "latency_s", "buses_used", "coverage",
            "silhouette", "capacity_violations",
            "max_route_distance_m", "mean_route_load", "error_msg"]


def _run_one(solver: Solver, scenario: Scenario) -> tuple[dict, object | None]:
    t0 = time.perf_counter()
    try:
        sol = solver.solve(scenario)
    except Exception as e:
        latency = time.perf_counter() - t0
        return {
            "algo": solver.name, "N": scenario.n_children, "seed": scenario.seed,
            "status": "error", "fleet_distance_m": float("nan"),
            "latency_s": latency, "buses_used": 0, "coverage": 0.0,
            "silhouette": float("nan"), "capacity_violations": 0,
            "max_route_distance_m": float("nan"), "mean_route_load": 0.0,
            "error_msg": f"{type(e).__name__}: {e}",
        }, None
    latency = time.perf_counter() - t0
    metrics = compute_all(scenario, sol)
    row = {
        "algo": solver.name, "N": scenario.n_children, "seed": scenario.seed,
        "status": "ok", "latency_s": latency, "error_msg": "",
        **metrics,
    }
    return row, sol


def _write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})


def _md_pivot(df_ok: pd.DataFrame, value: str, fmt: str = ".2f",
              with_std: bool = False) -> str:
    """Render a small markdown table without external `tabulate`."""
    if df_ok.empty:
        return "_no data_"
    if with_std:
        mean = df_ok.pivot_table(index="algo", columns="N", values=value, aggfunc="mean")
        std = df_ok.pivot_table(index="algo", columns="N", values=value, aggfunc="std")
        cols = sorted(mean.columns)
        lines = ["| algo | " + " | ".join(f"N={c}" for c in cols) + " |",
                 "|" + "---|" * (len(cols) + 1)]
        for algo in mean.index:
            cells = []
            for c in cols:
                m = mean.loc[algo, c]
                # std can be missing entirely if only 1 sample per cell
                s = std.loc[algo, c] if (algo in std.index and c in std.columns) else float("nan")
                if pd.isna(m):
                    cells.append("")
                elif pd.isna(s):
                    cells.append(f"{m:{fmt}}")
                else:
                    cells.append(f"{m:{fmt}} ± {s:{fmt}}")
            lines.append(f"| {algo} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    else:
        pv = df_ok.pivot_table(index="algo", columns="N", values=value, aggfunc="mean")
        cols = sorted(pv.columns)
        lines = ["| algo | " + " | ".join(f"N={c}" for c in cols) + " |",
                 "|" + "---|" * (len(cols) + 1)]
        for algo in pv.index:
            cells = [("" if pd.isna(pv.loc[algo, c]) else f"{pv.loc[algo, c]:{fmt}}")
                     for c in cols]
            lines.append(f"| {algo} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _write_summary(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"]
    if df_ok.empty:
        path.write_text("# No successful runs.\n")
        return

    out = ["# Benchmark Summary", "",
           "## Fleet distance (m) — mean ± std", "",
           _md_pivot(df_ok, "fleet_distance_m", ".0f", with_std=True),
           "", "## Latency (s) — mean ± std", "",
           _md_pivot(df_ok, "latency_s", ".2f", with_std=True),
           "", "## Buses used — mean", "",
           _md_pivot(df_ok, "buses_used", ".1f"),
           "", "## Coverage — mean", "",
           _md_pivot(df_ok, "coverage", ".3f"),
           "", "## Silhouette — mean", "",
           _md_pivot(df_ok, "silhouette", ".3f")]

    if (df["status"] == "error").any():
        errs = df[df["status"] == "error"][["algo", "N", "seed", "error_msg"]]
        out += ["", "## Errors", ""]
        out.append("| algo | N | seed | error |")
        out.append("|---|---|---|---|")
        for _, r in errs.iterrows():
            out.append(f"| {r['algo']} | {r['N']} | {r['seed']} | "
                       f"{str(r['error_msg'])[:200]} |")

    path.write_text("\n".join(out) + "\n")


def _scalability_plot(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"]
    if df_ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for algo, sub in df_ok.groupby("algo"):
        agg = sub.groupby("N")["latency_s"].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=algo)
    ax.set_xlabel("N (students)")
    ax.set_ylabel("Mean latency (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Scalability: latency vs problem size")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--densities", nargs="+", type=int, default=DEFAULT_DENSITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--algos", nargs="+", default=list(PAPER_ALGOS))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-grid", action="store_true",
                   help="Skip per-scenario comparison grid (still saves single-algo maps)")
    p.add_argument("--draw-roads", action="store_true",
                   help="Render route polylines along actual road shortest paths (slower)")
    args = p.parse_args(argv)

    out = args.output_dir
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Validate algo names
    unknown = [a for a in args.algos if a not in SOLVER_REGISTRY]
    if unknown:
        print(f"ERROR: unknown algos: {unknown}. Available: {list(SOLVER_REGISTRY)}")
        return 2

    rows: list[dict] = []
    n_total = len(args.densities) * len(args.seeds)
    scenario_idx = 0

    for N in args.densities:
        for seed in args.seeds:
            scenario_idx += 1
            print(f"\n=== Scenario {scenario_idx}/{n_total}: N={N}, seed={seed} ===")
            scenario = make_scenario(N, seed)

            solutions_by_name: dict[str, object] = {}
            for algo_name in args.algos:
                solver = SOLVER_REGISTRY[algo_name]()
                print(f"  [{algo_name}] solving …")
                row, sol = _run_one(solver, scenario)
                rows.append(row)
                if row["status"] == "ok":
                    print(f"    ok | dist={row['fleet_distance_m']:.0f}m "
                          f"buses={row['buses_used']} "
                          f"cov={row['coverage']:.2f} "
                          f"sil={row['silhouette']:.3f} "
                          f"lat={row['latency_s']:.2f}s")
                    if not args.no_plots:
                        plot_single(
                            scenario, sol,
                            title=f"{algo_name}  |  N={N}, seed={seed}",
                            out_path=plots_dir / f"N{N}_s{seed}_{algo_name}.png",
                            draw_roads=args.draw_roads,
                        )
                    solutions_by_name[algo_name] = sol
                else:
                    print(f"    ERROR: {row['error_msg']}")
                    traceback.print_exc()

            if not args.no_plots and not args.no_grid and solutions_by_name:
                plot_comparison_grid(
                    scenario, solutions_by_name,
                    out_path=plots_dir / f"N{N}_s{seed}_grid.png",
                    draw_roads=False,
                )

            # Persist incrementally so a crash mid-sweep doesn't lose data
            _write_csv(rows, out / "runs.csv")

    _write_csv(rows, out / "runs.csv")
    _write_summary(rows, out / "summary.md")
    _scalability_plot(rows, out / "scalability.png")

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDone. {n_ok}/{len(rows)} runs ok.")
    print(f"  CSV     : {out / 'runs.csv'}")
    print(f"  Summary : {out / 'summary.md'}")
    print(f"  Plots   : {plots_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

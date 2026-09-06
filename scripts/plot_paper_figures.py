"""Regenerate the vector figures used by docs/latex/optimization.tex.

Produces, from results/runs.csv only:
    docs/latex/figures/scalability.pdf   -- mean latency vs N (log y)
    docs/latex/figures/pareto_N400.pdf   -- distance/latency trade-off at N=400

Note on the Pareto plot: K-medoids and Sectorial are near-identical at N=400
(they differ by ~0.25% in distance and ~0.06% in latency), so a plain filled
marker for each makes one invisible underneath the other. Sectorial is therefore
drawn as a hollow ring on top of the filled K-medoids marker, and the overlap is
annotated with the measured gap rather than left for the reader to guess.

Usage:
    python -m scripts.plot_paper_figures [--runs results/runs.csv] [--outdir docs/latex/figures]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# The manuscript places these two figures side by side at ~0.49\textwidth, so
# they are reduced substantially in print; scale type and strokes up to stay
# legible at that size.
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.1,
})

# (csv algo key, paper label, tab10 color, marker, filled?, linestyle)
# K-medoids and Sectorial are near-identical on both metrics, so Sectorial gets
# a hollow marker and a dashed line: drawn on top of K-medoids, both stay
# readable instead of one silently masking the other.
STRATEGIES = [
    ("kmedoids_ortools",  "K-medoids+TSP",  "C0", "o", True,  "-"),
    ("sectorial",         "Sectorial",      "C1", "o", False, "--"),
    ("cvrptw_zones",      "CVRPTW-zones",   "C2", "s", True,  "-"),
    ("setcover_perchild", "SetCover-child", "C3", "^", True,  "-"),
    ("setcover_grid",     "SetCover-grid",  "C4", "v", True,  "-"),
    ("alns",              "ALNS",           "C5", "D", True,  "-"),
]


def load(runs_csv: Path):
    """{algo: {N: (mean_distance_km, mean_latency_s)}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(runs_csv.open()):
        if r["status"] != "ok":
            continue
        acc[r["algo"]][int(r["N"])].append(
            (float(r["fleet_distance_m"]) / 1000.0, float(r["latency_s"]))
        )
    return {
        algo: {
            N: (float(np.mean([v[0] for v in vals])),
                float(np.mean([v[1] for v in vals])))
            for N, vals in by_n.items()
        }
        for algo, by_n in acc.items()
    }


def plot_scalability(data, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, (algo, label, color, marker, filled, ls) in enumerate(STRATEGIES):
        if algo not in data:
            continue
        ns = sorted(data[algo])
        ax.plot(
            ns, [data[algo][N][1] for N in ns],
            marker=marker, color=color, label=label, linestyle=ls,
            markersize=10 if not filled else 8,
            linewidth=2.6 if i == 0 else 1.6,
            markerfacecolor=color if filled else "none",
            markeredgecolor=color, markeredgewidth=1.8,
            zorder=3 + i,
        )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(sorted({N for d in data.values() for N in d}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of students $N$")
    ax.set_ylabel("Mean latency (s)")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.margins(y=0.18)
    # Legend below the axes so it cannot cover the fastest series at N=400.
    ax.legend(ncol=3, framealpha=0.95, columnspacing=1.0,
              handletextpad=0.4, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_pareto(data, out: Path, N: int = 400) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.1))

    pts = {}
    for algo, label, color, marker, filled, _ls in STRATEGIES:
        if algo not in data or N not in data[algo]:
            continue
        dist, lat = data[algo][N]
        pts[algo] = (dist, lat, label)
        # Hollow markers get a higher zorder so a coincident filled marker
        # underneath stays visible through the ring.
        ax.scatter(
            dist, lat, s=230 if filled else 320,
            facecolors=color if filled else "none",
            edgecolors="black" if filled else color,
            linewidths=0.8 if filled else 2.2,
            color=color, marker=marker, label=label,
            alpha=0.9 if filled else 1.0,
            zorder=3 if filled else 4,
        )

    ax.set_yscale("log")
    ax.set_xlabel(f"Fleet distance at $N={N}$ (km)")
    ax.set_ylabel("Mean latency (s)")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    # Pad the data limits so no marker is clipped by, or sits under, the axes
    # spines: the spread across strategies is wide in both axes.
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    xpad = 0.08 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) / 3.0, max(ys) * 3.0)

    # Annotate the K-medoids / Sectorial coincidence with the measured gap.
    if "kmedoids_ortools" in pts and "sectorial" in pts:
        kd, kl, _ = pts["kmedoids_ortools"]
        sd, sl, _ = pts["sectorial"]
        d_km = abs(kd - sd)
        d_lat = abs(kl - sl)
        d_pct = 100.0 * d_km / min(kd, sd)
        ax.annotate(
            "K-medoids and Sectorial coincide\n"
            f"($\\Delta$ = {d_km:.1f} km, {d_pct:.2f}%; "
            f"$\\Delta$ = {d_lat:.2f} s)",
            xy=(min(kd, sd), min(kl, sl)),
            xytext=(0.17, 0.66), textcoords="axes fraction",
            fontsize=11, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35",
                            shrinkA=2, shrinkB=10,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6),
        )

    ax.annotate("lower-left is better", xy=(0.02, 0.04),
                xycoords="axes fraction", fontsize=11, style="italic",
                color="0.35")
    # Legend below the axes: an in-axes legend overlaps the extreme markers.
    ax.legend(ncol=3, framealpha=0.95, columnspacing=1.0,
              handletextpad=0.4, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("docs/latex/figures"))
    args = ap.parse_args()

    data = load(args.runs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_scalability(data, args.outdir / "scalability.pdf")
    plot_pareto(data, args.outdir / "pareto_N400.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

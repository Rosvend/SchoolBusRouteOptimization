"""Consistent map renderer for all algorithms.

A single `plot_solution` is used for both per-(scenario, algo) maps and the
side-by-side comparison grid that goes into the paper. Routes are rendered
following the actual road shortest path (matches roberto's visualization)
when `draw_roads=True`.
"""
from __future__ import annotations

from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..core.scenario import Scenario
from ..core.solution import Solution


def _route_color(idx: int, n_routes: int):
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_routes, 1))
    return cmap(idx % cmap.N)


def _draw_route(ax, scenario: Scenario, route: list[int], color, *,
                follow_roads: bool, linewidth: float = 1.2):
    if follow_roads:
        G = scenario.graph
        nodes = scenario.node_ids
        for u, v in zip(route[:-1], route[1:]):
            try:
                path = nx.shortest_path(G, int(nodes[u]), int(nodes[v]), weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                ax.plot([scenario.x[u], scenario.x[v]],
                        [scenario.y[u], scenario.y[v]],
                        color=color, linewidth=linewidth, alpha=0.7)
                continue
            xs = [G.nodes[p]["x"] for p in path]
            ys = [G.nodes[p]["y"] for p in path]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.7)
    else:
        rx = [scenario.x[i] for i in route]
        ry = [scenario.y[i] for i in route]
        ax.plot(rx, ry, color=color, linewidth=linewidth, alpha=0.7)


def plot_solution(scenario: Scenario,
                  solution: Solution,
                  *,
                  ax,
                  title: str,
                  draw_roads: bool = True,
                  add_basemap: bool = True):
    """Render scenario + solution onto an existing matplotlib Axes."""
    ci = scenario.children_idx
    x_children = scenario.x[ci]
    y_children = scenario.y[ci]

    n_routes = len(solution.routes)

    # Background: gray children dots
    ax.scatter(x_children, y_children, s=8, color="lightgray", zorder=2)

    # Each route in its own color
    for c, route in enumerate(solution.routes):
        if len(route) <= 2:
            continue
        color = _route_color(c, n_routes)
        # Highlight cluster members
        mids = [i for i in route if i != scenario.origin_index]
        ax.scatter([scenario.x[i] for i in mids],
                   [scenario.y[i] for i in mids],
                   color=color, s=20, zorder=3)
        _draw_route(ax, scenario, route, color, follow_roads=draw_roads)

    # School / depot
    ax.scatter(scenario.x[scenario.origin_index],
               scenario.y[scenario.origin_index],
               marker="s", color="red", s=100, zorder=6, label="School")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)

    if add_basemap:
        try:
            # Esri's grey canvas rather than OSM Mapnik or CartoDB: osm.org
            # blocks bulk tile fetches and CartoDB now needs an API key, and
            # both serve a watermarked tile instead of an HTTP error, so the
            # failure renders silently into the figure rather than raising
            # here. This one is keyless, and its muted palette keeps the route
            # colours legible in print.
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                            zoom=13, crs="EPSG:4326", attribution_size=4)
        except Exception:
            pass


def plot_single(scenario: Scenario,
                solution: Solution,
                *,
                title: str,
                out_path: Path,
                draw_roads: bool = True):
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_solution(scenario, solution, ax=ax, title=title, draw_roads=draw_roads)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(scenario: Scenario,
                         solutions_by_name: dict[str, Solution],
                         *,
                         out_path: Path,
                         draw_roads: bool = False):
    """5-up grid of all algorithms on the same scenario.

    Roads are off by default for the grid (faster, less visually noisy).
    """
    n = len(solutions_by_name)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows), squeeze=False)

    for ax_idx, (name, sol) in enumerate(solutions_by_name.items()):
        ax = axes[ax_idx // cols][ax_idx % cols]
        title = (f"{name}\n"
                 f"buses={sum(1 for r in sol.routes if len(r) > 2)}, "
                 f"routes={len(sol.routes)}")
        plot_solution(scenario, sol, ax=ax, title=title,
                      draw_roads=draw_roads, add_basemap=True)

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle(f"N={scenario.n_children}, seed={scenario.seed}",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

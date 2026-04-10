"""
visualize_results.py  —  Stage 3: All maps and dashboard.

Outputs saved to outputs/maps/:
  02_zones_overview.png      Children coloured by zone, polygons outlined.
  03_all_routes.png          Every bus route on one Valle-wide map.
  04_zone_<name>.png         Detailed per-zone map (one per active zone).
  05_dashboard.png           Summary statistics panel.

Run:
    python visualize_results.py
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import shape

try:
    import contextily as ctx
    _CTX = True
except ImportError:
    _CTX = False

import config as cfg

Path(cfg.MAPS_DIR).mkdir(parents=True, exist_ok=True)

# ── Colour helpers ────────────────────────────────────────────────────────
_TAB20 = plt.cm.get_cmap("tab20", 20)
_TAB10 = plt.cm.get_cmap("tab10", 10)

def zone_color(idx: int):   return _TAB20(idx % 20)
def bus_color(idx: int):    return _TAB10(idx % 10)


def _basemap(ax, zoom: int = 13, alpha: float = 0.38) -> None:
    if not _CTX:
        return
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=zoom, crs="EPSG:4326", alpha=alpha)
    except Exception:
        pass


def _style(ax, title: str = "", fontsize: int = 11) -> None:
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def _extent(ax, xs: np.ndarray, ys: np.ndarray, pad: float = 0.06) -> None:
    xp = (xs.max() - xs.min()) * pad + 0.003
    yp = (ys.max() - ys.min()) * pad + 0.003
    ax.set_xlim(xs.min() - xp, xs.max() + xp)
    ax.set_ylim(ys.min() - yp, ys.max() + yp)


def _draw_zone_polygons(ax, zones_geo: list[dict],
                        alpha_face=0.13, alpha_edge=0.55) -> None:
    """Lightly shade zone polygons."""
    for i, z in enumerate(zones_geo):
        geom = z["geometry"]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        patches = [MplPoly(np.array(p.exterior.coords), closed=True)
                   for p in polys]
        pc = PatchCollection(patches,
                             facecolor=zone_color(i),
                             edgecolor=zone_color(i),
                             alpha=alpha_face,
                             linewidths=0.8)
        ax.add_collection(pc)


def _label_zones(ax, zones_geo: list[dict], active_names: set) -> None:
    for z in zones_geo:
        if z["name"] not in active_names:
            continue
        cx, cy = z["geometry"].centroid.x, z["geometry"].centroid.y
        short  = z["name"].replace("Comuna ", "C").split(" - ")[0]
        ax.text(cx, cy, short, fontsize=5.5, ha="center", va="center",
                color="#1a1a1a",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def _load_zone_geometries(data: dict) -> list[dict]:
    """Re-attach Shapely geometries to zones_meta list."""
    # Reload from GeoJSONs (same logic as optimize_routes.load_zones)
    zones_geo = []
    with open(cfg.COMUNAS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        p    = feat["properties"]
        ref  = str(p.get("ref", "")).strip().zfill(2)
        name = str(p.get("name", f"Comuna {ref}")).strip()
        zones_geo.append({"name": name, "municipality": "Medellín",
                          "geometry": shape(feat["geometry"])})
    zones_geo.sort(key=lambda z: int(z["name"].split()[1])
                   if z["municipality"] == "Medellín" else 99)

    with open(cfg.VALLE_GEOJSON, encoding="utf-8") as f:
        gj2 = json.load(f)
    for feat in gj2["features"]:
        p = feat["properties"]
        if p.get("municipio") != "Medellín":
            zones_geo.append({"name": p["municipio"],
                               "municipality": p["municipio"],
                               "geometry": shape(feat["geometry"])})
    return zones_geo


# =============================================================================
# Map 02 — Zones overview
# =============================================================================

def plot_zones_overview(scenario: dict, data: dict,
                        zones_geo: list[dict]) -> None:
    x          = np.array(scenario["x"])
    y          = np.array(scenario["y"])
    origin_idx = int(scenario["origin_index"])
    labels     = np.array(data["zone_labels"])
    active     = {r["zone_name"] for r in data["results"]}

    fig, ax = plt.subplots(figsize=(15, 15), facecolor="#F4F6F9")
    ax.set_facecolor("#D6E4F0")
    _style(ax, f"Children by Zone — Valle de Aburrá\n"
           f"({cfg.N_CHILDREN} children, {len(active)} active zones)")

    _draw_zone_polygons(ax, zones_geo)
    _label_zones(ax, zones_geo, active)

    name_to_idx = {z["name"]: i for i, z in enumerate(zones_geo)}
    handles     = []

    for i, z in enumerate(zones_geo):
        mask = labels == i
        if not mask.any():
            continue
        color = zone_color(i)
        ax.scatter(x[mask], y[mask], color=color, s=20, zorder=5,
                   edgecolors="white", linewidths=0.3)
        n_z   = int(mask.sum())
        short = z["name"].replace("Comuna ", "C").split(" - ")[0]
        handles.append(mpatches.Patch(facecolor=color, edgecolor="#777",
                                      linewidth=0.5,
                                      label=f"{short}  ({n_z})"))

    ax.scatter(x[origin_idx], y[origin_idx], marker="*",
               c="red", s=500, zorder=7, edgecolors="white", linewidths=1.2)
    ax.annotate(cfg.SCHOOL_NAME,
                (x[origin_idx], y[origin_idx]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    handles.append(mpatches.Patch(facecolor="red", label=f"School ({cfg.SCHOOL_NAME})"))
    ax.legend(handles=handles, fontsize=7, loc="lower left",
              ncol=2, framealpha=0.92, edgecolor="#aaa")
    _extent(ax, x, y, pad=0.04)
    _basemap(ax, zoom=12, alpha=0.32)

    out = Path(cfg.MAPS_DIR) / "02_zones_overview.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Map 03 — All routes
# =============================================================================

def plot_all_routes(scenario: dict, data: dict,
                    zones_geo: list[dict]) -> None:
    x          = np.array(scenario["x"])
    y          = np.array(scenario["y"])
    origin_idx = int(scenario["origin_index"])
    results    = data["results"]
    summary    = data["summary"]

    fig, ax = plt.subplots(figsize=(16, 16), facecolor="#F4F6F9")
    ax.set_facecolor("#D6E4F0")
    _style(ax, f"All Bus Routes — Multi-Objective CVRPTW\n"
           f"Capacity={cfg.BUS_CAPACITY}  ·  Max route={cfg.MAX_ROUTE_MINUTES} min  "
           f"·  Buses={summary['total_buses']}  "
           f"·  Total={summary['total_distance_km']:.0f} km")

    _draw_zone_polygons(ax, zones_geo, alpha_face=0.08, alpha_edge=0.35)

    bus_global = 0
    for zone_res in results:
        for c, route in enumerate(zone_res["routes"]):
            if not route:
                bus_global += 1
                continue
            color      = bus_color(bus_global)
            path_nodes = [origin_idx] + route + [origin_idx]
            px = [x[i] for i in path_nodes]
            py = [y[i] for i in path_nodes]
            ax.plot(px, py, color=color, linewidth=1.2,
                    alpha=0.60, zorder=3, solid_capstyle="round")
            ax.scatter(x[route], y[route], color=color, s=16,
                       zorder=5, edgecolors="white", linewidths=0.25)
            bus_global += 1

    ax.scatter(x[origin_idx], y[origin_idx], marker="*",
               c="red", s=600, zorder=8, edgecolors="white", linewidths=1.5)
    ax.annotate(cfg.SCHOOL_NAME, (x[origin_idx], y[origin_idx]),
                xytext=(9, 9), textcoords="offset points",
                fontsize=10, fontweight="bold", color="red",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    _extent(ax, x, y, pad=0.04)
    _basemap(ax, zoom=12, alpha=0.38)

    out = Path(cfg.MAPS_DIR) / "03_all_routes.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Map 04 — Per-zone detailed maps
# =============================================================================

def plot_zone_detail(scenario: dict, zone_res: dict) -> None:
    x          = np.array(scenario["x"])
    y          = np.array(scenario["y"])
    origin_idx = int(scenario["origin_index"])
    routes     = zone_res["routes"]
    k          = zone_res["k_buses"]
    name       = zone_res["zone_name"]

    all_pts = list({origin_idx} | {n for r in routes for n in r})
    xs = np.array([x[i] for i in all_pts])
    ys = np.array([y[i] for i in all_pts])

    # Layout: left = combined view, right = individual bus panels
    ncols_right = min(k, 3)
    nrows_right = max(1, math.ceil(k / ncols_right))
    figw  = 8 + 6 * ncols_right
    figh  = max(8, 6 * nrows_right)

    fig   = plt.figure(figsize=(figw, figh), facecolor="#F4F6F9")
    short = name.replace("Comuna ", "C").split(" - ")[0]
    fig.suptitle(
        f"Zone: {name}\n"
        f"{zone_res['n_children']} children · {k} bus(es) · "
        f"{zone_res['total_distance_m']/1000:.1f} km · "
        f"max route {zone_res['max_route_min']:.0f} min",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Outer grid: [combined | individual grid]
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, ncols_right],
                             wspace=0.07)
    ax_all = fig.add_subplot(outer[0])
    ax_all.set_facecolor("#D6E4F0")
    _style(ax_all, "All buses combined")

    inner = outer[1].subgridspec(nrows_right, ncols_right,
                                  hspace=0.30, wspace=0.15)

    for c in range(k):
        row, col = divmod(c, ncols_right)
        ax_c = fig.add_subplot(inner[row, col])
        ax_c.set_facecolor("#D6E4F0")

        route  = routes[c]
        color  = bus_color(c)
        rt_min = zone_res["route_times_min"][c] if c < len(zone_res["route_times_min"]) else 0
        dist_km= zone_res["route_distances_m"][c] / 1000 if c < len(zone_res["route_distances_m"]) else 0
        over   = "⚠ OVER LIMIT" if rt_min > cfg.MAX_ROUTE_MINUTES else ""
        _style(ax_c, f"Bus {c+1}  ·  {len(route)} children\n"
               f"{rt_min:.0f} min  ·  {dist_km:.2f} km  {over}",
               fontsize=8)

        if route:
            path_nodes = [origin_idx] + route + [origin_idx]
            px = [x[i] for i in path_nodes]
            py = [y[i] for i in path_nodes]

            # Draw route with arrows to show direction
            for seg in range(len(path_nodes) - 1):
                ax_c.annotate("",
                    xy=(x[path_nodes[seg+1]], y[path_nodes[seg+1]]),
                    xytext=(x[path_nodes[seg]], y[path_nodes[seg]]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.3, alpha=0.7))

            ax_c.scatter(x[route], y[route], color=color, s=45, zorder=5,
                         edgecolors="white", linewidths=0.5)
            # Stop order numbers
            for stop_num, node_idx in enumerate(route, start=1):
                ax_c.text(x[node_idx], y[node_idx], str(stop_num),
                          fontsize=6, ha="center", va="center",
                          color="white", fontweight="bold", zorder=6)

            # Combined view
            ax_all.plot(px, py, color=color, linewidth=1.4, alpha=0.65)
            ax_all.scatter(x[route], y[route], color=color, s=20,
                           zorder=5, edgecolors="white", linewidths=0.25)

        ax_c.scatter(x[origin_idx], y[origin_idx], marker="*",
                     c="red", s=150, zorder=7,
                     edgecolors="white", linewidths=0.8)
        _extent(ax_c, np.append(xs, x[origin_idx]),
                np.append(ys, y[origin_idx]), pad=0.12)
        _basemap(ax_c, zoom=14, alpha=0.40)

    ax_all.scatter(x[origin_idx], y[origin_idx], marker="*",
                   c="red", s=300, zorder=7,
                   edgecolors="white", linewidths=1.0)
    # Legend for combined
    handles = [mpatches.Patch(facecolor=bus_color(c),
                               label=f"Bus {c+1} ({len(routes[c])})")
               for c in range(k)]
    handles.append(mpatches.Patch(facecolor="red", label="School"))
    ax_all.legend(handles=handles, fontsize=7.5, loc="best",
                  framealpha=0.92, ncol=1 if k <= 5 else 2)
    _extent(ax_all, np.append(xs, x[origin_idx]),
            np.append(ys, y[origin_idx]), pad=0.10)
    _basemap(ax_all, zoom=14, alpha=0.38)

    safe = name.replace(" ", "_").replace("/", "-").replace("ü", "u")
    out  = Path(cfg.MAPS_DIR) / f"04_zone_{safe}.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Map 05 — Dashboard
# =============================================================================

def plot_dashboard(data: dict) -> None:
    results = data["results"]
    summary = data["summary"]
    cfg_    = data["config"]

    fig = plt.figure(figsize=(20, 12), facecolor="#F4F6F9")
    fig.suptitle("School Bus Routing — Summary Dashboard\nValle de Aburrá · UPB",
                 fontsize=16, fontweight="bold", y=1.00)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    # ── Panel A: children & buses per zone ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_facecolor("#F4F6F9")
    zone_names  = [r["zone_name"].replace("Comuna ", "C").split(" - ")[0]
                   for r in results]
    n_children  = [r["n_children"]  for r in results]
    n_buses     = [r["k_buses"]     for r in results]
    x_pos       = np.arange(len(results))
    width       = 0.4
    colors_bar  = [zone_color(r["zone_idx"]) for r in results]

    bars1 = ax_a.bar(x_pos - width/2, n_children, width,
                     color=colors_bar, alpha=0.85, label="Children", edgecolor="white")
    ax_b2 = ax_a.twinx()
    bars2 = ax_b2.bar(x_pos + width/2, n_buses, width,
                      color=colors_bar, alpha=0.50, label="Buses",
                      edgecolor="white", hatch="//")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(zone_names, rotation=45, ha="right", fontsize=7.5)
    ax_a.set_ylabel("Children", fontsize=9)
    ax_b2.set_ylabel("Buses", fontsize=9)
    ax_a.set_title("Children and Buses per Zone", fontsize=10, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")
    ax_a.spines[["top"]].set_visible(False)
    lines = [mpatches.Patch(facecolor="#888", alpha=0.85, label="Children"),
             mpatches.Patch(facecolor="#888", alpha=0.50, hatch="//", label="Buses")]
    ax_a.legend(handles=lines, fontsize=8, loc="upper right")

    # ── Panel B: route time distribution ────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor("#F4F6F9")
    all_times = [t for r in results for t in r["route_times_min"]]
    ax_c.hist(all_times, bins=15, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax_c.axvline(cfg.MAX_ROUTE_MINUTES, color="red", linewidth=1.5,
                 linestyle="--", label=f"Limit ({cfg.MAX_ROUTE_MINUTES} min)")
    ax_c.set_xlabel("Route duration (min)", fontsize=9)
    ax_c.set_ylabel("Number of routes", fontsize=9)
    ax_c.set_title("Route Duration Distribution", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.25, linestyle="--")
    ax_c.spines[["top", "right"]].set_visible(False)

    # ── Panel D: distance per zone ───────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.set_facecolor("#F4F6F9")
    zone_km = [r["total_distance_m"] / 1000 for r in results]
    bars_d  = ax_d.bar(x_pos, zone_km, color=colors_bar, alpha=0.85,
                       edgecolor="white")
    for bar, km in zip(bars_d, zone_km):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f"{km:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(zone_names, rotation=45, ha="right", fontsize=7.5)
    ax_d.set_ylabel("Total distance (km)", fontsize=9)
    ax_d.set_title("Total Route Distance per Zone", fontsize=10, fontweight="bold")
    ax_d.grid(axis="y", alpha=0.25, linestyle="--")
    ax_d.spines[["top", "right"]].set_visible(False)

    # ── Panel E: KPI box ─────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis("off")
    kpis = [
        ("Children",           f"{cfg_['n_children']}"),
        ("Bus capacity",       f"{cfg_['bus_capacity']}"),
        ("Max route",          f"{cfg_['max_route_minutes']} min"),
        ("Boarding time",      f"{cfg_['boarding_seconds']} s/stop"),
        ("",                   ""),
        ("Active zones",       f"{summary['active_zones']}"),
        ("Total buses",        f"{summary['total_buses']}"),
        ("Total distance",     f"{summary['total_distance_km']:.1f} km"),
        ("Cumulative time",    f"{summary['cumulative_min']:.0f} min"),
        ("Longest route",      f"{summary['worst_route_min']:.1f} min"),
    ]
    y_start = 0.97
    ax_e.set_title("Key Metrics", fontsize=10, fontweight="bold", pad=6)
    for i, (label, value) in enumerate(kpis):
        y = y_start - i * 0.096
        if label == "":
            ax_e.axhline(y + 0.04, color="#cccccc", linewidth=0.8)
            continue
        ax_e.text(0.02, y, label, transform=ax_e.transAxes,
                  fontsize=9, color="#555555", va="top")
        ax_e.text(0.98, y, value, transform=ax_e.transAxes,
                  fontsize=9, fontweight="bold", color="#111111",
                  va="top", ha="right")
    ax_e.set_facecolor("#FFFFFF")
    for sp in ax_e.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#DDDDDD")

    out = Path(cfg.MAPS_DIR) / "05_dashboard.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 62)
    print("Stage 3 — Visualisation")
    print("=" * 62)

    scenario = np.load(cfg.SCENARIO_NPZ, allow_pickle=True)
    with open(cfg.RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    print("\nLoading zone geometries …")
    zones_geo = _load_zone_geometries(data)

    print("\n[1/4] Zone overview map …")
    plot_zones_overview(scenario, data, zones_geo)

    print("[2/4] All-routes map …")
    plot_all_routes(scenario, data, zones_geo)

    print("[3/4] Per-zone detail maps …")
    for zone_res in data["results"]:
        print(f"  → {zone_res['zone_name']}")
        plot_zone_detail(scenario, zone_res)

    print("[4/4] Dashboard …")
    plot_dashboard(data)

    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()

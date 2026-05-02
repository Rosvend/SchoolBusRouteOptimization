import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    import contextily as ctx
    _CTX = True
except ImportError:
    _CTX = False

import config as cfg

Path(cfg.MAPS_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

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


def _extent(ax, xs, ys, pad: float = 0.06) -> None:
    xs, ys = np.asarray(xs), np.asarray(ys)
    xp = max((xs.max() - xs.min()) * pad, 0.005)
    yp = max((ys.max() - ys.min()) * pad, 0.005)
    ax.set_xlim(xs.min() - xp, xs.max() + xp)
    ax.set_ylim(ys.min() - yp, ys.max() + yp)


def _load_individual_geometries() -> dict[str, object]:

    geoms: dict[str, object] = {}

    with open(cfg.COMUNAS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        name = str(feat["properties"].get("name", "")).strip()
        if name:
            geoms[name] = shape(feat["geometry"])

    with open(cfg.VALLE_GEOJSON, encoding="utf-8") as f:
        gj2 = json.load(f)
    for feat in gj2["features"]:
        p = feat["properties"]
        if p.get("municipio") != "Medellín":
            geoms[p["municipio"]] = shape(feat["geometry"])

    return geoms


def _load_routing_zone_geometries() -> list[dict]:

    individual = _load_individual_geometries()

    zones_geo = []
    for zdef in cfg.ZONE_DEFINITIONS:
        polys = [individual[m] for m in zdef["members"] if m in individual]
        if not polys:
            continue
        merged   = unary_union(polys)
        centroid = (merged.centroid.x, merged.centroid.y)
        zones_geo.append({
            "zone_name": zdef["zone_name"],
            "color":     zdef["color"],
            "members":   zdef["members"],
            "geometry":  merged,
            "centroid":  centroid,
        })
    return zones_geo


def _draw_zone_polygons(ax, zones_geo: list[dict],
                        alpha_face=0.15, alpha_edge=0.60) -> None:
    for zone in zones_geo:
        geom  = zone["geometry"]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        patches = [MplPoly(np.array(p.exterior.coords), closed=True)
                   for p in polys]
        pc = PatchCollection(patches,
                             facecolor=zone["color"],
                             edgecolor=zone["color"],
                             alpha=alpha_face,
                             linewidths=1.0)
        ax.add_collection(pc)


def _draw_member_polygons(ax, members: list[str],
                          individual_geoms: dict,
                          zone_color: str,
                          alpha_face: float = 0.18) -> None:

    import matplotlib.colors as mc
    import colorsys

    r, g, b = mc.to_rgb(zone_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    for idx, member_name in enumerate(members):
        if member_name not in individual_geoms:
            continue
        geom = individual_geoms[member_name]

        # Alternate lightness slightly so adjacent sub-polygons are distinct
        l_alt = max(0.25, min(0.75, l + (0.12 if idx % 2 == 0 else -0.08)))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l_alt, s)
        face_color = (r2, g2, b2)

        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        patches = [MplPoly(np.array(p.exterior.coords), closed=True)
                   for p in polys]
        pc = PatchCollection(patches,
                             facecolor=face_color,
                             edgecolor="white",
                             alpha=alpha_face + 0.05,
                             linewidths=1.4)
        ax.add_collection(pc)

        # Label at centroid — shorten long names for readability
        cx, cy = geom.centroid.x, geom.centroid.y
        short  = (member_name
                  .replace("Comuna ", "C")      
                  .replace(" - ", "\n"))        # put descriptor on new line
        ax.text(cx, cy, short,
                fontsize=6, ha="center", va="center",
                color="#111111", fontweight="bold",
                multialignment="center",
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="white")])


def _label_zones(ax, zones_geo: list[dict]) -> None:
    for zone in zones_geo:
        cx, cy = zone["centroid"]
        ax.text(cx, cy, zone["zone_name"],
                fontsize=7, ha="center", va="center",
                fontweight="bold", color="#1a1a1a",
                path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground="white")])


def _bus_color(bus_idx: int, zone_color: str):

    import matplotlib.colors as mc
    import colorsys
    r, g, b = mc.to_rgb(zone_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    offsets = [0.0, 0.22, -0.15, 0.38, -0.28, 0.50]
    l2 = max(0.20, min(0.80, l + offsets[bus_idx % len(offsets)]))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
    return (r2, g2, b2)


# =============================================================================
# Map 02 — Zones overview
# =============================================================================

def plot_zones_overview(data: dict, zones_geo: list[dict]) -> None:
    x              = np.array(data["x"])
    y              = np.array(data["y"])
    origin_idx     = int(data["origin_idx"])
    routing_labels = np.array(data["routing_labels"])
    zone_defs      = data["zone_defs"]

    fig, ax = plt.subplots(figsize=(15, 15), facecolor="#F4F6F9")
    ax.set_facecolor("#D6E4F0")
    _style(ax, f"Children by Routing Zone — Valle de Aburrá\n"
           f"({cfg.N_CHILDREN} children · 6 zones · UPB depot)")

    _draw_zone_polygons(ax, zones_geo)
    _label_zones(ax, zones_geo)

    handles = []
    for z_idx, zdef in enumerate(zone_defs):
        mask  = routing_labels == z_idx
        if not mask.any():
            continue
        color = zdef["color"]
        ax.scatter(x[mask], y[mask], color=color, s=22, zorder=5,
                   edgecolors="white", linewidths=0.35)
        n_z = int(mask.sum())
        handles.append(mpatches.Patch(
            facecolor=color, edgecolor="#555", linewidth=0.5,
            label=f"{zdef['zone_name']}  ({n_z} children)"))

    ax.scatter(x[origin_idx], y[origin_idx], marker="*",
               c="red", s=550, zorder=8,
               edgecolors="white", linewidths=1.3)
    ax.annotate(cfg.SCHOOL_NAME, (x[origin_idx], y[origin_idx]),
                xytext=(9, 9), textcoords="offset points",
                fontsize=10, fontweight="bold", color="red",
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="white")])
    handles.append(mpatches.Patch(facecolor="red",
                                  label=f"School ({cfg.SCHOOL_NAME})"))

    ax.legend(handles=handles, fontsize=9, loc="lower left",
              framealpha=0.93, edgecolor="#aaa")
    _extent(ax, x, y, pad=0.04)
    _basemap(ax, zoom=12, alpha=0.32)

    out = Path(cfg.MAPS_DIR) / "02_zones_overview.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Map 03 — All routes
# =============================================================================

def plot_all_routes(data: dict, zones_geo: list[dict]) -> None:
    x          = np.array(data["x"])
    y          = np.array(data["y"])
    origin_idx = int(data["origin_idx"])
    results    = data["results"]
    summary    = data["summary"]

    fig, ax = plt.subplots(figsize=(16, 16), facecolor="#F4F6F9")
    ax.set_facecolor("#D6E4F0")
    _style(ax,
           f"All Bus Routes — Multi-Objective CVRPTW (6 Zones)\n"
           f"Capacity={cfg.BUS_CAPACITY}  ·  Max route={cfg.MAX_ROUTE_MINUTES} min  "
           f"·  {summary['total_buses']} buses  "
           f"·  {summary['total_distance_km']:.0f} km total")

    _draw_zone_polygons(ax, zones_geo, alpha_face=0.09, alpha_edge=0.35)
    _label_zones(ax, zones_geo)

    legend_handles = []
    for zone_res in results:
        zone_color = zone_res["zone_color"]
        for c, route in enumerate(zone_res["routes"]):
            if not route:
                continue
            color      = _bus_color(c, zone_color)
            path_nodes = [origin_idx] + route + [origin_idx]
            px = [x[i] for i in path_nodes]
            py = [y[i] for i in path_nodes]
            ax.plot(px, py, color=color, linewidth=1.4,
                    alpha=0.65, zorder=3, solid_capstyle="round")
            ax.scatter(x[route], y[route], color=color, s=18,
                       zorder=5, edgecolors="white", linewidths=0.3)
            nc   = len(route)
            dist = zone_res["route_distances_m"][c] / 1000
            t    = zone_res["route_times_min"][c]
            legend_handles.append(mpatches.Patch(
                facecolor=color,
                label=f"{zone_res['zone_name']} Bus {c+1}  "
                      f"({nc} children · {t:.0f} min · {dist:.1f} km)"))

    ax.scatter(x[origin_idx], y[origin_idx], marker="*",
               c="red", s=650, zorder=9, edgecolors="white", linewidths=1.5)
    ax.annotate(cfg.SCHOOL_NAME, (x[origin_idx], y[origin_idx]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=11, fontweight="bold", color="red",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    legend_handles.append(
        mpatches.Patch(facecolor="red", label=f"School ({cfg.SCHOOL_NAME})"))

    ax.legend(handles=legend_handles, fontsize=7, loc="lower left",
              framealpha=0.92, edgecolor="#aaa", ncol=2)
    _extent(ax, x, y, pad=0.04)
    _basemap(ax, zoom=12, alpha=0.38)

    out = Path(cfg.MAPS_DIR) / "03_all_routes.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Map 04 — Detailed per-zone maps  
# =============================================================================

def plot_zone_detail(data: dict,
                     zone_res: dict,
                     individual_geoms: dict) -> None:

    x          = np.array(data["x"])
    y          = np.array(data["y"])
    origin_idx = int(data["origin_idx"])
    routes     = zone_res["routes"]
    k          = zone_res["k_buses"]
    name       = zone_res["zone_name"]
    zone_color = zone_res["zone_color"]
    members    = zone_res["members"]   # ← individual commune/muni names

    # Bounding box of all route points + depot
    all_pts = list({origin_idx} | {n for r in routes for n in r})
    xs = np.array([x[i] for i in all_pts])
    ys = np.array([y[i] for i in all_pts])

    # Also extend bbox to include zone polygons
    for m in members:
        if m in individual_geoms:
            b = individual_geoms[m].bounds   
            xs = np.append(xs, [b[0], b[2]])
            ys = np.append(ys, [b[1], b[3]])

    ncols_right = min(k, 3)
    nrows_right = max(1, math.ceil(k / ncols_right))
    figw = 9 + 6 * ncols_right
    figh = max(8, 6 * nrows_right)

    fig = plt.figure(figsize=(figw, figh), facecolor="#F4F6F9")

    # Build a readable sub-unit list for the title
    member_short = [m.replace("Comuna ", "C").split(" - ")[0]
                    for m in members]
    member_str   = " · ".join(member_short)

    fig.suptitle(
        f"Zone: {name}   [{member_str}]\n"
        f"{zone_res['n_children']} children · {k} bus(es) · "
        f"{zone_res['total_distance_m']/1000:.1f} km · "
        f"max route {zone_res['max_route_min']:.0f} min",
        fontsize=12, fontweight="bold", y=1.02,
    )

    outer = fig.add_gridspec(1, 2,
                             width_ratios=[1.3, max(ncols_right, 1)],
                             wspace=0.08)
    ax_all = fig.add_subplot(outer[0])
    ax_all.set_facecolor("#D6E4F0")
    _style(ax_all, "All buses combined\n(communes / sectors shown)")

    # ── Draw individual member polygons in BOTH panels ────────────────────
    _draw_member_polygons(ax_all, members, individual_geoms,
                          zone_color, alpha_face=0.22)

    inner = outer[1].subgridspec(nrows_right, ncols_right,
                                  hspace=0.38, wspace=0.20)

    legend_handles = []

    for c in range(k):
        row, col   = divmod(c, ncols_right)
        ax_c       = fig.add_subplot(inner[row, col])
        ax_c.set_facecolor("#D6E4F0")

        route   = routes[c]
        color   = _bus_color(c, zone_color)
        rt_min  = (zone_res["route_times_min"][c]
                   if c < len(zone_res["route_times_min"]) else 0)
        dist_km = (zone_res["route_distances_m"][c] / 1000
                   if c < len(zone_res["route_distances_m"]) else 0)
        over    = rt_min > cfg.MAX_ROUTE_MINUTES
        flag    = "  ⚠ OVER" if over else ""

        _style(ax_c,
               f"Bus {c+1}  ·  {len(route)} children\n"
               f"{rt_min:.0f} min  ·  {dist_km:.2f} km{flag}",
               fontsize=9)

        # Member polygons in individual panel for context
        _draw_member_polygons(ax_c, members, individual_geoms,
                              zone_color, alpha_face=0.18)

        if route:
            path_nodes = [origin_idx] + route + [origin_idx]

            # Directed arrows for pickup order
            for seg in range(len(path_nodes) - 1):
                x0 = x[path_nodes[seg]];   y0 = y[path_nodes[seg]]
                x1 = x[path_nodes[seg+1]]; y1 = y[path_nodes[seg+1]]
                ax_c.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.4, alpha=0.78,
                                   mutation_scale=12))

            # Child dots with stop-order numbers
            ax_c.scatter(x[route], y[route],
                         color=color, s=55, zorder=6,
                         edgecolors="white", linewidths=0.6)
            for stop_num, node_idx in enumerate(route, start=1):
                ax_c.text(x[node_idx], y[node_idx], str(stop_num),
                          fontsize=6, ha="center", va="center",
                          color="white", fontweight="bold", zorder=7)

            # Same route on combined panel
            px = [x[i] for i in path_nodes]
            py = [y[i] for i in path_nodes]
            ax_all.plot(px, py, color=color, linewidth=1.7,
                        alpha=0.72, zorder=4)
            ax_all.scatter(x[route], y[route], color=color, s=24,
                           zorder=5, edgecolors="white", linewidths=0.3)

        # School marker on individual panel
        ax_c.scatter(x[origin_idx], y[origin_idx], marker="*",
                     c="red", s=200, zorder=8,
                     edgecolors="white", linewidths=0.9)

        _extent(ax_c, xs, ys, pad=0.08)
        _basemap(ax_c, zoom=14, alpha=0.35)

        legend_handles.append(mpatches.Patch(
            facecolor=color,
            label=f"Bus {c+1}  ({len(route)} children)"))

    # School + legend on combined panel
    ax_all.scatter(x[origin_idx], y[origin_idx], marker="*",
                   c="red", s=380, zorder=8,
                   edgecolors="white", linewidths=1.1)

    # Add commune patches to legend
    member_legend = [
        mpatches.Patch(facecolor=zone_color, alpha=0.55, edgecolor="white",
                       label=m.replace("Comuna ", "C").split(" - ")[0]
                             + (f" — {m.split(' - ')[1]}"
                                if " - " in m else ""))
        for m in members if m in individual_geoms
    ]
    legend_handles.append(mpatches.Patch(facecolor="red", label="School (UPB)"))

    ax_all.legend(handles=member_legend + legend_handles,
                  fontsize=7.5, loc="best",
                  framealpha=0.93, edgecolor="#aaa",
                  title="Communes / Sectors  |  Buses",
                  title_fontsize=8)

    _extent(ax_all, xs, ys, pad=0.08)
    _basemap(ax_all, zoom=13, alpha=0.35)

    safe = name.replace(" ", "_").replace("/", "-")
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
    fig.suptitle(
        "School Bus Routing — Summary Dashboard\n"
        "Valle de Aburrá · UPB · 6 Geographic Zones",
        fontsize=16, fontweight="bold", y=1.00)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    zone_names  = [r["zone_name"]             for r in results]
    n_children  = [r["n_children"]            for r in results]
    n_buses     = [r["k_buses"]               for r in results]
    zone_km     = [r["total_distance_m"]/1000  for r in results]
    zone_colors = [r["zone_color"]             for r in results]
    x_pos       = np.arange(len(results))

    # ── Panel A: children & buses per zone ───────────────────────────────
    ax_a  = fig.add_subplot(gs[0, :2])
    ax_a.set_facecolor("#F4F6F9")
    width = 0.38
    bars1 = ax_a.bar(x_pos - width/2, n_children, width,
                     color=zone_colors, alpha=0.88,
                     edgecolor="white", linewidth=0.6)
    ax_a2 = ax_a.twinx()
    ax_a2.bar(x_pos + width/2, n_buses, width,
              color=zone_colors, alpha=0.50,
              edgecolor="white", hatch="//", linewidth=0.6)
    for bar, v in zip(bars1, n_children):
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3, str(v),
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(zone_names, fontsize=10)
    ax_a.set_ylabel("Children", fontsize=9)
    ax_a2.set_ylabel("Buses", fontsize=9)
    ax_a.set_title("Children and Buses per Zone", fontsize=11,
                   fontweight="bold")
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")
    ax_a.spines[["top"]].set_visible(False)
    ax_a2.spines[["top"]].set_visible(False)
    ax_a.legend(
        handles=[mpatches.Patch(facecolor="#888", alpha=0.88,
                                label="Children"),
                 mpatches.Patch(facecolor="#888", alpha=0.50,
                                hatch="//", label="Buses")],
        fontsize=9, loc="upper right")

    # ── Panel B: route time distribution ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_facecolor("#F4F6F9")
    all_times = [t for r in results for t in r["route_times_min"]]
    ax_b.hist(all_times, bins=12, color="#4A90D9",
              edgecolor="white", alpha=0.85)
    ax_b.axvline(cfg.MAX_ROUTE_MINUTES, color="red", linewidth=1.8,
                 linestyle="--",
                 label=f"Limit ({cfg.MAX_ROUTE_MINUTES} min)")
    ax_b.axvline(float(np.mean(all_times)), color="orange", linewidth=1.5,
                 linestyle="--",
                 label=f"Mean ({np.mean(all_times):.0f} min)")
    ax_b.set_xlabel("Route duration (min)", fontsize=9)
    ax_b.set_ylabel("Number of routes", fontsize=9)
    ax_b.set_title("Route Duration Distribution", fontsize=11,
                   fontweight="bold")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.25, linestyle="--")
    ax_b.spines[["top", "right"]].set_visible(False)

    # ── Panel C: distance per zone ────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :2])
    ax_c.set_facecolor("#F4F6F9")
    bars_c = ax_c.bar(x_pos, zone_km, color=zone_colors,
                      alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, km in zip(bars_c, zone_km):
        ax_c.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.4, f"{km:.1f} km",
                  ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(zone_names, fontsize=10)
    ax_c.set_ylabel("Total distance (km)", fontsize=9)
    ax_c.set_title("Total Route Distance per Zone", fontsize=11,
                   fontweight="bold")
    ax_c.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c.spines[["top", "right"]].set_visible(False)

    # ── Panel D: KPI box ──────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.axis("off")
    ax_d.set_title("Key Metrics", fontsize=11, fontweight="bold", pad=8)
    kpis = [
        ("Children",          f"{cfg_['n_children']}"),
        ("Bus capacity",      f"{cfg_['bus_capacity']}"),
        ("Max route",         f"{cfg_['max_route_minutes']} min"),
        ("Boarding time",     f"{cfg_['boarding_seconds']} s / stop"),
        ("SEPARATOR",         ""),
        ("Routing zones",     f"{summary['active_zones']}"),
        ("Total buses",       f"{summary['total_buses']}"),
        ("Total distance",    f"{summary['total_distance_km']:.1f} km"),
        ("Longest route",     f"{summary['worst_route_min']:.1f} min"),
        ("Cumulative time",   f"{summary['cumulative_min']:.0f} min"),
    ]
    y_pos = 0.96
    step  = 0.096
    for i, (label, value) in enumerate(kpis):
        y = y_pos - i * step
        if label == "SEPARATOR":
            # Draw a plain horizontal line without transform argument
            line_y = y + 0.04
            ax_d.plot([0.0, 1.0], [line_y, line_y],
                      color="#cccccc", linewidth=0.8,
                      transform=ax_d.transAxes, clip_on=False)
            continue
        ax_d.text(0.04, y, label, transform=ax_d.transAxes,
                  fontsize=10, color="#555555", va="top")
        ax_d.text(0.96, y, value, transform=ax_d.transAxes,
                  fontsize=10, fontweight="bold", color="#111111",
                  va="top", ha="right")
    for sp in ax_d.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#DDDDDD")
    ax_d.set_facecolor("#FFFFFF")

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

    with open(cfg.RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    print("\nLoading routing zone geometries …")
    zones_geo = _load_routing_zone_geometries()

    print("Loading individual commune/municipality geometries …")
    individual_geoms = _load_individual_geometries()

    print("\n[1/4] Zone overview map …")
    plot_zones_overview(data, zones_geo)

    print("[2/4] All-routes map …")
    plot_all_routes(data, zones_geo)

    print("[3/4] Per-zone detail maps (with individual communes shown) …")
    for zone_res in data["results"]:
        print(f"  → {zone_res['zone_name']}  "
              f"[{', '.join(zone_res['members'])}]")
        plot_zone_detail(data, zone_res, individual_geoms)

    print("[4/4] Dashboard …")
    plot_dashboard(data)

    print("\nStage 3 complete.")
    print(f"All maps saved to: {cfg.MAPS_DIR}/")


if __name__ == "__main__":
    main()
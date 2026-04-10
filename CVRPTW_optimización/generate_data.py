"""
generate_data.py  —  Stage 1: Road network + population-weighted child sampling.

What this does:
  1. Builds 21 sampling zones:
       • 16 Medellín comunas  → barrio polygons (EPSG:9377→WGS84) + CSV pop.
       • 5 other municipalities → polygons from valle_aburra_urbano.geojson
                                  + DANE 2026 population estimates.
  2. Downloads the full Valle de Aburrá drive network from OpenStreetMap.
  3. Adds OSMnx speed/travel-time data to every edge.
  4. Builds a pool of eligible road nodes per zone (residential/urban roads
     inside each polygon, using a Shapely STRtree spatial index).
  5. Samples N_CHILDREN stops:
       • Zone is chosen with probability ∝ population.
       • Within the zone a random eligible node is picked (no repeats).
  6. Snaps the UPB school to the nearest network node.
  7. Computes the full N×N travel-time (seconds) and distance (metres)
     matrices using Dijkstra.
  8. Saves everything to outputs/scenario_data.npz.

Run:
    python generate_data.py
"""

import json
from pathlib import Path

import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.strtree import STRtree

import config as cfg

rng = np.random.default_rng(cfg.SEED)
Path("outputs/maps").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)


# =============================================================================
# 1.  Build sampling zones
# =============================================================================

def _reproject_9377_to_wgs84(geom):
    """Reproject a Shapely geometry from EPSG:9377 to WGS-84 in-place."""
    tf = Transformer.from_crs("EPSG:9377", "EPSG:4326", always_xy=True)

    def _convert_ring(coords):
        return [tf.transform(x, y) for x, y in coords]

    if geom.geom_type == "Polygon":
        ext   = _convert_ring(geom.exterior.coords)
        holes = [_convert_ring(r.coords) for r in geom.interiors]
        return Polygon(ext, holes)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([
            Polygon(_convert_ring(p.exterior.coords),
                    [_convert_ring(r.coords) for r in p.interiors])
            for p in geom.geoms
        ])
    raise ValueError(f"Unsupported geometry: {geom.geom_type}")


def _load_barrio_population() -> dict[str, int]:
    """Parse the semicolon-delimited population CSV. Returns {codigo: pop}."""
    pop = {}
    with open(cfg.POPULATION_CSV, encoding="utf-8-sig") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().replace('"', '').split(';')
            if len(parts) == 3:
                try:
                    pop[parts[0]] = int(parts[2])
                except ValueError:
                    pass
    return pop


def build_zones() -> list[dict]:
    """
    Returns a list of zone dicts, each with:
        name       : str
        municipality: str   ('Medellín' or municipality name)
        commune    : str    (commune name, or same as municipality if none)
        population : int
        geometry   : Shapely Polygon/MultiPolygon in WGS-84
    Ordered: 16 Medellín comunas first, then 5 municipalities.
    """
    zones = []

    # ── Medellín comunas: geometry from comunas_medellin.geojson ─────────
    #    Fine-grained population: aggregate barrio CSV by commune code.
    pop_by_codigo = _load_barrio_population()

    # Build commune → total population by summing barrios whose code
    # starts with the two-digit commune number.
    # barrio codigo format: "CCBB" where CC=commune, BB=barrio within commune
    commune_pop: dict[str, int] = {}
    for codigo, pop in pop_by_codigo.items():
        if len(codigo) == 4 and codigo.isdigit():
            cc = codigo[:2]
            commune_pop[cc] = commune_pop.get(cc, 0) + pop

    with open(cfg.COMUNAS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)

    for feat in gj["features"]:
        p    = feat["properties"]
        ref  = str(p.get("ref", "")).strip().zfill(2)   # zero-pad to "01".."16"
        name = str(p.get("name", f"Comuna {ref}")).strip()
        geom = shape(feat["geometry"])                   # already WGS-84
        pop  = commune_pop.get(ref, commune_pop.get(ref.lstrip("0"), 10_000))
        zones.append({
            "name":         name,
            "municipality": "Medellín",
            "commune":      name,
            "population":   pop,
            "geometry":     geom,
        })

    # Sort Medellín communes by ref number
    zones.sort(key=lambda z: int(z["commune"].split()[1]) if z["municipality"] == "Medellín" else 99)

    # ── Non-Medellín municipalities: geometry from valle_aburra_urbano ───
    with open(cfg.VALLE_GEOJSON, encoding="utf-8") as f:
        gj2 = json.load(f)

    muni_geoms: dict[str, object] = {}
    for feat in gj2["features"]:
        p = feat["properties"]
        if p.get("municipio") != "Medellín":
            muni_geoms[p["municipio"]] = shape(feat["geometry"])  # WGS-84

    for muni_name, pop in cfg.MUNICIPALITY_POPULATION.items():
        if muni_name not in muni_geoms:
            print(f"  WARNING: '{muni_name}' not found in {cfg.VALLE_GEOJSON}")
            continue
        zones.append({
            "name":         muni_name,
            "municipality": muni_name,
            "commune":      muni_name,
            "population":   pop,
            "geometry":     muni_geoms[muni_name],
        })

    total_pop = sum(z["population"] for z in zones)
    print(f"  Built {len(zones)} zones  |  total population: {total_pop:,}")
    for z in zones:
        share = z["population"] / total_pop * 100
        print(f"    {z['name']:<35s}  pop={z['population']:>8,}  ({share:.1f}%)")
    return zones


# =============================================================================
# 2.  Download road network
# =============================================================================

def download_network() -> nx.MultiDiGraph:
    print("Downloading road network from OpenStreetMap …")
    G = ox.graph_from_bbox(
        bbox=(cfg.BBOX["west"],  cfg.BBOX["south"],
              cfg.BBOX["east"],  cfg.BBOX["north"]),
        network_type="drive",
        simplify=True,
    )
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()
    G = ox.add_edge_speeds(G, fallback=cfg.SPEED_FALLBACK_KPH)
    G = ox.add_edge_travel_times(G)
    print(f"  {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges  (largest SCC)")
    return G


# =============================================================================
# 3.  Build eligible node pools per zone (spatial index)
# =============================================================================

def build_zone_pools(G: nx.MultiDiGraph,
                     zones: list[dict]) -> dict[str, list]:
    """
    Returns {zone_name: [node_id, ...]} containing only nodes that lie inside
    the zone polygon AND belong to at least one residential/urban road edge.
    Uses STRtree for efficient bulk point-in-polygon.
    """
    print("Building eligible node pools per zone …")

    # Collect urban-road nodes
    urban_ids: set = set()
    for u, v, data in G.edges(data=True):
        hw = data.get("highway", "")
        hw_set = set(hw) if isinstance(hw, list) else {hw}
        if hw_set & cfg.RESIDENTIAL_ROAD_TYPES:
            urban_ids.add(u)
            urban_ids.add(v)

    urban_nodes  = [n for n in G.nodes if n in urban_ids]
    node_points  = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in urban_nodes]
    print(f"  Urban-road nodes: {len(urban_nodes):,}")

    # STRtree on zone geometries
    zone_geoms = [z["geometry"] for z in zones]
    tree       = STRtree(zone_geoms)

    pools: dict[str, list] = {z["name"]: [] for z in zones}
    for node_id, pt in zip(urban_nodes, node_points):
        hits = tree.query(pt, predicate="within")
        if len(hits) > 0:
            zone_name = zones[int(hits[0])]["name"]
            pools[zone_name].append(node_id)

    for z in zones:
        n = len(pools[z["name"]])
        if n == 0:
            print(f"  WARNING: {z['name']} has 0 eligible nodes!")
        else:
            print(f"  {z['name']:<35s}: {n:,} eligible nodes")

    return pools


# =============================================================================
# 4.  Population-weighted child sampling
# =============================================================================

def sample_children(zones: list[dict],
                    pools: dict[str, list]) -> tuple[list, list[str]]:
    """
    Sample N_CHILDREN unique node-ids.
    Returns (node_ids, zone_names_per_child).
    """
    eligible = [z for z in zones if pools[z["name"]]]
    weights  = np.array([z["population"] for z in eligible], dtype=float)
    weights /= weights.sum()

    sampled_nodes: set = set()
    node_list: list    = []
    node_zones: list   = []

    attempts = 0
    while len(node_list) < cfg.N_CHILDREN:
        attempts += 1
        if attempts > cfg.N_CHILDREN * 500:
            raise RuntimeError(
                f"Only {len(node_list)}/{cfg.N_CHILDREN} unique nodes found. "
                "Lower N_CHILDREN or relax RESIDENTIAL_ROAD_TYPES."
            )
        zone = rng.choice(eligible, p=weights)
        pool = pools[zone["name"]]
        node = int(rng.choice(pool))
        if node not in sampled_nodes:
            sampled_nodes.add(node)
            node_list.append(node)
            node_zones.append(zone["name"])

    print(f"  Sampled {cfg.N_CHILDREN} unique child stops")
    zone_counts = {}
    for zn in node_zones:
        zone_counts[zn] = zone_counts.get(zn, 0) + 1
    for z in zones:
        cnt = zone_counts.get(z["name"], 0)
        if cnt:
            print(f"    {z['name']:<35s}: {cnt} children")
    return node_list, node_zones


# =============================================================================
# 5.  Travel-time and distance matrices
# =============================================================================

def compute_matrices(G: nx.MultiDiGraph,
                     node_ids: list) -> tuple[np.ndarray, np.ndarray]:
    n = len(node_ids)
    time_mat = np.zeros((n, n))
    dist_mat = np.zeros((n, n))

    print(f"Computing {n}×{n} matrices (Dijkstra) …")
    for i in tqdm(range(n)):
        t_len = nx.single_source_dijkstra_path_length(
            G, source=node_ids[i], weight="travel_time")
        d_len = nx.single_source_dijkstra_path_length(
            G, source=node_ids[i], weight="length")
        for j in range(n):
            time_mat[i, j] = float(t_len.get(node_ids[j], np.inf))
            dist_mat[i, j] = float(d_len.get(node_ids[j], np.inf))

    # Replace inf with large-but-finite penalty
    max_t = np.nanmax(time_mat[np.isfinite(time_mat)]) * 10
    max_d = np.nanmax(dist_mat[np.isfinite(dist_mat)]) * 10
    time_mat = np.where(np.isinf(time_mat), max_t, time_mat)
    dist_mat = np.where(np.isinf(dist_mat), max_d, dist_mat)
    return time_mat, dist_mat


# =============================================================================
# 6.  Scenario overview map
# =============================================================================

def plot_scenario(xi: np.ndarray, yi: np.ndarray,
                  origin_idx: int, node_zones: list[str],
                  zones: list[dict]) -> None:
    """Two-panel map: left = zone polygons + children, right = stop density."""
    zone_name_to_idx = {z["name"]: i for i, z in enumerate(zones)}
    cmap = plt.cm.get_cmap("tab20", len(zones))

    fig, axes = plt.subplots(1, 2, figsize=(22, 11),
                              facecolor="#F4F6F9")
    fig.suptitle(
        f"Valle de Aburrá — School Bus Scenario\n"
        f"{cfg.N_CHILDREN} children · {len(zones)} zones · UPB depot",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # ── Left: zones + children ───────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#D6E4F0")
    ax.set_title("Zone Polygons & Child Stops", fontsize=12, pad=8)

    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection

    for i, z in enumerate(zones):
        geom = z["geometry"]
        polys_raw = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        patches = [MplPoly(np.array(p.exterior.coords), closed=True)
                   for p in polys_raw]
        pc = PatchCollection(patches, facecolor=cmap(i), alpha=0.30,
                             edgecolor=cmap(i), linewidths=0.8)
        ax.add_collection(pc)
        # Label at centroid
        cx, cy = z["geometry"].centroid.x, z["geometry"].centroid.y
        label  = z["name"].replace("Comuna ", "C").split(" - ")[0]
        ax.text(cx, cy, label, fontsize=5.5, ha="center", va="center",
                color="black", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          alpha=0.5, linewidth=0))

    # Children coloured by zone
    child_xi = xi[:-1]
    child_yi = yi[:-1]
    child_colors = [cmap(zone_name_to_idx.get(zn, 0)) for zn in node_zones]
    ax.scatter(child_xi, child_yi, c=child_colors, s=16, zorder=5,
               edgecolors="white", linewidths=0.25)
    ax.scatter(xi[origin_idx], yi[origin_idx], marker="*",
               c="red", s=400, zorder=7, edgecolors="white",
               linewidths=1.0, label="School (UPB)")

    ax.set_xlim(cfg.BBOX["west"], cfg.BBOX["east"])
    ax.set_ylim(cfg.BBOX["south"], cfg.BBOX["north"])
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=12, crs="EPSG:4326", alpha=0.30)
    except Exception:
        pass

    # ── Right: population bar chart ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F4F6F9")
    ax2.set_title("Population per Zone (2026 estimate)", fontsize=12, pad=8)

    names = [z["name"].replace("Comuna ", "C").split(" - ")[0] for z in zones]
    pops  = [z["population"] for z in zones]
    colors_bar = [cmap(i) for i in range(len(zones))]
    bars = ax2.barh(names, pops, color=colors_bar, edgecolor="white",
                    linewidth=0.5)
    for bar, pop in zip(bars, pops):
        ax2.text(bar.get_width() + max(pops) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{pop:,}", va="center", fontsize=7, color="#333333")
    ax2.set_xlabel("Population", fontsize=10)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(cfg.MAPS_DIR) / "01_scenario_overview.png"
    fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 62)
    print("Stage 1 — Data Generation  (Valle de Aburrá)")
    print("=" * 62)

    print("\n[1/5] Building sampling zones …")
    zones = build_zones()

    print("\n[2/5] Downloading road network …")
    G = download_network()

    print("\n[3/5] Building node pools …")
    pools = build_zone_pools(G, zones)

    print(f"\n[4/5] Sampling {cfg.N_CHILDREN} children …")
    child_nodes, node_zones = sample_children(zones, pools)

    school_node = ox.nearest_nodes(G, X=cfg.SCHOOL_LON, Y=cfg.SCHOOL_LAT)
    node_ids    = child_nodes + [school_node]
    origin_idx  = len(node_ids) - 1

    xi = np.array([G.nodes[n]["x"] for n in node_ids])
    yi = np.array([G.nodes[n]["y"] for n in node_ids])

    print("\n[5/5] Computing travel-time & distance matrices …")
    time_mat, dist_mat = compute_matrices(G, node_ids)
    avg_min = np.mean(time_mat[time_mat > 0]) / 60
    print(f"  Average travel time (non-zero): {avg_min:.1f} min")

    np.savez(
        cfg.SCENARIO_NPZ,
        node_ids    = np.array(node_ids),
        x           = xi,
        y           = yi,
        time_matrix = time_mat,
        dist_matrix = dist_mat,
        origin_index= origin_idx,
        bus_capacity= cfg.BUS_CAPACITY,
        seed        = cfg.SEED,
        node_zones  = np.array(node_zones),
        zone_names  = np.array([z["name"] for z in zones]),
    )
    print(f"  Saved → {cfg.SCENARIO_NPZ}")

    print("\nGenerating scenario map …")
    plot_scenario(xi, yi, origin_idx, node_zones, zones)

    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    main()

"""
data_generation.py
==================
Stage 1 — Road network + scenario data.

Downloads the Valle de Aburrá road network from OpenStreetMap,
samples N_CHILDREN random stop locations INSIDE the zone GeoJSON polygons,
computes pairwise shortest-path distances, and saves everything to
scenario_data.npz.

Run:
    python data_generation.py
"""

import json
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED          = 42
NORTH, SOUTH  =  6.435,  6.030
EAST,  WEST   = -75.330, -75.680

ORIGIN_LONLAT = (-75.58683269546758, 6.243283526198216)  # school / depot
N_CHILDREN    = 200
BUS_CAPACITY  = 20

# GeoJSON that defines the valid sampling area (comunas + municipios del valle)
VALLE_GEOJSON = "valle_aburra_urbano.geojson"

# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# 1. Build valid sampling polygon from the GeoJSON
print("Loading zone polygons ...")
with open(VALLE_GEOJSON, encoding="utf-8") as f:
    gj = json.load(f)

valid_area = unary_union([shape(feat["geometry"]) for feat in gj["features"]])
print(f"  Valid area bounds: {valid_area.bounds}")

# 2. Download road network
# osmnx 2.x bbox format: (left, bottom, right, top) = (west, south, east, north)
print("Downloading Valle de Aburra road network ...")
G = ox.graph_from_bbox(
    bbox=(WEST, SOUTH, EAST, NORTH),
    network_type="drive",
    simplify=True,
)

nodes_in_bbox = [
    n for n in G.nodes
    if WEST <= G.nodes[n]["x"] <= EAST and SOUTH <= G.nodes[n]["y"] <= NORTH
]
G2 = G.subgraph(nodes_in_bbox).copy()
largest_scc = max(nx.strongly_connected_components(G2), key=len)
G2 = G2.subgraph(largest_scc).copy()
print(f"  Cropped graph (largest SCC): {G2.number_of_nodes()} nodes, "
      f"{G2.number_of_edges()} edges")

# 3. Filter nodes to only those inside the valid GeoJSON area
print("Filtering nodes to valid zone area ...")
all_nodes   = list(G2.nodes)
valid_nodes = [
    n for n in all_nodes
    if valid_area.contains(Point(G2.nodes[n]["x"], G2.nodes[n]["y"]))
]
print(f"  {len(valid_nodes):,} / {len(all_nodes):,} nodes inside zone polygons")

if len(valid_nodes) < N_CHILDREN:
    raise RuntimeError(
        f"Only {len(valid_nodes)} valid nodes found but N_CHILDREN={N_CHILDREN}. "
        "Reduce N_CHILDREN or expand the GeoJSON coverage."
    )

# 4. Sample children from valid nodes only
ni = list(rng.choice(valid_nodes, size=N_CHILDREN, replace=False))

# Snap depot to nearest node in full graph (school may sit on any road)
x_all = np.array([G2.nodes[n]["x"] for n in all_nodes])
y_all = np.array([G2.nodes[n]["y"] for n in all_nodes])
dists_to_origin = (x_all - ORIGIN_LONLAT[0])**2 + (y_all - ORIGIN_LONLAT[1])**2
origin_node = all_nodes[int(np.argmin(dists_to_origin))]

ni.append(origin_node)   # last entry is always the school / depot
xi = np.array([G2.nodes[n]["x"] for n in ni])
yi = np.array([G2.nodes[n]["y"] for n in ni])
print(f"  {N_CHILDREN} child nodes sampled (all inside zones) + 1 depot")
print(f"  Depot: ({G2.nodes[origin_node]['x']:.5f}, {G2.nodes[origin_node]['y']:.5f})")

# 5. Compute pairwise shortest-path distances
print("Computing pairwise shortest-path distances ...")
n_points    = len(ni)
dist_matrix = np.zeros((n_points, n_points))

for i in tqdm(range(n_points)):
    lengths = nx.single_source_dijkstra_path_length(G2, source=ni[i], weight="length")
    for j in range(n_points):
        dist_matrix[i, j] = float(lengths[ni[j]])

# 6. Save
out_data = "scenario_data.npz"
np.savez(
    out_data,
    node_ids     = np.array(ni),
    x            = xi,
    y            = yi,
    dist_matrix  = dist_matrix,
    origin_index = n_points - 1,
    bus_capacity = BUS_CAPACITY,
    seed         = SEED,
)
print(f"  Saved to {out_data}")

# 7. Plot
fig, ax = plt.subplots(figsize=(14, 14))
ax.scatter(xi[:-1], yi[:-1], marker=".", color="steelblue", s=12,
           label=f"Children (n={N_CHILDREN}, all inside zones)")
ax.scatter(xi[-1], yi[-1], marker="s", color="red", s=80,
           zorder=6, label="School / depot")

for _ in range(10):
    src  = rng.choice(ni[:-1])
    path = nx.shortest_path(G2, src, origin_node, weight="length")
    ax.plot([G2.nodes[p]["x"] for p in path],
            [G2.nodes[p]["y"] for p in path], "-r", linewidth=0.8, alpha=0.4)

ax.set_xlim(WEST, EAST)
ax.set_ylim(SOUTH, NORTH)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend()
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13, crs="EPSG:4326")
fig.savefig("scenario_map.png", dpi=150, bbox_inches="tight")
print("  Map saved to scenario_map.png")
plt.close(fig)
print("Done.")

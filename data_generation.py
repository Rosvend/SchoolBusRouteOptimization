import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from tqdm import tqdm

SEED = 42
rng = np.random.default_rng(SEED)

PLACE = "Medellin, Colombia"
NORTH, SOUTH = 6.312, 6.182
EAST, WEST = -75.531, -75.626
ORIGIN_LONLAT = (-75.58683269546758, 6.243283526198216)  # school / depot
N_CHILDREN = 200
BUS_CAPACITY = 20 

# 1. download road network
print("Downloading Medellín road network …")
G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)

nodes_in_bbox = [
    n for n in G.nodes
    if WEST <= G.nodes[n]["x"] <= EAST and SOUTH <= G.nodes[n]["y"] <= NORTH
]
G2 = G.subgraph(nodes_in_bbox).copy()

# Keep only the largest strongly connected component so every node
# can reach every other node (required for the distance matrix).
largest_scc = max(nx.strongly_connected_components(G2), key=len)
G2 = G2.subgraph(largest_scc).copy()
print(f"  Cropped graph (largest SCC): {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

# 2. Sample child locations + find nearest node to origin 
all_nodes = list(G2.nodes)
ni = list(rng.choice(all_nodes, size=N_CHILDREN))

# Snap the origin to the closest node in the graph
x_all = np.array([G2.nodes[n]["x"] for n in all_nodes])
y_all = np.array([G2.nodes[n]["y"] for n in all_nodes])
dists_to_origin = (x_all - ORIGIN_LONLAT[0]) ** 2 + (y_all - ORIGIN_LONLAT[1]) ** 2
origin_node = all_nodes[int(np.argmin(dists_to_origin))]

ni.append(origin_node)  # last entry is the origin / school

xi = np.array([G2.nodes[n]["x"] for n in ni])
yi = np.array([G2.nodes[n]["y"] for n in ni])

print(f"  {N_CHILDREN} child nodes + 1 origin node sampled")

# 3. Compute pairwise shortest-path distances
print("Computing pairwise shortest-path distances …")
n_points = len(ni)
dist_matrix = np.zeros((n_points, n_points))

for i in tqdm(range(n_points)):
    lengths = nx.single_source_dijkstra_path_length(G2, source=ni[i], weight="length")
    for j in range(n_points):
        dist_matrix[i, j] = float(lengths[ni[j]])

# 4. Save results
out_data = "scenario_data.npz"
np.savez(
    out_data,
    node_ids=np.array(ni),
    x=xi,
    y=yi,
    dist_matrix=dist_matrix,
    origin_index=n_points - 1,
    bus_capacity=BUS_CAPACITY,
    seed=SEED,
)
print(f"  Data saved to {out_data}")

# 5. Plot
fig, ax = plt.subplots(figsize=(14, 14))
ax.scatter(xi[:-1], yi[:-1], marker=".", color="black", label="Children")
ax.scatter(xi[-1], yi[-1], marker="s", color="red", s=60, zorder=5, label="Origin (school)")

# Draw a few sample shortest paths to the origin
for _ in range(10):
    src = rng.choice(ni[:-1])
    path = nx.shortest_path(G2, src, origin_node, weight="length")
    path_x = [G2.nodes[p]["x"] for p in path]
    path_y = [G2.nodes[p]["y"] for p in path]
    ax.plot(path_x, path_y, "-r", linewidth=0.8, alpha=0.6)

ax.set_xlim(WEST, EAST)
ax.set_ylim(SOUTH, NORTH)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend()
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15, crs="EPSG:4326")

out_map = "scenario_map.png"
fig.savefig(out_map, dpi=150, bbox_inches="tight")
print(f"  Map saved to {out_map}")
plt.close(fig)

print("Done.")
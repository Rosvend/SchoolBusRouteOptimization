"""Generate scenario_data.npz using the locally cached medellin_drive.graphml."""
import numpy as np
import osmnx as ox
import networkx as nx
from tqdm import tqdm

SEED = 42
rng = np.random.default_rng(SEED)

NORTH, SOUTH = 6.312, 6.182
EAST, WEST = -75.531, -75.626
ORIGIN_LONLAT = (-75.58683269546758, 6.243283526198216)
N_CHILDREN = 200
BUS_CAPACITY = 20

GRAPHML_PATH = "SchoolBusRouteOptimization/medellin_drive.graphml"

print("Cargando grafo desde archivo local...")
G = ox.load_graphml(GRAPHML_PATH)

nodes_in_bbox = [
    n for n in G.nodes
    if WEST <= G.nodes[n]["x"] <= EAST and SOUTH <= G.nodes[n]["y"] <= NORTH
]
G2 = G.subgraph(nodes_in_bbox).copy()
largest_scc = max(nx.strongly_connected_components(G2), key=len)
G2 = G2.subgraph(largest_scc).copy()
print(f"  Grafo recortado (SCC): {G2.number_of_nodes()} nodos, {G2.number_of_edges()} aristas")

all_nodes = list(G2.nodes)
ni = list(rng.choice(all_nodes, size=N_CHILDREN))

x_all = np.array([G2.nodes[n]["x"] for n in all_nodes])
y_all = np.array([G2.nodes[n]["y"] for n in all_nodes])
dists_to_origin = (x_all - ORIGIN_LONLAT[0]) ** 2 + (y_all - ORIGIN_LONLAT[1]) ** 2
origin_node = all_nodes[int(np.argmin(dists_to_origin))]

ni.append(origin_node)
xi = np.array([G2.nodes[n]["x"] for n in ni])
yi = np.array([G2.nodes[n]["y"] for n in ni])
print(f"  {N_CHILDREN} nodos de niños + 1 origen muestreados")

print("Calculando matriz de distancias (Dijkstra)...")
n_points = len(ni)
dist_matrix = np.zeros((n_points, n_points))
for i in tqdm(range(n_points)):
    lengths = nx.single_source_dijkstra_path_length(G2, source=ni[i], weight="length")
    for j in range(n_points):
        dist_matrix[i, j] = float(lengths.get(ni[j], float("inf")))

np.savez(
    "scenario_data.npz",
    node_ids=np.array(ni),
    x=xi,
    y=yi,
    dist_matrix=dist_matrix,
    origin_index=n_points - 1,
    bus_capacity=BUS_CAPACITY,
    seed=SEED,
)
print("  Guardado en scenario_data.npz")

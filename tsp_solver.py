"""
tsp_solver.py
=============
Stage 3 (legacy / standalone) — Per-cluster TSP routing.

This script runs TSP on the output of clustering.py WITHOUT zone partitioning.
If you are using zone_partitioning.py, you do NOT need to run this separately —
the same logic is embedded inside zone_partitioning.py per zone.

Keep this file for reference or for running a non-zoned baseline.

Run:
    python src/data_generation.py
    python src/clustering.py
    python src/tsp_solver.py
"""

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Load data
scenario   = np.load("scenario_data.npz")
full_dist  = scenario["dist_matrix"]
x          = scenario["x"]
y          = scenario["y"]
origin_idx = int(scenario["origin_index"])

clust        = np.load("clustering_result.npz")
labels       = clust["labels"]
medoids      = clust["medoids"]
k            = int(clust["k"])
children_idx = clust["children_idx"]


def solve_tsp(dist_sub, depot):
    n = dist_sub.shape[0]
    if n <= 2:
        return list(range(n))

    int_dist = (dist_sub * 1000).astype(np.int64)
    manager  = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing  = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(int_dist[manager.IndexToNode(from_index),
                             manager.IndexToNode(to_index)])

    cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools could not find a solution")

    route, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    return route


routes         = []
total_distance = 0.0

for c in range(k):
    child_mask       = labels == c
    child_local      = np.where(child_mask)[0]
    scenario_indices = children_idx[child_local]
    sub_indices      = np.append(scenario_indices, origin_idx)
    depot_pos        = len(sub_indices) - 1
    dist_sub         = full_dist[np.ix_(sub_indices, sub_indices)]
    route_local      = solve_tsp(dist_sub, depot_pos)
    route_full       = [int(sub_indices[i]) for i in route_local]
    routes.append(route_full)

    route_dist = sum(
        full_dist[route_full[i], route_full[i + 1]]
        for i in range(len(route_full) - 1)
    )
    total_distance += route_dist
    print(f"Cluster {c:2d}: {len(scenario_indices):3d} children  |  "
          f"Route distance: {route_dist:,.0f} m")

print(f"\nTotal route distance (all buses): {total_distance:,.0f} m")

np.savez(
    "tsp_result.npz",
    **{f"route_{c}": np.array(r) for c, r in enumerate(routes)},
    k              = k,
    total_distance = total_distance,
)
print("Routes saved to tsp_result.npz")

# Plot
fig, ax    = plt.subplots(figsize=(14, 14))
cmap       = plt.colormaps.get_cmap("tab20").resampled(k)
x_children = x[children_idx]
y_children = y[children_idx]

for c in range(k):
    mask  = labels == c
    color = cmap(c)
    ax.scatter(x_children[mask], y_children[mask], color=color, s=20,
               label=f"Bus {c} ({mask.sum()} children)")
    route = routes[c]
    ax.plot([x[i] for i in route], [y[i] for i in route],
            color=color, linewidth=1.2, alpha=0.7)
    for stop_num, node_idx in enumerate(route):
        ax.annotate(str(stop_num), (x[node_idx], y[node_idx]),
                    fontsize=5, fontweight="bold", color=color,
                    ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")
    mi = medoids[c]
    ax.scatter(x_children[mi], y_children[mi], color=color, s=120,
               marker="D", edgecolors="black", linewidths=1.0)

ax.scatter(x[origin_idx], y[origin_idx], marker="s", color="red",
           s=80, zorder=5, label="School")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13, crs="EPSG:4326")
out_map = "tsp_routes_map.png"
fig.savefig(out_map, dpi=150, bbox_inches="tight")
print(f"Map saved to {out_map}")
plt.close(fig)
print("Done.")

"""
clustering.py
=============
Stage 2 (legacy / standalone) — Capacitated k-medoids clustering.

This script runs the global clustering WITHOUT zone partitioning.
If you are using zone_partitioning.py, you do NOT need to run this separately —
the same logic is embedded inside zone_partitioning.py per zone.

Keep this file for reference or for running a non-zoned baseline.

Run:
    python src/clustering.py
"""

import numpy as np
import kmedoids
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx

# Load scenario data
data         = np.load("scenario_data.npz")
dist_matrix  = data["dist_matrix"]
x            = data["x"]
y            = data["y"]
origin_idx   = int(data["origin_index"])
bus_capacity = int(data["bus_capacity"])

n_children  = dist_matrix.shape[0] - 1
children_idx = [i for i in range(dist_matrix.shape[0]) if i != origin_idx]
D            = dist_matrix[np.ix_(children_idx, children_idx)]
x_children   = x[children_idx]
y_children   = y[children_idx]

# Symmetrise
D = (D + D.T) / 2.0

k = int(np.ceil(n_children / bus_capacity))
print(f"Children: {n_children}  |  Bus capacity: {bus_capacity}  |  Clusters (k): {k}")


def capacitated_assign(D, medoids, capacity):
    n = D.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge("s", ("c", i), capacity=1, weight=0)
    for i in range(n):
        for m_idx, m in enumerate(medoids):
            cost = int(D[i, m] * 1000)
            G.add_edge(("c", i), ("m", m_idx), capacity=1, weight=cost)
    for m_idx in range(len(medoids)):
        G.add_edge(("m", m_idx), "t", capacity=capacity, weight=0)
    G.nodes["s"]["demand"] = -n
    G.nodes["t"]["demand"] = n
    flow_dict   = nx.min_cost_flow(G)
    labels      = np.empty(n, dtype=int)
    total_cost  = 0.0
    for i in range(n):
        for m_idx in range(len(medoids)):
            if flow_dict[("c", i)].get(("m", m_idx), 0) == 1:
                labels[i]   = m_idx
                total_cost += D[i, medoids[m_idx]]
                break
    return labels, total_cost


def recompute_medoids(D, labels, k):
    medoids = []
    for c in range(k):
        members = np.where(labels == c)[0]
        sub     = D[np.ix_(members, members)]
        best    = members[np.argmin(sub.sum(axis=1))]
        medoids.append(int(best))
    return medoids


result  = kmedoids.fasterpam(D, k, random_state=42)
medoids = list(result.medoids)
print(f"Initial PAM cost: {result.loss:.1f} m")

max_iter = 50
for iteration in range(1, max_iter + 1):
    labels, total_cost = capacitated_assign(D, medoids, bus_capacity)
    new_medoids        = recompute_medoids(D, labels, k)
    if new_medoids == medoids:
        print(f"Converged after {iteration} iteration(s)  |  Total cost: {total_cost:.1f} m")
        break
    medoids = new_medoids
else:
    print(f"Stopped after {max_iter} iterations  |  Total cost: {total_cost:.1f} m")

medoid_indices = medoids

print(f"Cluster sizes (bus_capacity={bus_capacity}):")
for c in range(k):
    size = int(np.sum(labels == c))
    print(f"  Cluster {c:2d}: {size} children  (medoid child index {medoid_indices[c]})")

out_file = "clustering_result.npz"
np.savez(
    out_file,
    labels      = labels,
    medoids     = np.array(medoid_indices),
    k           = k,
    total_cost  = total_cost,
    children_idx= np.array(children_idx),
)
print(f"Clustering saved to {out_file}")

# Plot
fig, ax = plt.subplots(figsize=(14, 14))
cmap = plt.cm.get_cmap("tab20", k)
for c in range(k):
    mask = labels == c
    ax.scatter(x_children[mask], y_children[mask], color=cmap(c), s=20,
               label=f"Cluster {c} ({mask.sum()})")
    mi = medoid_indices[c]
    ax.scatter(x_children[mi], y_children[mi], color=cmap(c), s=120,
               marker="D", edgecolors="black", linewidths=1.0)

ax.scatter(x[origin_idx], y[origin_idx], marker="s", color="red",
           s=80, zorder=5, label="School")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13, crs="EPSG:4326")
out_map = "clustering_map.png"
fig.savefig(out_map, dpi=150, bbox_inches="tight")
print(f"Map saved to {out_map}")
plt.close(fig)
print("Done.")

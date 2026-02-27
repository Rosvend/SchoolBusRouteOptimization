import numpy as np
import kmedoids
import matplotlib.pyplot as plt
import contextily as ctx

# ── Load scenario data ────────────────────────────────────────────────
data = np.load("scenario_data.npz")
dist_matrix = data["dist_matrix"]
x = data["x"]
y = data["y"]
origin_idx = int(data["origin_index"])
bus_capacity = int(data["bus_capacity"])

n_children = dist_matrix.shape[0] - 1  # last row/col is the school

# Extract the children-only sub-matrix (exclude the origin/school)
children_idx = [i for i in range(dist_matrix.shape[0]) if i != origin_idx]
D = dist_matrix[np.ix_(children_idx, children_idx)]
x_children = x[children_idx]
y_children = y[children_idx]

# Symmetrise (Dijkstra on a directed graph may give asymmetric costs)
D = (D + D.T) / 2.0

# determine k from bus capacity 
k = int(np.ceil(n_children / bus_capacity))
print(f"Children: {n_children}  |  Bus capacity: {bus_capacity}  |  Clusters (k): {k}")

#kmedoids clustering
result = kmedoids.fasterpam(D, k, random_state=42)

labels = result.labels
medoid_indices = result.medoids
total_cost = result.loss

print(f"PAM converged  |  Total cost: {total_cost:.1f} m")
for c in range(k):
    size = int(np.sum(labels == c))
    print(f"  Cluster {c:2d}: {size} children  (medoid child index {medoid_indices[c]})")

#save results
out_file = "clustering_result.npz"
np.savez(
    out_file,
    labels=labels,
    medoids=np.array(medoid_indices),
    k=k,
    total_cost=total_cost,
    children_idx=np.array(children_idx),
)
print(f"Clustering saved to {out_file}")

# plot
fig, ax = plt.subplots(figsize=(14, 14))

cmap = plt.cm.get_cmap("tab20", k)
for c in range(k):
    mask = labels == c
    ax.scatter(
        x_children[mask], y_children[mask],
        color=cmap(c), s=20, label=f"Cluster {c} ({mask.sum()})",
    )
    # highlight medoid
    mi = medoid_indices[c]
    ax.scatter(
        x_children[mi], y_children[mi],
        color=cmap(c), s=120, marker="D", edgecolors="black", linewidths=1.0,
    )

# school / depot
ax.scatter(x[origin_idx], y[origin_idx], marker="s", color="red", s=80,
           zorder=5, label="School")

ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15, crs="EPSG:4326")

out_map = "clustering_map.png"
fig.savefig(out_map, dpi=150, bbox_inches="tight")
print(f"Map saved to {out_map}")
plt.close(fig)

print("Done.")

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

from MultiobjectiveSectorial import GeographicClusterer, ClusteringConfig, prepare_data_from_graph



# ============================
# 1. CARGAR DATOS
# ============================

print("Cargando scenario_data.npz...")
data = np.load("scenario_data.npz")

dist_matrix = data["dist_matrix"]
x = data["x"]
y = data["y"]
origin_idx = int(data["origin_index"])
bus_capacity = int(data["bus_capacity"])

n = len(x)
node_ids = list(range(n))

print(f"Nodos: {n}")
print(f"Índice colegio: {origin_idx}")
print(f"Capacidad bus: {bus_capacity}")

# 🔍 DIAGNÓSTICO: Ver rango de coordenadas
print(f"\n🔍 Rango de coordenadas:")
print(f"   X: [{x.min():.6f}, {x.max():.6f}]")
print(f"   Y: [{y.min():.6f}, {y.max():.6f}]")
print(f"   Colegio: ({x[origin_idx]:.6f}, {y[origin_idx]:.6f})")


# ============================
# 2. DETECTAR CRS DE ORIGEN
# ============================

if np.nanmax(np.abs(x)) > 1000 and np.nanmax(np.abs(y)) > 1000:
    source_crs = "EPSG:32618"
    print("\n✅ Coordenadas detectadas como proyectadas en metros → usando EPSG:32618")
else:
    source_crs = "EPSG:4326"
    print("\n✅ Coordenadas detectadas como geográficas → usando EPSG:4326")


# ============================
# 3. PREPARAR DATOS
# ============================

node_ids, coords, school_idx, distance_matrix, distances_to_school = \
    prepare_data_from_graph(
        ni=node_ids,
        xi=x,  # Ahora x = longitud
        yi=y,  # Ahora y = latitud
        dist_matrix=dist_matrix,
        origin_node=node_ids[origin_idx]
    )


# ============================
# 4. CONFIGURAR MODELO
# ============================

config = ClusteringConfig(
    capacity=bus_capacity,
    alpha=0.7,
    beta=0.3,
    refinement_iterations=5
)

clusterer = GeographicClusterer(config)


# ============================
# 5. ENTRENAR
# ============================

print("\n🔄 Ejecutando clustering...")
clusterer.fit(
    node_ids=node_ids,
    coords=coords,
    school_idx=school_idx,
    distance_matrix=distance_matrix,
    distances_to_school=distances_to_school
)

labels = clusterer.labels_


# ============================
# 6. DEBUG
# ============================

print("\n📊 RESULTADOS DEL CLUSTERING:")
unique = np.unique(labels)
print(f"   Clusters encontrados: {unique}")

for c in unique:
    if c >= 0:  # Excluir el colegio (c=-1)
        count = np.sum(labels == c)
        print(f"   Ruta {c+1}: {count} paradas")


# ============================
# 7. GRAFICAR (MAPA REAL)
# ============================

print("\n🗺️  Generando visualización sobre mapa de Medellín...")

gdf_wgs84 = gpd.GeoDataFrame(
    {"cluster": labels},
    geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
    crs=source_crs
)

gdf_mercator = gdf_wgs84.to_crs(epsg=3857)
print(f"   CRS reproyectado: {gdf_mercator.crs}")
print(f"   Bounds Web Mercator: {gdf_mercator.total_bounds}")

fig, ax = plt.subplots(figsize=(16, 14))

clusters_validos = sorted(gdf_mercator[gdf_mercator["cluster"] >= 0]["cluster"].unique())
n_clusters = len(clusters_validos)

if len(clusters_validos) > 0:
    cmap = plt.cm.get_cmap("tab20", len(clusters_validos))

    for i, c in enumerate(clusters_validos):
        subset = gdf_mercator[gdf_mercator["cluster"] == c]
        subset.plot(
            ax=ax,
            markersize=80,
            color=cmap(i),
            label=f"Cluster {c + 1}",
            alpha=0.75,
            edgecolor="black",
            zorder=3
        )

school_point = gdf_mercator.iloc[school_idx]
gpd.GeoSeries([school_point.geometry], crs=gdf_mercator.crs).plot(
    ax=ax,
    color="red",
    markersize=320,
    marker="*",
    label="Colegio",
    zorder=10,
    edgecolor="darkred",
    linewidth=1.5
)

try:
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        crs=gdf_mercator.crs
    )
    print("   ✅ Mapa base agregado exitosamente")
except Exception as e:
    print(f"   ⚠️  Error al agregar mapa base: {e}")
    ax.set_facecolor("#e8e8e8")

minx, miny, maxx, maxy = gdf_mercator.total_bounds
padding_x = (maxx - minx) * 0.08 if maxx > minx else 200
padding_y = (maxy - miny) * 0.08 if maxy > miny else 200

ax.set_xlim(minx - padding_x, maxx + padding_x)
ax.set_ylim(miny - padding_y, maxy + padding_y)

ax.set_title(
    f"Segmentación de Rutas Escolares - Medellín, Colombia\n"
    f"{n_clusters} rutas optimizadas | Capacidad máxima: {bus_capacity} estudiantes/bus",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    fontsize=10,
    framealpha=0.95,
    edgecolor="black",
    title="Rutas",
    title_fontsize=12
)

plt.tight_layout()


# ============================
# 8. GUARDAR + MOSTRAR
# ============================

output_file = "clusters_medellin_rutas_escolares.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n💾 Imagen guardada en: {output_file}")

plt.show(block=True)


# ============================
# 9. ESTADÍSTICAS FINALES
# ============================

print("\n" + "="*70)
print("📊 ESTADÍSTICAS DE CLUSTERING")
print("="*70)

stats = clusterer.get_cluster_stats()

print(f"\n🚌 RESUMEN DE RUTAS:")
print(f"   Total de rutas generadas: {stats['n_clusters']}")
print(f"   Tamaños de rutas: {stats['cluster_sizes']}")
print(f"   Tamaño promedio: {np.mean(stats['cluster_sizes']):.1f} paradas/ruta")
print(f"   Tamaño mínimo: {min(stats['cluster_sizes'])} paradas")
print(f"   Tamaño máximo: {max(stats['cluster_sizes'])} paradas")
print(f"   Desviación estándar: {np.std(stats['cluster_sizes']):.2f}")

print(f"\n📏 DISTANCIAS:")
print(f"   Distancia intra-cluster promedio: {np.mean(stats['avg_intra_cluster_distance']):.2f} unidades")
print(f"   Distancia promedio al colegio: {np.mean(stats['avg_distance_to_school']):.2f} unidades")

print(f"\n⚙️  PARÁMETROS USADOS:")
print(f"   α (peso angular): {config.alpha}")
print(f"   β (peso radial): {config.beta}")
print(f"   Capacidad máxima: {config.capacity}")
print(f"   Iteraciones de refinamiento: {config.refinement_iterations}")

print("\n✅ Proceso completado exitosamente")
print("="*70)
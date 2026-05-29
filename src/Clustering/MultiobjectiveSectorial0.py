# clustering_pipeline.py
"""
Pipeline de Clustering Geográfico para Rutas de Transporte Escolar.
Adaptado para trabajar con grafos de NetworkX y coordenadas geográficas reales.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ClusteringConfig:
    """Configuración del algoritmo de clustering"""
    capacity: int  # Capacidad máxima por cluster
    alpha: float = 0.6  # Peso del componente angular [0,1]
    beta: float = 0.4   # Peso del componente radial [0,1]
    distance_normalization: str = 'minmax'  # 'minmax' o 'zscore'
    refinement_iterations: int = 3  # Iteraciones de refinamiento
    sector_penalty: int = 0  # Penalización para mantener sectorización inicial (>=0)
    
    def __post_init__(self):
        assert abs(self.alpha + self.beta - 1.0) < 1e-6, "alpha + beta debe ser 1.0"
        assert 0 <= self.alpha <= 1, "alpha debe estar en [0,1]"


class GeographicClusterer:
    """
    Clustering geográfico multiobjetivo para rutas de transporte.
    
    Combina información angular y radial respecto a un origen común,
    con restricciones de capacidad y refinamiento basado en distancia real.
    """
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.node_ids = None  # IDs originales de los nodos del grafo
        self.coords = None
        self.school_idx = None
        self.distance_matrix = None
        self.distances_to_school = None
        self.labels_ = None
        self.n_clusters_ = None
        
    def fit(self, 
            node_ids: List,
            coords: np.ndarray,
            school_idx: int,
            distance_matrix: np.ndarray,
            distances_to_school: np.ndarray) -> 'GeographicClusterer':
        """
        Ejecuta el pipeline completo de clustering.
        
        Parameters:
        -----------
        node_ids : List
            Lista de IDs de nodos (del grafo NetworkX)
            Ejemplo: [123, 456, 789, ..., origin_node]
        coords : np.ndarray (n_nodes, 2)
            Coordenadas (lon, lat) o (x, y) de cada nodo
        school_idx : int
            Índice del nodo que representa el colegio (típicamente el último)
        distance_matrix : np.ndarray (n_nodes, n_nodes)
            Matriz de distancias reales en red vial (Dijkstra)
        distances_to_school : np.ndarray (n_nodes,)
            Vector de distancias desde cada nodo al colegio
        """
        self.node_ids = node_ids
        self.coords = coords
        self.school_idx = school_idx
        self.distance_matrix = distance_matrix
        self.distances_to_school = distances_to_school
        
        print(f"Iniciando clustering con {len(node_ids)} nodos...")
        print(f"   - Nodo colegio: {node_ids[school_idx]}")
        print(f"   - Coordenadas colegio: ({coords[school_idx, 0]:.6f}, {coords[school_idx, 1]:.6f})")
        
        # 1. Calcular ángulos y distancias normalizadas
        angles = self._compute_angles()
        normalized_distances = self._normalize_distances()

        # 2. Calcular score multiobjetivo y etiquetas iniciales por barrido angular
        scores = self._compute_multiobjective_score(angles, normalized_distances)
        initial_labels = self._angular_sweep_assignment(angles, scores)

        # determinar número de clusters
        n_nodes = len(node_ids) - 1  # sin colegio
        k = int(np.ceil(n_nodes / self.config.capacity))

        print(f"Clusters (k): {k}")

        # medoids iniciales a partir de labels (recompute similar a la otra versión)
        medoids = self._recompute_medoids(initial_labels, k)

        # boucle iterativa: asignación por flow + recomputo de medoids
        labels = initial_labels.copy()

        for it in range(5):
            print(f"Iteración {it+1}")
            labels = self._assign_with_min_cost_flow(medoids, initial_labels)
            new_medoids = self._recompute_medoids(labels, k)
            if new_medoids == medoids:
                print("Convergió")
                break
            medoids = new_medoids

        self.labels_ = labels
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))

        print(f"Clustering completado: {self.n_clusters_} clusters generados")
        return self
    
    def _compute_angles(self) -> np.ndarray:
        """
        Calcula el ángulo de cada nodo respecto al colegio.
        
        Returns:
        --------
        angles : np.ndarray (n_nodes,)
            Ángulos en radianes [-π, π]
        """
        school_coords = self.coords[self.school_idx]
        
        # Vector desde colegio a cada nodo
        vectors = self.coords - school_coords
        
        # Ángulo usando arctan2 (maneja correctamente todos los cuadrantes)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # El colegio tiene ángulo 0 por convención
        angles[self.school_idx] = 0.0
        
        return angles
    
    def _normalize_distances(self) -> np.ndarray:
        """
        Normaliza las distancias al colegio.
        
        Returns:
        --------
        normalized : np.ndarray (n_nodes,)
            Distancias normalizadas en [0, 1]
        """
        distances = self.distances_to_school.copy()
        
        # Evitar división por cero
        distances[self.school_idx] = 0.0
        
        if self.config.distance_normalization == 'minmax':
            # Min-Max scaling
            d_min = distances.min()
            d_max = distances.max()
            if d_max - d_min > 0:
                normalized = (distances - d_min) / (d_max - d_min)
            else:
                normalized = np.zeros_like(distances)
                
        elif self.config.distance_normalization == 'zscore':
            # Z-score normalization
            mean = distances.mean()
            std = distances.std()
            if std > 0:
                normalized = (distances - mean) / std
                # Llevar a [0, 1] aproximadamente
                normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
            else:
                normalized = np.zeros_like(distances)
        else:
            raise ValueError(f"Método de normalización desconocido: {self.config.distance_normalization}")
        
        return normalized

    def _recompute_medoids(self, labels: np.ndarray, k: int) -> List[int]:
        """
        Recomputa medoids para k clusters usando combinación de costo interno
        y distancia al colegio (similar a la versión experimental).
        """
        medoids = []
        lambda_acc = 0.5

        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue

            sub_dist_matrix = self.distance_matrix[np.ix_(members, members)]
            dist_to_school = self.distances_to_school[members]

            internal_costs = sub_dist_matrix.sum(axis=1)
            combined_score = (1 - lambda_acc) * internal_costs + lambda_acc * dist_to_school

            best_idx = members[int(np.argmin(combined_score))]
            medoids.append(int(best_idx))

        return medoids
    
    def _compute_multiobjective_score(self, 
                                      angles: np.ndarray,
                                      normalized_distances: np.ndarray) -> np.ndarray:
        """
        Calcula el score multiobjetivo para ordenamiento.
        
        Formulación Matemática:
        ----------------------
        Score(i) = α · (θ_i + π)/(2π) + β · d_norm(i)
        
        Donde:
        - θ_i ∈ [-π, π]: ángulo del nodo i respecto al colegio
        - (θ_i + π)/(2π): normaliza ángulo a [0, 1]
        - d_norm(i) ∈ [0, 1]: distancia normalizada al colegio
        - α: peso del componente angular (prioriza sectorización)
        - β: peso del componente radial (prioriza cercanía al colegio)
        - α + β = 1
        
        Returns:
        --------
        scores : np.ndarray (n_nodes,)
            Scores para ordenamiento
        """
    
        alpha = self.config.alpha
        beta = self.config.beta

        # Representación circular
        cos_theta = np.cos(angles)
        sin_theta = np.sin(angles)

        # Proyección angular (orden circular suave)
        angular_component = np.arctan2(sin_theta, cos_theta)
        angular_component = (angular_component + np.pi) / (2 * np.pi)

        scores = alpha * angular_component + beta * normalized_distances
        return scores
    
    def _angular_sweep_assignment(self, 
                                  angles: np.ndarray,
                                  scores: np.ndarray) -> np.ndarray:
        """
        Asigna nodos a clusters mediante barrido angular con control de capacidad.
        
        Returns:
        --------
        labels : np.ndarray (n_nodes,)
            Etiquetas de cluster para cada nodo
        """
        n_nodes = len(scores)
        labels = np.full(n_nodes, -1, dtype=int)

        active_nodes = np.arange(n_nodes)
        active_nodes = active_nodes[active_nodes != self.school_idx]

        sorted_indices = active_nodes[np.argsort(scores[active_nodes])]

        current_cluster = 0
        current_count = 0

        for idx in sorted_indices:
            #Permite pequeño margen
            if current_count >= self.config.capacity:
                current_cluster += 1
                current_count = 0

            labels[idx] = current_cluster
            current_count += 1

        labels[self.school_idx] = -1
        return labels
    
    def _refine_clusters(self):
        """
        Refinamiento local usando la matriz de distancias reales.
        """
        for _ in range(self.config.refinement_iterations):
            improved = False
            active_clusters = np.unique(self.labels_[self.labels_ >= 0])

            for node in range(len(self.labels_)):
                if node == self.school_idx:
                    continue

                current_cluster = self.labels_[node]
                current_cost = self._compute_node_cohesion(node, current_cluster)

                for other_cluster in active_clusters:
                    if other_cluster == current_cluster:
                        continue

                    #  Capacidad flexible
                    size = np.sum(self.labels_ == other_cluster)
                    if size > self.config.capacity + 2:
                        continue

                    new_cost = self._compute_node_cohesion(node, other_cluster)

                    if new_cost < current_cost * 0.9:
                        self.labels_[node] = other_cluster
                        improved = True
                        break

            if not improved:
                break
    
    def _compute_node_cohesion(self, node: int, cluster_id: int) -> float:
        """
        Calcula la cohesión de un nodo respecto a un cluster.
        
        Returns:
        --------
        cohesion : float
            Distancia promedio (menor es mejor)
        """
        cluster_nodes = np.where(self.labels_ == cluster_id)[0]

        if len(cluster_nodes) == 0:
            return np.inf

        intra = np.mean(self.distance_matrix[node, cluster_nodes])

        # Incluir distancia al colegio
        to_school = self.distances_to_school[node]

        # Penalización por dispersión
        dispersion = np.std(self.distance_matrix[np.ix_(cluster_nodes, cluster_nodes)])

        gamma = 0.3   # peso origen
        delta = 0.2   # peso dispersión

        return intra + gamma * to_school + delta * dispersion
    
    def get_cluster_stats(self) -> Dict:
        """
        Calcula estadísticas de los clusters generados.
        
        Returns:
        --------
        stats : dict
            Diccionario con métricas de calidad
        """
        active_clusters = np.unique(self.labels_[self.labels_ >= 0])
        
        stats = {
            'n_clusters': len(active_clusters),
            'cluster_sizes': [],
            'avg_intra_cluster_distance': [],
            'avg_distance_to_school': [],
            'cluster_node_ids': {}  # Nuevo: mapeo de cluster a node_ids originales
        }
        
        for cluster_id in active_clusters:
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            cluster_nodes = [self.node_ids[i] for i in cluster_indices]
            
            stats['cluster_sizes'].append(len(cluster_indices))
            stats['cluster_node_ids'][cluster_id] = cluster_nodes
            
            # Distancia intra-cluster promedio
            if len(cluster_indices) > 1:
                intra_distances = self.distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                avg_intra = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
            else:
                avg_intra = 0.0
            stats['avg_intra_cluster_distance'].append(avg_intra)
            
            # Distancia promedio al colegio
            avg_to_school = np.mean(self.distances_to_school[cluster_indices])
            stats['avg_distance_to_school'].append(avg_to_school)
        
        return stats
    
    def get_cluster_nodes(self, cluster_id: int) -> Tuple[List, np.ndarray]:
        """
        Obtiene los nodos de un cluster específico.
        
        Returns:
        --------
        node_ids : List
            IDs originales de los nodos del grafo
        indices : np.ndarray
            Índices en las estructuras internas
        """
        indices = np.where(self.labels_ == cluster_id)[0]
        node_ids = [self.node_ids[i] for i in indices]
        return node_ids, indices
    
    def visualize(self, figsize=(14, 10), save_path: Optional[str] = None):
        """
        Visualiza clusters sobre mapa real (OpenStreetMap).
        """
        if self.labels_ is None:
            raise ValueError("Ejecuta fit() antes de visualizar")

        # =========================
        # 1. Crear GeoDataFrame
        # =========================
        gdf = gpd.GeoDataFrame(
            {
                "cluster": self.labels_
            },
            geometry=[Point(xy) for xy in self.coords],
            crs="EPSG:4326"  # lat/lon
        )

        # =========================
        # 2. Reproyectar a Web Mercator
        # =========================
        gdf = gdf.to_crs(epsg=3857)

        # =========================
        # 3. Plot
        # =========================
        fig, ax = plt.subplots(figsize=figsize)

        active_clusters = gdf[gdf["cluster"] >= 0]["cluster"].unique()

        if len(active_clusters) == 0:
            print("No hay clusters válidos")
            return

        # colores
        cmap = plt.cm.get_cmap("tab20", len(active_clusters))

        for c in active_clusters:
            subset = gdf[gdf["cluster"] == c]
            subset.plot(
                ax=ax,
                markersize=40,
                color=cmap(c),
                label=f"Cluster {c} ({len(subset)})",
                alpha=0.7
            )

        # =========================
        # 4. Colegio
        # =========================
        school_point = gdf.iloc[self.school_idx]

        gpd.GeoSeries([school_point.geometry], crs=gdf.crs).plot(
            ax=ax,
            color="red",
            markersize=150,
            marker="*",
            label="Colegio"
        )

        # =========================
        # 5. Basemap (Medellín)
        # =========================
        ctx.add_basemap(
            ax,
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=14
        )

        # =========================
        # 6. Estética
        # =========================
        ax.set_title("Clusters geográficos sobre Medellín")
        ax.set_axis_off()
        ax.legend()

        plt.tight_layout()

        # guardar
        if save_path is None:
            save_path = "clusters_mapa_medellin.png"

        plt.savefig(save_path, dpi=150)
        print(f"🖼 Guardado en {save_path}")

        plt.show()

    def _assign_with_min_cost_flow(self, medoid_indices):
        """
        Asigna nodos a clusters respetando capacidad usando min-cost flow.
        """

        import networkx as nx

        n = len(self.coords)
        capacity = self.config.capacity
        k = len(medoid_indices)

        G = nx.DiGraph()

        # source → nodos
        for i in range(n):
            if i == self.school_idx:
                continue
            G.add_edge("s", ("n", i), capacity=1, weight=0)

        # nodos → clusters (costo = distancia)
        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx, m in enumerate(medoid_indices):
                cost = int(self.distance_matrix[i, m])  # Dijkstra real
                G.add_edge(("n", i), ("c", c_idx), capacity=1, weight=cost)

        # clusters → sink (capacidad bus)
        for c_idx in range(k):
            G.add_edge(("c", c_idx), "t", capacity=capacity, weight=0)

        # demandas
        G.nodes["s"]["demand"] = -(n - 1)  # sin colegio
        G.nodes["t"]["demand"] = (n - 1)

        flow = nx.min_cost_flow(G)

        labels = np.full(n, -1)

        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx in range(k):
                if flow[("n", i)].get(("c", c_idx), 0) == 1:
                    labels[i] = c_idx
                    break

        return labels

    def _assign_with_min_cost_flow(self, medoid_indices, initial_labels=None):
        """
        Variante que admite penalización para mantener la sectorización inicial.
        Si `self.config.sector_penalty > 0` y se pasa `initial_labels`, se suma
        la penalización al costo cuando la asignación difiere.
        """
        import networkx as nx

        n = len(self.coords)
        capacity = self.config.capacity
        k = len(medoid_indices)

        G = nx.DiGraph()

        # source → nodos
        for i in range(n):
            if i == self.school_idx:
                continue
            G.add_edge("s", ("n", i), capacity=1, weight=0)

        # nodos → clusters (costo = distancia + posible penalización)
        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx, m in enumerate(medoid_indices):
                cost = int(self.distance_matrix[i, m])
                if initial_labels is not None and self.config.sector_penalty > 0:
                    if initial_labels[i] != c_idx:
                        cost += int(self.config.sector_penalty)

                G.add_edge(("n", i), ("c", c_idx), capacity=1, weight=cost)

        # clusters → sink (capacidad bus)
        for c_idx in range(k):
            G.add_edge(("c", c_idx), "t", capacity=capacity, weight=0)

        # demandas
        G.nodes["s"]["demand"] = -(n - 1)  # sin colegio
        G.nodes["t"]["demand"] = (n - 1)

        flow = nx.min_cost_flow(G)

        labels = np.full(n, -1)

        for i in range(n):
            if i == self.school_idx:
                continue
            for c_idx in range(k):
                if flow[("n", i)].get(("c", c_idx), 0) == 1:
                    labels[i] = c_idx
                    break

        return labels
    
    def _get_initial_medoids(self, k):
        """
        Selecciona k nodos como centros iniciales usando ángulo.
        """
        angles = self._compute_angles()

        valid_nodes = [i for i in range(len(angles)) if i != self.school_idx]
        sorted_nodes = sorted(valid_nodes, key=lambda i: angles[i])

        # dividir en k segmentos
        medoids = []
        step = len(sorted_nodes) // k

        for i in range(k):
            medoids.append(sorted_nodes[i * step])

        return medoids
    
# =============================================================================
# FUNCIÓN DE INTEGRACIÓN CON TU CÓDIGO
# =============================================================================

def prepare_data_from_graph(ni: List, 
                            xi: np.ndarray, 
                            yi: np.ndarray,
                            dist_matrix: np.ndarray,
                            origin_node) -> Tuple:
    """
    Prepara los datos desde tu estructura de grafo para el clustering.
    
    Parameters:
    -----------
    ni : List
        Lista de node IDs (children + origin_node al final)
    xi : np.ndarray
        Coordenadas x (longitud)
    yi : np.ndarray
        Coordenadas y (latitud)
    dist_matrix : np.ndarray
        Matriz de distancias de Dijkstra
    origin_node : 
        ID del nodo origen (colegio)
    
    Returns:
    --------
    node_ids : List
        Lista de IDs de nodos
    coords : np.ndarray (n, 2)
        Coordenadas apiladas
    school_idx : int
        Índice del colegio
    distance_matrix : np.ndarray
        Matriz de distancias
    distances_to_school : np.ndarray
        Vector de distancias al colegio
    """
    # Preparar coordenadas
    coords = np.column_stack([xi, yi])
    
    # Encontrar índice del colegio
    school_idx = ni.index(origin_node)
    
    # Extraer distancias al colegio (fila/columna del origin_node)
    distances_to_school = dist_matrix[school_idx, :]
    
    return ni, coords, school_idx, dist_matrix, distances_to_school


# =============================================================================
# EJEMPLO DE USO CON TU ESTRUCTURA
# =============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("EJEMPLO DE INTEGRACIÓN CON TU CÓDIGO DE NETWORKX")
    print("="*80)
    print()
    
    # Simular tu estructura de datos
    # En tu código real, esto vendría de:
    # ni, xi, yi, dist_matrix, origin_node
    
    print("⚠️  Este es un ejemplo. Usa tu código real con NetworkX.")
    print()
    print("Ejemplo de integración:")
    print("""
    # Después de tu código de generación de datos:
    # ni = [..., origin_node]  # Lista de node IDs
    # xi, yi = coordenadas
    # dist_matrix = matriz de Dijkstra
    # origin_node = ID del nodo colegio
    
    # Preparar datos
    node_ids, coords, school_idx, distance_matrix, distances_to_school = \\
        prepare_data_from_graph(ni, xi, yi, dist_matrix, origin_node)
    
    # Configurar clustering
    config = ClusteringConfig(
        capacity=15,
        alpha=0.7,
        beta=0.3
    )
    
    # Ejecutar clustering
    clusterer = GeographicClusterer(config)
    clusterer.fit(
        node_ids=node_ids,
        coords=coords,
        school_idx=school_idx,
        distance_matrix=distance_matrix,
        distances_to_school=distances_to_school
    )
    
    # Ver resultados
    stats = clusterer.get_cluster_stats()
    print(f"Clusters generados: {stats['n_clusters']}")
    
    # Visualizar
    clusterer.visualize()
    
    # Obtener nodos de un cluster específico
    cluster_0_nodes, cluster_0_indices = clusterer.get_cluster_nodes(0)
    print(f"Cluster 0 contiene los nodos: {cluster_0_nodes}")
    """)
    
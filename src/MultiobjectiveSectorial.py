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
    
    def __post_init__(self):
        assert abs(self.alpha + self.beta - 1.0) < 1e-6, "alpha + beta debe ser 1.0"
        assert 0 <= self.alpha <= 1, "alpha debe estar en [0,1]"


import numpy as np
import networkx as nx

class GeographicClusterer:

    def __init__(self, config):
        self.config = config

    def get_cluster_stats(self):
        if not hasattr(self, "labels_"):
            raise AttributeError("The clusterer must be fit before requesting statistics")

        labels = np.asarray(self.labels_)
        active_clusters = np.array(sorted(c for c in np.unique(labels) if c >= 0))
        cluster_sizes = [int(np.sum(labels == c)) for c in active_clusters]

        intra_cluster_distances = []
        distance_to_school = []

        for c in active_clusters:
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue

            submatrix = self.distance_matrix[np.ix_(members, members)]
            if submatrix.size:
                mask = ~np.eye(len(members), dtype=bool)
                values = submatrix[mask]
                if values.size:
                    intra_cluster_distances.append(float(np.mean(values)))

            distance_to_school.append(float(np.mean(self.distances_to_school[members])))

        return {
            "n_clusters": int(len(active_clusters)),
            "cluster_sizes": cluster_sizes,
            "avg_intra_cluster_distance": intra_cluster_distances,
            "avg_distance_to_school": distance_to_school,
            "school_idx": int(self.school_idx),
        }

    # =========================
    # ANGULOS
    # =========================
    def _compute_angles(self):
        school = self.coords[self.school_idx]
        vec = self.coords - school
        angles = np.arctan2(vec[:, 1], vec[:, 0])
        angles[self.school_idx] = 0.0
        return angles

    # =========================
    # NORMALIZACIÓN DISTANCIA
    # =========================
    def _normalize_distances(self):
        d = self.distances_to_school.copy()
        d[self.school_idx] = 0.0

        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 0:
            return (d - d_min) / (d_max - d_min)
        return np.zeros_like(d)

    # =========================
    # SCORE MULTIOBJETIVO
    # =========================
    def _compute_score(self, angles, dist_norm):
        a = (angles + np.pi) / (2 * np.pi)
        return self.config.alpha * a + self.config.beta * dist_norm

    # =========================
    # SWEEP SECTORIAL
    # =========================
    def _angular_sweep(self, scores):
        n = len(scores)
        labels = np.full(n, -1)

        idx = np.arange(n)
        idx = idx[idx != self.school_idx]

        order = idx[np.argsort(scores[idx])]

        cluster = 0
        count = 0

        for i in order:
            if count >= self.config.capacity:
                cluster += 1
                count = 0

            labels[i] = cluster
            count += 1

        return labels

    # =========================
    # MEDOIDS
    # =========================
    def _recompute_medoids(self, labels, k):
        """
        Recomputa los medoids integrando la accesibilidad hacia el colegio.
        Busca un equilibrio entre la cohesión del cluster y la eficiencia de la ruta.
        """
        medoids = []

        for c in range(k):
            members = np.where(labels == c)[0]

            if len(members) == 0:
                # Caso de seguridad: si el cluster está vacío, mantenemos una referencia
                continue

            # 1. Matriz de distancias interna del cluster
            sub_dist_matrix = self.distance_matrix[np.ix_(members, members)]
            
            # 2. Distancia de cada miembro hacia el colegio
            dist_to_school = self.distances_to_school[members]

            # 3. Cálculo del Score de Medoid
            # Buscamos minimizar: Distancia_Promedio_Interna + Factor_Inercia_Colegio
            # Un factor de 0.5 da igual importancia a la cohesión que a la cercanía al destino
            lambda_acc = 0.5 
            
            # Normalizamos para que ambas métricas sean comparables
            internal_costs = sub_dist_matrix.sum(axis=1)
            
            # Combinación lineal para encontrar el "Líder de Ruta"
            combined_score = (1 - lambda_acc) * internal_costs + lambda_acc * dist_to_school
            
            best_idx = members[np.argmin(combined_score)]
            medoids.append(int(best_idx))

        return medoids

    # =========================
    # MIN COST FLOW
    # =========================
    def _assign_with_flow(self, medoids, initial_labels):

        n = len(self.coords)
        k = len(medoids)
        cap = self.config.capacity

        G = nx.DiGraph()

        # source → nodos
        for i in range(n):
            if i == self.school_idx:
                continue
            G.add_edge("s", ("n", i), capacity=1, weight=0)

        # nodos → clusters
        for i in range(n):
            if i == self.school_idx:
                continue

            for c_idx, m in enumerate(medoids):

                # distancia real Dijkstra
                cost = self.distance_matrix[i, m]

                # 🔥 penalización sectorial (TU IDEA)
                if initial_labels[i] != c_idx:
                    cost += 10000  # puedes ajustar

                G.add_edge(
                    ("n", i),
                    ("c", c_idx),
                    capacity=1,
                    weight=int(cost)
                )

        # clusters → sink
        for c in range(k):
            G.add_edge(("c", c), "t", capacity=cap, weight=0)

        # demandas
        G.nodes["s"]["demand"] = -(n - 1)
        G.nodes["t"]["demand"] = (n - 1)

        flow = nx.min_cost_flow(G)

        labels = np.full(n, -1)

        for i in range(n):
            if i == self.school_idx:
                continue

            for c in range(k):
                if flow[("n", i)].get(("c", c), 0) == 1:
                    labels[i] = c
                    break

        return labels

    # =========================
    # FIT FINAL
    # =========================
    def fit(self, node_ids, coords, school_idx, distance_matrix, distances_to_school):

        self.node_ids = node_ids
        self.coords = coords
        self.school_idx = school_idx
        self.distance_matrix = distance_matrix
        self.distances_to_school = distances_to_school

        n = len(coords) - 1
        k = int(np.ceil(n / self.config.capacity))

        print(f"Clusters (k): {k}")

        # 1. ángulo + distancia
        angles = self._compute_angles()
        dist_norm = self._normalize_distances()

        # 2. score multiobjetivo
        scores = self._compute_score(angles, dist_norm)

        # 3. sectorización base
        initial_labels = self._angular_sweep(scores)

        # 4. medoids iniciales
        medoids = self._recompute_medoids(initial_labels, k)

        # 5. iteración con flow
        for it in range(5):
            print(f"Iteración {it+1}")

            labels = self._assign_with_flow(medoids, initial_labels)
            new_medoids = self._recompute_medoids(labels, k)

            if new_medoids == medoids:
                print("Convergió")
                break

            medoids = new_medoids

        self.labels_ = labels
        return self
    
def prepare_data_from_graph(ni, xi, yi, dist_matrix, origin_node):

    coords = np.column_stack((xi, yi)).astype(float)

    if origin_node not in ni:
        raise ValueError("origin_node no está en ni")

    school_idx = ni.index(origin_node)

    distances_to_school = dist_matrix[school_idx, :].copy()
    distances_to_school[school_idx] = 0.0

    # simetrizar por seguridad
    if not np.allclose(dist_matrix, dist_matrix.T):
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

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
    
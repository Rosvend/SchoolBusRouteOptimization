# Capacitated School Bus Routing via Clustering and Network-Based Optimization

## Overview

This repository implements a heuristic pipeline for school bus route planning in Medellin using a real road network.

The workflow is split into three stages:

1. Scenario generation from OpenStreetMap road data.
2. Capacitated clustering of student stops using k-medoids with min-cost flow assignment.
3. Per-cluster route optimization with Google OR-Tools (TSP-style routing).

The current setup models one school/depot and bus-capacity-constrained assignments, and computes one closed route per bus cluster.

## How It Works

### 1. Road-network distances

Let the road network be a directed weighted graph:

$$
G = (V, E), \quad w_e > 0 \; \forall e \in E
$$

For sampled nodes $i, j \in V$, the shortest-path distance is:

$$
d_{ij} = \min_{p \in \mathcal{P}(i,j)} \sum_{e \in p} w_e
$$

where $\mathcal{P}(i,j)$ is the set of directed paths from $i$ to $j$.

In clustering, the implementation symmetrizes distances:

$$
D^{\text{sym}}_{ij} = \frac{d_{ij} + d_{ji}}{2}
$$

### 2. Number of buses (clusters)

Given $n$ children and bus capacity $C$:

$$
k = \left\lceil \frac{n}{C} \right\rceil
$$

### 3. Capacitated assignment to medoids

Let $M = \{m_1, \dots, m_k\}$ be current medoids (child indices).
Define assignment binary variables:

$$
x_{ic} =
\begin{cases}
1, & \text{if child } i \text{ is assigned to cluster } c \\
0, & \text{otherwise}
\end{cases}
$$

Objective:

$$
\min \sum_{i=1}^{n} \sum_{c=1}^{k} D^{\text{sym}}_{i,m_c} \, x_{ic}
$$

Subject to:

$$
\sum_{c=1}^{k} x_{ic} = 1 \quad \forall i
$$

$$
\sum_{i=1}^{n} x_{ic} \le C \quad \forall c
$$

$$
x_{ic} \in \{0,1\}
$$

This assignment is solved through an equivalent min-cost flow network in the code.

### 4. Medoid recomputation

For each cluster $S_c = \{i : x_{ic}=1\}$, the new medoid is:

$$
m_c = \arg\min_{h \in S_c} \sum_{i \in S_c} D^{\text{sym}}_{ih}
$$

The algorithm alternates assignment and medoid update until medoids stop changing.

### 5. Per-cluster TSP routing

For each cluster, define node set:

$$
V_c = S_c \cup \{o\}
$$

where $o$ is the school/depot.

Using directed road distances $d_{ij}$ from the scenario matrix, OR-Tools solves a one-vehicle closed route minimizing total distance:

$$
\min \sum_{(i,j) \in V_c \times V_c} d_{ij} \, y_{ij}
$$

with standard routing constraints (exactly one visit per non-depot node, flow continuity, and start/end at depot).

Equivalent degree constraints for a TSP tour are:

$$
\sum_{j \in V_c,\, j \ne i} y_{ij} = 1 \quad \forall i \in V_c
$$

$$
\sum_{i \in V_c,\, i \ne j} y_{ij} = 1 \quad \forall j \in V_c
$$

plus subtour-elimination conditions.

### 6. Total fleet distance

If route $r_c = (r_{c,0}, r_{c,1}, \dots, r_{c,T_c})$ for cluster $c$ includes depot start/end, then:

$$
L_c = \sum_{t=0}^{T_c-1} d_{r_{c,t}, r_{c,t+1}}
$$

and total distance reported by the script is:

$$
L_{\text{total}} = \sum_{c=1}^{k} L_c
$$

## Repository Structure

```bash
├── src/
│   ├── data_generation.py
│   ├── clustering.py
│   └── tsp_solver.py
├── pyproject.toml
├── scenario_data.npz
├── clustering_result.npz
├── tsp_result.npz
└── cache
```

## Requirements

- Python 3.14+
- Dependencies are defined in pyproject.toml:
  - numpy
  - networkx
  - osmnx
  - kmedoids
  - ortools
  - matplotlib
  - contextily
  - tqdm

## Installation

With uv:

```bash
uv sync
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## How To Run

Run the pipeline in order from the repository root:

```bash
python src/data_generation.py
python src/clustering.py
python src/tsp_solver.py
```

# Capacitated School Bus Routing via Clustering and Network-Based Optimization

## Overview

This repository implements and benchmarks heuristic pipelines for school bus route planning in Medellín / Valle de Aburrá using a real road network from OpenStreetMap.

The project has two layers:

1. **Baseline pipeline** (`src/`) — a three-stage workflow: scenario generation, capacitated k-medoids clustering with min-cost flow assignment, and per-cluster TSP routing with Google OR-Tools.
2. **Benchmarking harness** (`src/benchmarks/`) — a unified evaluation framework that runs five routing algorithms over the same scenarios and reports comparable metrics (fleet distance, latency, coverage, silhouette, capacity violations).

Both layers model one school/depot and bus-capacity-constrained assignments, and compute one closed route per bus.

## Project Status

- The baseline pipeline (k-medoids + OR-Tools) is complete and produces full-coverage routes.
- A unified benchmarking harness compares six algorithms on identical scenarios:
  1. `kmedoids_ortools` — capacitated k-medoids (min-cost flow) + OR-Tools TSP per cluster.
  2. `sectorial` — angular/radial multi-objective sweep + min-cost-flow refinement + OR-Tools TSP.
  3. `cvrptw_zones` — CVRPTW with fixed geographic zones, time windows, and max route duration.
  4. `setcover_perchild` — per-child Set Cover ILP (PuLP/CBC) + nearest-neighbor tour.
  5. `setcover_grid` — grid-seeded Set Cover ILP + greedy TSP.
  6. `alns` — adaptive large-neighbourhood search: random/worst/Shaw removal with
     greedy and regret-2 reinsertion, under the same per-bus wall-clock budget as
     the clustering pipelines.
- `genetic` (flat-permutation GA) is still registered and runnable, but it is superseded
  by `alns` as the metaheuristic baseline and is out of the default sweep.
- Latest sweep results live in `results/` (`runs.csv`, `summary.md`, `scalability.png`, per-scenario maps under `plots/`).
- A talking-points write-up of the comparative evaluation is in `docs/BENCHMARK_INSIGHTS.md`; the LaTeX paper is under `docs/latex/`.

### Headline benchmark results

From an early sweep at N=100 students (mean over 3 seeds). **Stale**: predates the
11-seed sweep and the `sectorial`/`alns` strategies; `results/runs.csv` is authoritative.

| algo              | fleet distance (m) | latency (s) | buses | coverage | silhouette |
|-------------------|--------------------|-------------|-------|----------|------------|
| kmedoids_ortools  | 163,024            | 25.05       | 5.0   | 1.000    |  0.354     |
| cvrptw_zones      | 186,630            | 100.01      | 7.0   | 1.000    |  0.250     |
| genetic           | 202,934            | 22.32       | 5.0   | 1.000    | -0.076     |
| setcover_grid     | 209,615            | 0.68        | 6.0   | 1.000    |  0.262     |
| setcover_perchild | 213,575            | 0.04        | 6.0   | 1.000    |  0.302     |

`kmedoids_ortools` produces the shortest fleet distance and the most spatially compact clusters; the Set Cover variants are the speed/coverage sweet spot. See `docs/BENCHMARK_INSIGHTS.md` for a full discussion across N ∈ {50, 100, 200, 400}.

## Methodology (baseline pipeline)

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
│   ├── data_generation.py          # baseline: OSMnx scenario generator
│   ├── clustering.py               # baseline: capacitated k-medoids
│   ├── tsp_solver.py               # baseline: OR-Tools per-cluster TSP
│   └── benchmarks/                 # unified benchmarking harness
│       ├── run_benchmark.py        # CLI entry point
│       ├── core/                   # Scenario, Solver ABC, Solution, metrics
│       ├── adapters/               # 5 algorithm adapters
│       ├── data/                   # shared scenario generator + zones
│       └── viz/                    # plotter (single map + comparison grid)
├── results/
│   ├── runs.csv                    # one row per (algo, N, seed)
│   ├── summary.md                  # pivoted mean ± std tables
│   ├── scalability.png             # latency vs N (log-y)
│   └── plots/                      # per-scenario maps + 5-up grids
├── docs/
│   ├── BENCHMARK_INSIGHTS.md       # comparative evaluation write-up
│   └── latex/                      # paper source
├── pyproject.toml
├── scenario_data.npz
├── clustering_result.npz
├── tsp_result.npz
└── cache/                          # OSMnx graph cache
```

## Requirements

- Python 3.14+
- Dependencies are defined in `pyproject.toml`:
  - numpy, networkx, osmnx, kmedoids, ortools
  - pulp (Set Cover ILPs), shapely (zone polygons), scikit-learn (silhouette), pandas
  - matplotlib, contextily, tqdm

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

### Baseline pipeline (single algorithm, three scripts)

Run from the repository root:

```bash
python src/data_generation.py
python src/clustering.py
python src/tsp_solver.py
```

### Benchmark sweep (all algorithms)

```bash
# full default sweep: N ∈ {50, 100, 200, 400} × 3 seeds × 6 algorithms (~1 h)
python -m src.benchmarks.run_benchmark

# quick smoke test
python -m src.benchmarks.run_benchmark \
    --densities 100 --seeds 1 \
    --algos kmedoids_ortools alns
```

Outputs land in `results/` (CSV + summary + plots) and are written incrementally so a crash mid-sweep does not lose data.

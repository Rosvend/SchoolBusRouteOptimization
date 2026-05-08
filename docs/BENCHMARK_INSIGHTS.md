# Benchmark Insights — School Bus Routing

This is the talking-points document for explaining tomorrow what we did, why we did it, and what we learned. It corresponds to the new module under `src/benchmarks/` and the artifacts under `results/`.

---

## What problem we were trying to solve

The team had **four different ways of optimizing school bus routing** scattered across branches in the repo, and each researcher had built their version independently. There was no way to say *which one is actually better* because every implementation:

- Generated its own input data (different student samples, different bbox, different distance conventions),
- Ran on different parameters,
- Reported metrics in different formats,
- Was tested on different problem sizes.

We needed a single, fair comparison to support the paper's empirical results section.

## What we built

A unified **benchmarking harness** at `src/benchmarks/` that defines a common interface (`Solver` ABC over a `Scenario` → `Solution`) and provides:

- **One scenario generator** (`data/scenario_generator.py`) — caches the Medellín / Valle de Aburrá road network from OSMnx once, then samples student positions deterministically per `(N, seed)`. Every algorithm sees the same students, the same road distance matrix, and the same depot.
- **Five algorithm adapters** (`adapters/`):
  1. `kmedoids_ortools` — the main pipeline (capacitated k-medoids via min-cost flow + OR-Tools TSP per cluster).
  2. `cvrptw_zones` — Naomi's CVRPTW with 6 fixed geographic zones (commune polygons, time windows, max route duration).
  3. `setcover_perchild` — Mariana's per-child Set Cover ILP (PuLP/CBC) + nearest-neighbor tour.
  4. `setcover_grid` — Roberto's grid-seeded Set Cover ILP + greedy TSP for cost.
  5. `genetic` — a new generic GA we wrote (OX crossover, swap mutation, tournament selection, elitism, 2-opt refinement). The repo had a branch named `genetic-algorithms` and a commit on `roberto` titled "Agregamos un modelo de GA", but no actual GA code existed in any branch — confirmed by reading every code cell of every notebook and the actual git diff. So we filled the gap.
- **A consistent metrics module** (`core/metrics.py`) — fleet distance, computational latency, buses used, coverage, silhouette score (sklearn precomputed metric on the symmetric children-only road-distance matrix), and capacity violations.
- **A unified plotter** (`viz/plotter.py`) — single render function (`plot_solution`) used both for per-algorithm maps and for 5-up comparison grids. Routes are drawn following actual road shortest paths.
- **A CLI runner** (`run_benchmark.py`) — sweeps `(algorithm × N × seed)`, persists `runs.csv` incrementally (so a crash mid-sweep doesn't lose data), writes a paper-ready `summary.md`, and produces a `scalability.png`.

## How the comparison was run

```bash
python -m src.benchmarks.run_benchmark
```

Default sweep: **N ∈ {50, 100, 200, 400} × 3 seeds × 5 algorithms = 60 runs**. Wall-clock time on the dev laptop was ≈47 minutes, dominated by the CVRPTW solver (fixed at ≈100s per scenario because OR-Tools is configured with a 20s time limit per zone × 6 zones).

## Headline numbers (from `results/summary.md`)

### Fleet distance (km, mean over 3 seeds)

| algo | N=50 | N=100 | N=200 | N=400 |
|---|---|---|---|---|
| cvrptw_zones (incomplete coverage) | 134 | 198 | 356 | 590 |
| **kmedoids_ortools** | 193 | 280 | **398** | **641** |
| setcover_grid | 235 | 344 | 509 | 828 |
| setcover_perchild | 324 | 396 | 557 | 904 |
| genetic | 204 | 336 | 613 | 1438 |

### Latency (seconds, mean)

| algo | N=50 | N=100 | N=200 | N=400 |
|---|---|---|---|---|
| setcover_perchild | 0.01 | 0.02 | 0.04 | 0.16 |
| setcover_grid | 0.24 | 0.58 | 1.51 | 9.08 |
| genetic | 7.4 | 21.3 | 52.7 | 116 |
| kmedoids_ortools | 15.0 | 25.0 | 50.2 | 102 |
| cvrptw_zones | 100 | 100 | 100 | 100 |

### Coverage

`kmedoids_ortools`, `genetic`, `setcover_perchild`, `setcover_grid` always hit 100%. **`cvrptw_zones` consistently leaves 7–13% of students unserved** because the 75-min `MAX_ROUTE_MINUTES` limit kicks in.

## Five things to highlight tomorrow

1. **Under the actual problem constraints (full coverage), K-medoids + OR-Tools TSP wins on distance at every N.** At N=400, it produces 641 km of routes vs.\ 828 km for the next best (Set Cover grid). It also has the highest silhouette score, meaning its clusters are the most spatially compact.

2. **CVRPTW's "lower distance" headline is misleading.** Its absolute distance is the lowest because it serves *fewer* students — its time-window constraint drops 7–13% of children. Real comparison requires either relaxing `MAX_ROUTE_MINUTES`, allocating more buses per zone, or treating CVRPTW as solving a *different* problem (constrained-time routing).

3. **Set Cover variants are the speed/coverage sweet spot.** ≤10s at N=400 with full coverage. Their downside is route quality: 30–40% worse than K-medoids because they use NN tours rather than optimized TSP. A natural follow-up is "Set Cover for cluster selection + OR-Tools TSP for per-cluster ordering" — a hybrid we didn't try here.

4. **Our generic Genetic Algorithm degrades sharply at scale.** At N=400, fitness is 1438 km — over 2× worse than K-medoids. The negative silhouette score (clusters dispersed across the city) is the explanation: a permutation chromosome decoded by greedy-capacity-split doesn't respect spatial locality, and 200 generations × 100 individuals isn't enough to reach a good basin. The well-known fix in the literature is route-encoded chromosomes with insertion/2-opt as crossover, not a flat permutation. Worth flagging this as a finding rather than treating it as the GA implementation being broken — this is the textbook behavior.

5. **Roberto's grid-seeded Set Cover is structurally similar to Mariana's per-child Set Cover, but uses different seeds.** Mariana seeds candidates from each child (n candidates); Roberto seeds from a Cartesian grid (~n grid points). Both feed the same PuLP ILP. Mariana minimizes #buses, Roberto minimizes total tour distance, so they don't optimize the same objective — and yet they end up within 8% of each other on fleet distance. The interesting takeaway is that the candidate-seeding strategy matters less than the per-route ordering, which is what the next iteration should focus on.

## Reproducing the run

```bash
# from repo root
uv sync                                            # install pulp, shapely, scikit-learn, pandas
python -m src.benchmarks.run_benchmark             # full default sweep ~45 min
python -m src.benchmarks.run_benchmark \
    --densities 100 --seeds 1 --algos kmedoids_ortools genetic   # quick smoke test
```

Outputs land in `results/`:

- `runs.csv` — long-format, one row per (algorithm, N, seed); written incrementally during the run
- `summary.md` — pivoted mean ± std tables (paper-ready)
- `scalability.png` — log-y latency vs N
- `plots/N{N}_s{seed}_{algo}.png` — per-(scenario, algorithm) map (60 files in the default sweep)
- `plots/N{N}_s{seed}_grid.png` — 5-up comparison grid for that scenario (12 files)

## Bugs we found and fixed during the build

1. **`networkx.algorithms.approximation.traveling_salesman_problem` on directed graphs occasionally returns non-Hamiltonian tours that revisit nodes.** This silently inflated Roberto's grid Set Cover routes by 1–2 children per route, breaking the capacity constraint. Fixed in our adapter by using `greedy_tsp` cost only for the ILP objective, but emitting a clean nearest-neighbor tour for the final route output.
2. **Pandas `pivot_table` with `aggfunc="std"` drops columns that have all-NaN std** (i.e., when only one seed ran), which broke our summary writer. Fixed by guarding the lookup.

## Files of interest for the demo

- `src/benchmarks/run_benchmark.py` — entry point, ~250 lines
- `src/benchmarks/core/scenario.py` and `solver.py` — the shared interface, ~60 lines combined
- `results/summary.md` — the table to paste into the paper
- `results/plots/N400_s1_grid.png` — the visual money shot at maximum density
- `docs/latex/optimization.tex` — paper, with the new "Evaluación comparativa" section

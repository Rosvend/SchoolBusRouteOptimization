# Internal review response — CITIS XII 2026 submission

Running log of the coauthor + external review pass on `optimization.tex`.
Each item records the **verdict** (was the criticism actually true?), **what
changed**, and whether it **needs sign-off**.

Ground rules applied throughout: no fabricated numbers, nothing that could
de-anonymize us, and every claim traced to either the manuscript, `results/runs.csv`,
or the solver source.

## Constraints established before editing

| Constraint | Value | Source |
|---|---|---|
| Page limit | **12–15 pages** (a minimum *and* a maximum) | CITIS official author instructions |
| Required structure | Abstract, Introduction, Methods, Results, Discussion, Conclusions, References | idem |
| Review model | Double-blind | idem |
| Page count before edits | **14** | `optimization.log`: `Output written on optimization.pdf (14 pages, ...)` |
| Headroom | **~1 page** | derived |
| Hardcoded cross-references | **zero** — every reference is `\ref{}`-based | grep `(Section\|Table\|Figure\|Eq\|Alg)[~ ]*[0-9]` over `*.tex` → 0 hits |

Because there are no hardcoded cross-references, renumbering was mechanically
safe; we still chose **not** to renumber (see item 10).

## Verification of the paper's existing numbers

Before changing anything we re-derived every statistic the paper reports from
`results/runs.csv`. **All of them reproduce exactly:**

| Reported | Recomputed |
|---|---|
| Friedman distance χ²=46.7, p<10⁻⁸ | χ²=46.7143, p=6.4967e-09 |
| Friedman latency χ²=42.3, p<10⁻⁷ | χ²=42.3333, p=5.0428e-08 |
| Mean ranks 1.67/1.67/2.92/4.50/4.75/5.50 | identical |
| Wilcoxon K-medoids vs Sectorial p=0.68 | p=0.6772 |
| Latency ranks: per-child 1.00, grid 2.25 | identical |
| Table 2/3 cell values (e.g. 465±31, 464±32, 995±43) | identical |

However, **no committed code reproduced them** (no stats script, `scipy` absent
from `pyproject.toml`). Fixed — see item 9d.

---

## Items

### 1. NP-hardness framing — VERDICT: TRUE (gap). TEXT. Done.
The text already said "super-exponentially" but never gave the combinatorial
count. Added the factorial enumeration argument *alongside* the NP-hardness
claim, keeping both as the reviewer asked: a bus over $n_c$ stops admits $n_c!$
orders, and the partition must be chosen jointly with the orders.
Did **not** replace "NP-hard".

### 2. Name the heuristics — VERDICT: TRUE, and the attribution checks out. TEXT. Done.
Verified against `refs.bib` before naming:
- `HelsgaunAnHeuristic` = Helsgaun, *An Effective Implementation of the
  Lin–Kernighan Traveling Salesman Heuristic*, Roskilde Univ., 2000. ✔
- `Croes1958TwoOpt` = Croes, *A Method for Solving Traveling-Salesman Problems*,
  Oper. Res. 6(6):791–812, 1958 — the origin of 2-opt. ✔

Both now named inline. *Note for later (not changed):* the Helsgaun entry cites
the tech-report form; the canonical citable version is EJOR 126(1):106–130,
doi 10.1016/S0377-2217(99)00284-2. Worth upgrading at camera-ready.

### 3. Cut the "remainder of the paper" paragraph — VERDICT: PREMISE PARTLY FALSE. TEXT. Done differently.
We did **not** cut it. CITIS requires an explicit
Abstract/Introduction/Methods/Results/Discussion/Conclusions structure, and this
paragraph is the cheapest place to satisfy that. Rewritten so it states that
§3–§5 jointly constitute the Methods. This resolves item 10's structural
requirement at zero page cost. **Flagged: this reverses the reviewer's
recommendation — tell us if you disagree.**

### 4. Figure 2 invisible marker — VERDICT: TRUE, confirmed from the underlying values. FIGURE. Done.
Confirmed numerically rather than assumed, per the brief. At $N=400$:

| Strategy | Distance | Latency |
|---|---|---|
| K-medoids | 465.222 km | 101.144 s |
| Sectorial | 464.082 km | 101.205 s |

→ Δ = 1.14 km (**0.25 %**) in x and 0.06 s (**0.06 %**) in y. Genuinely
coincident; the reviewer's hypothesis was right.

**Problem found:** no script in the repo (or in any git object —
`git log --all -S pareto` is empty) generates `pareto_N400.pdf` or
`scalability.pdf`. Both were produced ad-hoc on 2026-06-20 and are
gitignored by the blanket `*.pdf` rule, so the paper **could not be rebuilt from
a clean clone**.

Fixed by adding `scripts/plot_paper_figures.py`, which regenerates both figures
from `results/runs.csv`. Styling preserved from the originals (tab10 C0–C5 in
strategy order, same labels and axes). Overlap handled by making it *legible and
explicit* rather than hiding it:
- Sectorial is a **hollow ring** drawn over the filled K-medoids marker, so both
  are visible even when exactly coincident.
- An annotation states the measured gap (1.1 km, 0.25 %; 0.06 s).
- Both captions now say so.

Two further overlaps were caught while checking the regenerated output — neither
was in the review notes:
- In the Pareto plot, an upper-right legend **hid the Genetic point entirely**
  (995 km, 116 s). Legend moved below the axes.
- In the scalability plot, Sectorial's line **completely masked K-medoids'** at
  every density. Sectorial is now dashed with hollow markers over a thicker solid
  K-medoids curve; the legend also covered SetCover-child at $N=400$ and was moved.

### 5. Road-network vs Euclidean distance — VERDICT: LARGELY ALREADY ADDRESSED. No duplication.
Checked before writing anything, as instructed:
- Introduction already states it: *"different distance models (Euclidean vs.\
  road network)"*.
- §3.1 already formalizes directed vs symmetrized distances, including
  $d_{ij}\neq d_{ji}$ and $D^{\mathrm{sym}}$.
- Related Work §2.2 mentions "on a real road network" in passing.
- The abstract already says "real OpenStreetMap road network" and "directed
  shortest-path distances".

Per the brief ("if it's already explicit, don't duplicate it"), **no change
made**. Scarce page budget spent elsewhere.

### 6. CITIS relevance — VERDICT: framing TRUE; emissions data ABSENT. Partly done, partly deferred.
**Done (framing):** new Introduction paragraph placing the contribution under the
official CITIS track **"Smart mobility and data-driven urban planning"** (within
*Energy, Advanced Manufacturing, and Sustainable Cities*).

**Done (honest sustainability hook, no invented numbers):** fleet distance *is*
vehicle-kilometres travelled, so we state that solver choice alone changes VKT by
**2.1×** at $N=400$ (995 vs 465 km) — arithmetic on numbers already in the
results tables, not a new estimate.

**DEFERRED — no data in repo.** Searched for any emissions/cost basis:
`grep -riE "emission|co2|vkt|fuel|carbon|cost_per_km|diesel|litre"` across all
non-binary files returns **only false positives** ("optimization", "literature").
There is no emission factor, fuel model, cost-per-km, or municipal operational
data anywhere. Converting VKT → CO₂ or comparing against real Valle de Aburrá
operations is therefore stated as **future work**, not estimated. Supported by
two real citations (Sabet & Farooq 2022; Garside et al. 2024).

### 7a. Missing formal models — VERDICT: TRUE. TEXT. Done.
- **Zonal CVRPTW:** added a decision-variable model (Eqs. for objective,
  visit/drop, capacity, time propagation, duration span) with $y^z_{ij}$, drop
  indicators $u_i$, arrival times $a_i$, span penalty $\gamma\sigma_z$ and drop
  penalty $\Pi$. Real parameter values used ($b=90$ s, $T_{\max}=75$ min).
- **Set-covering variants:** formalized the two candidate-pool constructions
  ($R_i=\{i\}\cup\mathrm{NN}_{C-1}(i)$ for per-child; grid-seeded greedy routes
  for grid) and the two objectives (fleet size vs total distance).

**Substantive finding while doing this (not in the review notes).** We checked
the implementation rather than describing the intent, and the set-cover variants
solve a *covering* ($\ge 1$) not a *partitioning* ($=1$) program **and execute
the selected routes as generated** (`setcover_perchild.py:81-89`,
`setcover_grid.py:137-148`). Students in two selected routes are therefore
**visited twice, and both visits are charged to fleet distance**. Quantified from
`runs.csv`: `mean_route_load` is exactly 20.0 in every set-cover run, and at
$N=400$ per-child selects 27 capacity-filled routes = **540 seats for 400
students** (~25 % redundant visits). This is the dominant explanation for those
methods' poor distance and fleet-size numbers and was previously undisclosed.
Now stated explicitly in §4.5. **Worth your attention — it's a real property of
our baselines, not a wording issue.**

### 7b. Eq. (10) placeholder — VERDICT: TRUE. TEXT. Done.
`\text{subtour-elimination constraints}` replaced with the explicit
Dantzig–Fulkerson–Johnson family,
$\sum_{i\in S}\sum_{j\in S} y_{ij}\le|S|-1$ for all proper $S$ with
$2\le|S|\le|V_c|-1$, labelled `eq:dfj`. Chose DFJ over MTZ because it states in
one line (page budget) and is the classical form.
Added an honest sentence that we do **not** enumerate this exponential family:
OR-Tools maintains a single circuit by construction and improves it with Guided
Local Search, so the formulation defines the optimum we approximate, not the
algorithm used.

### 7c. Coverage contradiction (§3.2 vs §4.4/Table 4) — VERDICT: TRUE. TEXT. Done.
Real contradiction: §3.2 asserted full coverage as a hard constraint for all
methods while §4.4 allows ~1 % unserved. Resolved **at the model level** as
requested, using the option the reviewer listed second *and* the first:
§3.2 now scopes hard coverage to the five full-coverage strategies **and** states
the CVRPTW's soft-coverage relaxation formally
($\sum_c x_{ic} + u_i = 1$, objective $+\,\Pi\sum_i u_i$), with the full model
given in §4.4.

### 8. Table 1 "dominant cost" contradicts Tables 2/3 — VERDICT: TRUE, and Table 1 was the wrong side. DATA-DEPENDENT → MEASURED.
The reviewer was right that the paper contradicted itself, and right about which
side was correct. We **measured** it rather than arguing it, by wrapping the real
call sites (`capacitated_assign`, `_assign_with_min_cost_flow`, `solve_tsp`) in
`time.perf_counter` timers via a throwaway script — **no `src/` changes** — and
re-running K-medoids and Sectorial at every density, 1 seed:

| N | min-cost flow | per-cluster TSP | FasterPAM |
|---|---|---|---|
| 50 | 0.006 s (0.04 %) | 15.03 s (99.83 %) | 0.019 s |
| 100 | 0.010 s (0.04 %) | 25.01 s (99.96 %) | <0.001 s |
| 200 | 0.123 s (0.25 %) | 50.01 s (99.73 %) | 0.001 s |
| 400 | 1.091 s (1.08 %) | 100.03 s (98.91 %) | 0.002 s |

(Sectorial is within 0.1 pp of these at every density.)

So the min-cost flow is **≤1.1 %** of runtime, and the TSP stage is
*budget-bound*: it costs almost exactly $k\cdot B$ with $B=5$ s, giving
15.03/25.01/50.01/100.03 s for $k=3,5,10,20$. Since $k=\lceil n/C\rceil$, total
latency grows **linearly** in $n$ — exactly what Table 3 shows.

Changes: Table 1 now has **two** columns, "Asymptotic worst case" (keeping
$O(T n^3)$, which is correct as a bound) and "Measured dominant stage". The
complexity-analysis prose reports the measured split, and notes the flow term's
share is still rising with $n$, so it would dominate only for instances an order
of magnitude larger. Figure 1's caption now agrees with Table 1.

### 9a. No external baseline — VERDICT: TRUE. DATA-DEPENDENT → LIMITATION.
Searched: `grep -riE "\.vrp|solomon|augerat|cvrplib|uchoa|christofides|alns|tabu"`
→ only the surname "Solomon" in `refs.bib` (a co-author of the Ellegood survey)
and LaTeX `tabular` matches. There are **no external instance files, no CVRPLIB
loader, and no ALNS or tabu-search implementation** in the repo. All instances
are synthetic, generated on the real OSM graph.

Not fabricated. Written up as an explicit, prominent limitation: the study
establishes *relative* standings under identical conditions but does not
calibrate absolute quality against published instances or a state-of-the-art
metaheuristic. Promoted to the **first** item of future work.

### 9b. Dated reference list — VERDICT: TRUE. TEXT. Done.
Confirmed: of 21 references only 2 were ≥2021, and one of those is a software
citation — effectively **1 recent research reference out of 21**.

Added 6 entries, every field verified against Crossref or the publisher (no
invented metadata):

| Citation | Venue | DOI |
|---|---|---|
| Wang, Yang, Liao, Chen (2025), *The SBRP: A comprehensive review* | European Transport Studies 2:100043 | 10.1016/j.ets.2025.100043 |
| Wu et al. (2024), *Neural Combinatorial Optimization … VRPs* | arXiv:2406.00415 | 10.48550/arXiv.2406.00415 |
| Ben Sghaier (2026), *The Urban Bus Routing Problem …* | Int. J. ITS Res. 24(2):993–1017 | 10.1007/s13177-026-00633-w |
| Sabet & Farooq (2022), *Green VRP: State of the Art* | IEEE Access 10:101622–101642 | 10.1109/ACCESS.2022.3208899 |
| Garside, Ahmad, Muhtazaruddin (2024), *… green VRP and its variants* | Oper. Res. Perspectives 12:100303 | 10.1016/j.orp.2024.100303 |
| Demšar (2006), *Statistical Comparisons of Classifiers …* | JMLR 7:1–30 | — (JMLR, verified on jmlr.org) |

Placed in §2.1 (recent SBRP synthesis), §2.3 (neural CO + hybrid + green VRP),
the new sustainability paragraph, and §6 (Demšar, for Nemenyi). Refs 21 → 27.

### 9c. GA hyperparameters never stated — VERDICT: TRUE. DATA-DEPENDENT → FOUND IN CODE.
Read out of `src/benchmarks/adapters/genetic.py:100-107`. These are the values
**actually used**: `run_benchmark.py:209` instantiates every solver as
`SOLVER_REGISTRY[algo_name]()` with no kwargs, and no other instantiation of
`GeneticSolver` exists in the repo — so the signature defaults *are* the
experimental settings.

| Symbol | Value |
|---|---|
| $P$ population | 100 |
| $G$ generations | 200 |
| $p_m$ mutation rate | 0.1 |
| $t$ tournament size | 3 |
| $e$ elites | 2 |
| $S$ stagnation window | 30 |

Two details worth stating that the paper's pseudocode implied wrongly: **order
crossover is applied unconditionally, so there is no crossover rate**
(`genetic.py:147-148`), and the decoder's **2-opt is capped at a single pass**
(`genetic.py:66`). Also added the other solvers' real budgets ($T=50$ for
k-medoids, $T=5$ and $\alpha=0.6,\beta=0.4$ for sectorial, 75 min / 90 s / 300
for CVRPTW) since they were symbolic-only too.

Not stated in the paper and left alone: `lambda_acc = 0.5`, hard-coded at
`sectorial.py:167` and not exposed as a parameter. **Flagging it — you may want
to document or expose it.**

### 9d. Statistical power / Nemenyi — VERDICT: TRUE. Done without new data.
Per your decision, seeds stay at 3 (a 5-seed re-run would have changed every
table, figure and statistic on deadline day). The Nemenyi post-hoc turned out to
be computable from the existing 12 blocks, so the "promised post-hoc" is now
actually delivered:

$\mathrm{CD} = q_\alpha\sqrt{k(k+1)/6N}$ with $k=6$, $N=12$, $q_{0.05}=2.850$
→ **CD = 2.177**. **7 of 15** pairs significant on fleet distance:

- Significant: both leaders vs SC-grid (2.833), vs Genetic (3.083), vs
  SC-per-child (3.833); CVRPTW vs SC-per-child (2.583).
- Not separable: K-medoids vs Sectorial (0.000), both leaders vs CVRPTW (1.250),
  and every pair within {SC-grid, Genetic, SC-per-child}.

This *strengthens* the headline claim while bounding it honestly. The Limitations
paragraph was rewritten accordingly: instead of a vague "3 seeds is thin", it now
names exactly which comparisons 12 blocks cannot resolve and states these are
power limits, not evidence of equivalence.

Also added `scripts/compute_statistics.py` so every statistic in the paper is
reproducible from `results/runs.csv` (it was not, before), plus `scipy` to
`pyproject.toml`.

### 9e. Over-broad GA conclusion — VERDICT: TRUE. TEXT. Done.
Scoped in all three places it appeared:
- **Abstract:** "a flat genetic encoding degrades sharply" → "the one genetic
  encoding we test — a flat permutation with a positional capacity decoder".
- **Discussion:** added an explicit scope paragraph naming what is *not* tested
  (route/cluster-based chromosomes, repair operators, ALNS, learned construction
  policies) and concluding that a naive encoding is a poor default for SBRP, not
  that metaheuristics are.
- **Conclusions:** same scoping.

### 10. CITIS structure + availability — VERDICT: structure TRUE; availability needs your action.
**Structure (your decision: light-touch).** Section numbers unchanged; the
roadmap paragraph now names §3–§5 as Methods (see item 3). No renumbering, so no
cross-reference risk and no reflow.

**Availability (your decision: anonymized mirror link).** Added a "Code and data
availability" paragraph naming only the toolchain (OSMnx, NetworkX, OR-Tools,
FasterPAM, CBC/PuLP, scikit-learn), noting the scenario generator is
deterministic given $(N,\text{seed})$. The URL is a **deliberately visible
placeholder**:

> `\textbf{[ANONYMIZED MIRROR URL --- INSERT BEFORE SUBMISSION]}`

It renders in the PDF so it cannot be silently forgotten. **We did not invent a
URL.**

---

## Second pass (after "code is not required")

**Availability statement removed**, per your professor. Consequences handled:
- The paragraph cited OSMnx, OR-Tools, FasterPAM and scikit-learn, and I had
  earlier redirected §4.1's toolchain list to point at it. Removing it would have
  orphaned four citations and left a dangling cross-reference, so the toolchain
  list was **restored to §4.1** where it originally lived.
- The `[ANONYMIZED MIRROR URL]` placeholder is gone, so the mirror is no longer
  needed and **nothing is blocking on you for the paper itself**. The code-side
  anonymity scrub (`zone_definitions.py:1`, git metadata) only matters if you
  ever publish the repo.
- Still 15 pages: the removal freed ~9 lines and restoring the toolchain list
  cost ~7, so it was close to neutral. Page 15 is completely full (all 25
  references render, ending at the foot of the page), so there is **no slack** to
  restore Algorithm 2 or the DOIs.

**Full numeric audit against `results/runs.csv`.** I recomputed every
data-derived number in the manuscript. All tables and statistics matched exactly.
**Two prose claims did not** and are now corrected:

| Claim | Was | Actual | Fixed to |
|---|---|---|---|
| CVRPTW's longest route vs the others at $N{=}400$ | "35--60\% shorter" | 36.9\%--62.6\% | "37--63\% shorter" |
| Set Cover (per-child) latency | "solves every instance in under $0.2$~s" | slowest single run is **0.223 s** | "under a quarter of a second" |

**Pseudocode overclaim fixed.** Three places stated or implied that *every*
strategy comes with pseudocode. Only two do (k-medoids, GA) — and only three did
even before Algorithm 2 was cut. The contributions bullet, the roadmap paragraph
and the Conclusions now claim a formal model and complexity analysis per
strategy, without implying pseudocode for all six. This is the kind of thing a
reviewer checks by counting the algorithm floats.

**Silhouette-vs-distance claim corrected.** The Discussion asserted that
"silhouette tracks fleet distance closely across the sweep". Tested: within a
density it is $r=-0.94$ at $N{=}400$ and $r=-0.82$ at $N{=}200$, but
$r=-0.31$ at $N{=}100$ and $r=-0.02$ ($p=0.95$) at $N{=}50$. The pooled
correlation over all 72 runs ($r=-0.47$) is an artefact of distance growing with
$N$. So the claim was false at small $N$; the text now states the measured
scale-dependence instead. Also removed the "even edges ahead at the largest size"
reading of Sectorial's $1.1$~km lead, which is well inside seed noise.

**Naming made consistent**: the merged table said "Set Cover (child)" while
Table 1, Table 3 and the prose said "Set Cover (per-child)". Now uniform.

### Extended seed sweep (in progress at time of writing)
Because the statistical design was the paper's main acknowledged weakness, the
benchmark is being extended from 3 seeds to 11 (seeds 4--11 added). Method notes
for the record:
- One `run_benchmark` invocation **per seed**, each into its own
  `results/extra_seeds/seed<N>/`. The harness writes only its own rows to
  `<out>/runs.csv`, so a single invocation pointed at `results/` would have
  overwritten the original 3-seed data; and its loop is `for N: for seed:`, so a
  truncated multi-seed run would leave the design unbalanced. Per-seed runs mean
  each completion is a balanced increment over all four densities, and
  `scripts/merge_runs.py` refuses to merge an unbalanced set.
- The added seeds ran with `--no-plots`. This does not affect any measurement:
  `_run_one` times only `solver.solve()`, and plotting happens after the timer
  stops. Noted for transparency since the original three seeds ran with plotting
  enabled between scenarios.
- Scenarios are deterministic in `(N, seed)` and the OSM graph is cached, so the
  added seeds are drawn from the same generator as the originals.
- Nothing else was run on the machine during the sweep: OR-Tools budgets are
  wall-clock, so CPU contention would have degraded solution quality.

**Completed: 11 seeds, 44 blocks, 264 runs, 0 failures.** All gate checks passed,
so the extended data was integrated. `results/runs.csv` is now the full 264-run
set; the original three-seed file is preserved as
`results/runs_seeds1-3_original.csv`, and the per-seed directories remain under
`results/extra_seeds/`.

| Gate | 3 seeds | 11 seeds | Held? |
|---|---|---|---|
| K-medoids vs Sectorial tied | $p=0.68$, $\Delta=0.00$ | $p=0.36$, $\Delta=0.045$ | yes |
| Leaders best two on distance | ranks 1.67/1.67 | ranks 1.67/1.63 | yes |
| Genetic worst, degrades with $N$ | 995 km at $N{=}400$ | 1006 km | yes |
| Set cover fastest by orders of magnitude | yes | yes | yes |
| Coverage 0.998 CVRPTW / 1.000 others | yes | yes | yes |

**What the extra seeds bought.** $\mathrm{CD}$ fell from $2.18$ to $1.14$ and
significant pairs rose from **7 of 15 to 11 of 15**. Crucially the leaders now
separate from the zonal CVRPTW ($\Delta=1.26$--$1.31$), and CVRPTW from all three
trailing methods — exactly the comparisons the previous Limitations paragraph had
to concede were unresolvable. Only four pairs remain open: the genuine
K-medoids/Sectorial tie, and the three trailing methods among themselves.

**Two supporting claims changed and were corrected, not papered over:**
1. The Discussion said the two leaders "attain the two highest silhouette
   scores". At 11 seeds the ranking is K-medoids $0.276$, **CVRPTW $0.272$**,
   Sectorial $0.266$ — so cohesion no longer singles out the two leaders. The
   text now says so explicitly.
2. The $N{=}100$ standard deviations grew substantially (e.g. K-medoids
   $157\pm6 \rightarrow 163\pm18$). The three-seed run had understated variance;
   the wider intervals are the more honest figure.

Also: the bolded distance minimum at $N{=}400$ moved from Sectorial to K-medoids
(460 vs 461 km) — a 1.5 km difference that is well inside seed noise, and the
text says as much rather than claiming a winner.

Every table body was regenerated mechanically by `scripts/emit_tables.py` from
the CSV rather than hand-transcribed, and both figures by
`scripts/plot_paper_figures.py`. A dangling "unlike at three seeds" phrase
introduced during the rewrite was removed, since the paper now reports only the
eleven-seed design.

### Figure 3 rebuilt (was illegible)
Figure 3 was six map panels at `0.55\textwidth` — each under an inch wide, so its
caption's promise of visible "wedge structure" versus "overlapping routes" could
not actually be seen. Replaced with **three panels in one row at
`0.95\textwidth`** (`scripts/plot_qualitative_figure.py`, new), one per solution
paradigm: Sectorial (clustering), Set Cover grid (covering ILP), Genetic (flat
metaheuristic). Because the new image is aspect 2.41 against the old 1.21, each
panel is **1.7× larger while the figure is shorter**, so this cost no pages. The
caption now describes what is actually visible in each panel, including the
set-cover double-visit behaviour. Generated after the sweep finished, so no CPU
contention with the wall-clock-budgeted solvers.

---

## Page budget: what it cost (READ THIS)

**Final: 15 pages — at the maximum, inside the 12–15 limit.** Verified by
`optimization.log`: `Output written on optimization.pdf (15 pages, ...)`.

My planning estimate of "+1 page" was wrong. The additions above came to **+5
pages** on first compile (19 pages, Conclusions pushed from p.12 to p.17). The
original manuscript already used 14 of the 15 available pages, so fitting the
review response required real cuts, not just tightening. In order of application:

**Lossless (no content removed):**
1. Every added passage rewritten ~35 % shorter — substance and all numbers kept.
2. **Tables 2 and 3 merged** into one table with two column groups (fleet
   distance | latency). Both `\label`s retained, so every `\ref{}` still
   resolves; the paper now cites it as Table 2.
3. **Figures 1 and 2 paired side by side** in a single float via `minipage`.
   In-plot typography scaled up so they stay legible at ~0.49\textwidth.
4. Algorithm bodies set `\footnotesize`; tables set `\small`.
5. Figure 3 reduced from `\textwidth` to `0.55\textwidth`.
6. Table 1 restructured from 4 columns back to 3 — the "Clustering / selection"
   column was folded into the asymptotic column. This also fixed a **75 pt
   margin overflow** my 4-column version had introduced.
7. Long author lists in two references (Wu et al., 9 authors; Pedregosa et al.,
   16) collapsed to "et al." via bibtex `and others`.

**Actual losses — reversible, and I want your call on them:**
8. **Algorithm 2 (Sectorial pseudocode) was removed.** This was the least
   load-bearing element: Sectorial remains fully specified by
   Eq. (\ref{eq:sectorial-score}), the §4.3 prose, and its explicit reuse of the
   baseline's minimum-cost-flow reassignment (Algorithm 1). The GA pseudocode is
   now Algorithm 2. **If you would rather keep it, the cheapest thing to drop
   instead is Figure 3** (the six-panel map grid), which carries no quantitative
   claim.
9. **DOIs removed from the 20 pre-existing references** (`splncs04` prints each
   DOI on its own line, so this recovered ~16 lines). DOIs were **kept for the 5
   new journal/arXiv references** so reviewers can verify the recency fix.
   Trivially restorable at camera-ready if an extra page is granted.
10. Two of the six new citations were dropped to save space:
    **Ben Sghaier (2026)** and **Garside et al. (2024)**. Their bib entries are
    still in `refs.bib`, just uncited — re-citing is one line each. The recency
    fix still lands: Wang 2025, Wu 2024, Sabet 2022 and Demšar 2006 are cited,
    taking the reference list from 1 recent research item to 4.

Nothing that a reviewer asked for was dropped: items 1, 2, 4, 6, 7a, 7b, 7c, 8,
9a–9e and 10 are all in the submitted text.

## Third pass (post-review, pre-submission)

### 1. Stale figure/prose mismatch — fixed
Figure 2's embedded annotation is generated from the data and read
"$\Delta$ = 1.0 km, 0.22%", but the Fig. 2 caption and the Discussion both said
**1.5 km (0.3%)**. Recomputed from `results/runs.csv`: K-medoids 459.856 km vs
Sectorial 460.885 km at $N{=}400$ → **1.029 km = 0.224%**. The figure was right;
the two prose figures were wrong and are now 1.0 km / 0.22%.

Cause, for the record: this was **not** left over from the 3-seed run — the old
gap was 1.14 km. I introduced 1.5 km by hand while updating the caption during
the eleven-seed integration. The table bodies were generated mechanically and
were correct; only the two hand-edited prose spots were wrong. Lesson applied:
the two numbers that appear both in a generated figure and in hand-written prose
are exactly where drift happens.

### 2. Literal "Methods" section — added
Previously §3 Problem Formulation, §4 Solution Strategies, §5 Experimental Setup
were mapped to "Methods" only in the roadmap prose. Now consolidated under a real
`\section{Methods}`:

| | Before | After |
|---|---|---|
| Methods | (three separate sections) | **3 Methods** |
| | 3 Problem Formulation | 3.1 Problem formulation |
| | 4.1 Common benchmark framework | 3.2 Common benchmark framework |
| | 4.2–4.6 the five other strategies | 3.3–3.7 |
| | 4.7 Complexity analysis | 3.8 Complexity analysis |
| | 5 Experimental Setup | 3.9 Experimental setup |
| Results / Discussion / Conclusions | 6 / 7 / 8 | 4 / 5 / 6 |

Rendered top-level sequence is now exactly
**Abstract → 1 Introduction → 2 Related Work → 3 Methods → 4 Results →
5 Discussion → 6 Conclusions → References**.

Each of the six strategies stays a *numbered* subsection, so nothing lost
prominence; only the five problem-formulation sub-parts were demoted to
third-level run-in headings, which is precisely the style the LLNCS template
prescribes ("Only two levels of headings should be numbered. Lower level
headings remain unnumbered; they are formatted as run-in headings"). All
cross-references were re-pointed and re-resolve: `sec:problem`→3.1,
`sec:setup`→3.9, and the CVRPTW soft-coverage reference now points at the
CVRPTW subsection (3.5) rather than at the whole strategies section. Still 15
pages — converting subsections to run-in headings recovered roughly what the new
section header cost.

### 3. Reference traceability audit
Checked all 25 cited references. **None is fabricated.** Evidence per source:

| Source of verification | Count | References |
|---|---|---|
| Present in your Mendeley library | 11 | Park 2010, Ellegood 2020, Corberán 2002, Cakir, Comert 2018, Helsgaun, Dahiya 2018, Joshi 2019, Baranwal 2016, Hatamlou 2018, Akitaya 2021 |
| Verified by DOI against Crossref | 7 | Dantzig–Ramser 1959 (10.1287/mnsc.6.1.80), Clarke–Wright 1964 (10.1287/opre.12.4.568), Croes 1958 (10.1287/opre.6.6.791), Voudouris 1999 (10.1016/S0377-2217(98)00099-X), Rousseeuw 1987 (10.1016/0377-0427(87)90125-7), Boeing 2017 (10.1016/j.compenvurbsys.2017.05.004), Wang 2025 (10.1016/j.ets.2025.100043), Sabet 2022 (10.1109/ACCESS.2022.3208899) |
| Verified on publisher site (JMLR has no DOIs) | 2 | Demšar 2006 (JMLR 7:1–30), Pedregosa 2011 (JMLR 12:2825–2830) |
| Books / preprint / software, verified by title+publisher | 5 | Holland 1992 (MIT Press), Kaufman & Rousseeuw 1990 (Wiley), Wu et al. 2024 (arXiv:2406.00415), OR-Tools (Google, URL) |

**Important disclosure:** the four *recent* references I added during the review
response — **Wang 2025, Wu 2024, Sabet 2022, Demšar 2006** — are **not** from the
Mendeley group. I found them via literature search and verified each against
Crossref or the publisher before citing. If you want the library to be the single
source of truth, add those four to 'Optimizacion UPB'; I can do that through the
Mendeley MCP on request.

Two entries that were in `refs.bib` but **never cited** (Ben Sghaier 2026,
Garside et al. 2024 — both real, both dropped for space) have been removed, so
`refs.bib` now contains exactly the 25 cited works and nothing else.

*Caveat on the Mendeley check:* `mendeley_list_folders` returns an empty list, so
I could not filter to the 'Optimizacion UPB' group specifically — I searched the
whole authenticated library, which also contains unrelated material (aviation
fuel, disability, cardiovascular papers). The 11 hits above are genuine matches
on title and authors; absence of the other 14 means they are not in your personal
library, not that they are unverifiable.

## Fourth pass — EasyChair submission-guideline compliance

Audited against the ten EasyChair submission instructions. **Nine passed as-is;
one failed and was fixed.**

| # | Guideline | Result |
|---|---|---|
| 1 | LNNS/LLNCS template only | PASS — `\documentclass[runningheads]{llncs}` + `splncs04` |
| 2 | Strict 12–15 pages incl. figures/tables/refs | PASS — **15** |
| 3 | Max 6 authors | PASS — 2, entered only in EasyChair metadata |
| 4 | Anonymity + metadata scrubbed | PASS — 0 author strings in rendered text; PDF Info carries only `Creator=TeX`, `Producer=pdfTeX-1.40.25`, no Author/Title/Subject/Keywords fields, embedded-figure paths are relative (`./figures/...`), no username or absolute path anywhere |
| 5 | English throughout | PASS — only Spanish tokens are the place names Medellín / Valle de Aburrá |
| 6 | Originality + theme alignment | PASS — mapped to "Smart mobility and data-driven urban planning" in the Introduction |
| 7 | Self-citation < 20 % | PASS — **0 %**; no reference is authored by either author |
| 8 | Print-quality graphics, placed in text | PASS — Figs 1–2 are vector PDF; Fig 3 is 2136×888 px at `0.95\textwidth` ≈ **476 DPI**; all ten floats resolve within ~120 lines of their first `\ref` and interleave with the text (pp. 7–12) |
| 9 | **AI usage statement** | **WAS MISSING → added** |
| 10 | PDF via EasyChair | PASS |

### The one failure, and how the space was found
Guideline 9 requires an AI usage statement; the paper had none. (A grep for
"artificial intelligence" matched only Holland 1992's book *title* — a false
positive worth noting, since it would make a careless check look like a pass.)

Adding it to a paper already at 15/15 would have hit 16 pages and the
over-length desk rejection. The space came from finally executing the original
review **item 3**: cutting the "the remainder of the paper is organized as
follows" paragraph. That item was declined earlier because the paragraph was
carrying the Methods mapping — but now that §3 is a literal `\section{Methods}`,
the paragraph only restated the table of contents, so its premise had become
true. Net: still 15 pages.

The statement is placed at the end, before the references, as an unnumbered
run-in heading — the same convention the LLNCS template prescribes for
acknowledgements.

> **Use of generative AI.** Generative AI assisted with drafting and copy-editing
> the text, with the scripts that compute the reported statistics and render the
> figures from the benchmark output, and with literature search. It did not
> generate experimental results: every reported number is the output of the
> described code run on the described instances, and the authors verified all
> numbers, figures and references against the underlying data and primary
> sources. The authors assume full responsibility for the content of this paper.

**This one needs your sign-off.** It is a declaration *you* make and are
responsible for. I wrote it to cover what I can attest to — this review and
integration session. If AI was also used earlier (writing the solvers, earlier
drafts, translation), the statement must be widened to cover that. Please read it
against your actual workflow before submitting.

## Verification performed

Re-run after the eleven-seed integration, from a **clean build** (`latexmk -C`
then `latexmk -pdf`):

| Check | Result |
|---|---|
| Page count | **15** (limit 12–15) |
| LaTeX errors, undefined references, undefined citations | **none** |
| BibTeX warnings | **0** (`optimization.blg`: `warning$ -- 0`), 25 entries |
| Hardcoded (non-`\ref`) cross-references | **0** — re-grepped after all edits |
| Worst margin overflow | 6.3 pt (was 75.6 pt before the Table 1 fix) |
| Anonymity scan of rendered text (`Hincapie`, `Sandoval`, `Pontificia`, `naomi`, `github`, `4open`) | **no hits** |
| `scripts/compute_statistics.py` reproduces every reported statistic | **yes** |
| `scripts/plot_paper_figures.py` regenerates both figures from `runs.csv` | **yes** |
| All six strategies visible/distinguishable in both figures | **yes**, checked by rendering the compiled page |

## Needs your sign-off / action

1. **BLOCKING — anonymized mirror URL.** Create the mirror and replace the
   placeholder. Before mirroring, scrub these, which would otherwise
   de-anonymize you through the *code*:
   - `optimization.tex:5-7` — camera-ready author block (in a comment; harmless
     in the PDF, fatal in a shared repo).
   - `src/benchmarks/data/zone_definitions.py:1` — docstring reads *"ported from
     naomi:CVRPTW_optimización/config.py"*.
   - git author metadata on every commit.
2. **Algorithm 2 (Sectorial pseudocode) removal** — forced by the page limit.
   Reversible; see item 8 under "Page budget" for what to drop instead.
3. **Item 3 reversal** — we kept the roadmap paragraph instead of cutting it.
   Overrule us if you prefer it gone.
4. **Item 7a finding** — the set-cover double-visit behaviour is now disclosed in
   the paper. Confirm you're comfortable stating it (we think it is far better
   disclosed by us than discovered by a reviewer).
5. **Anonymity check before upload** — `pdfinfo` reports
   `Custom Metadata: yes` on the compiled PDF; do not ship `.synctex.gz` or the
   `.tex` alongside it.
6. **`lambda_acc = 0.5`** is hard-coded at `sectorial.py:167` and is not exposed
   as a parameter or documented in the paper. Decide whether to expose it.

## Not part of the submission (repo hygiene, found in passing)

- `README.md:30-36` headline table is stale (says k-medoids N=100 = 163,024 m,
  genetic = 202,934 m; `results/summary.md` says 156,662 ± 5,630 and
  205,679 ± 2,985). README also says "five routing algorithms" in three places
  but the registry has six.
- `docs/BENCHMARK_INSIGHTS.md` is referenced twice by README but was deleted;
  recoverable via `git show 4ce35f8:docs/BENCHMARK_INSIGHTS.md`.
- `docs/latex/figures/best_algo_N400.png` (2.0 MB) is tracked but never included.
- `.gitignore` had the LaTeX artifact block duplicated verbatim (lines 218-231
  and 236-249); left as-is, only appended the figures exception.
- No `llncs.cls` / `splncs04.bst` vendored and no Makefile, so the build depends
  on a `texlive-full` host.

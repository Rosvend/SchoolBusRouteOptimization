"""Adaptive large-neighbourhood search for the capacitated school bus routing problem.

Domain layer for the ALNS strategy: the solution state, the destroy/repair
operator pool, route construction helpers and the search driver. The adaptive
loop itself (roulette-wheel operator selection, simulated-annealing acceptance,
stopping rule) comes from the ``alns`` package; everything problem-specific
lives here.

This module deliberately imports nothing from the benchmark harness, so the
same search runs unchanged on external CVRP instances where customer demands
are not unit.

Index space
-----------
Customers and the depot are referred to by their row in the distance matrix
``ctx.dist``. A route is a list of customer indices *without* the depot; the
depot is re-attached only when a route is exported. Distances are directed:
``dist[i, j] != dist[j, i]`` in general, which is why 2-opt below recomputes a
candidate tour rather than differencing two arcs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime


@dataclass(frozen=True)
class AlnsParams:
    """Search settings. Fixed across a whole sweep: no per-instance tuning."""

    # Degree of destruction: q ~ U[min_frac * n, max_frac * n], capped.
    destroy_min_frac: float = 0.10
    destroy_max_frac: float = 0.25
    destroy_cap: int = 60
    # Ropke-Pisinger selection bias; 1 = uniform, larger = greedier.
    worst_p: float = 3.0
    shaw_p: float = 3.0
    # Weight of "already on the same route" in the Shaw relatedness measure,
    # as a fraction of the mean customer-to-customer distance.
    shaw_route_bonus: float = 0.25
    regret_k: int = 2
    # Roulette-wheel scores for (new best, better, accepted, rejected) and the
    # reaction factor theta.
    op_scores: tuple[float, float, float, float] = (5.0, 2.0, 1.0, 0.5)
    op_decay: float = 0.8
    # Simulated annealing, fitted so that a solution `sa_worse` worse than the
    # initial one is accepted with probability `sa_accept_prob` at the start.
    sa_worse: float = 0.05
    sa_accept_prob: float = 0.50
    sa_horizon_iters: int = 25_000


class AlnsContext:
    """Immutable problem data shared by every state (referenced, never copied)."""

    def __init__(self, dist, depot, customers, capacity, demands=None,
                 params=None):
        self.dist = np.asarray(dist, dtype=float)
        self.depot = int(depot)
        self.customers = np.asarray(customers, dtype=np.int64)
        self.capacity = float(capacity)
        self.params = params or AlnsParams()

        m = self.dist.shape[0]
        if demands is None:
            demands = np.ones(m, dtype=float)
            demands[self.depot] = 0.0
        self.demands = np.asarray(demands, dtype=float)

        # Symmetrised distances drive the Shaw relatedness measure only; route
        # cost always uses the directed matrix.
        self.sym = (self.dist + self.dist.T) / 2.0

        # Nested-list mirror of the distance matrix. Scalar numpy indexing costs
        # ~100 ns against ~30 ns for a list of lists, and objective evaluation is
        # scalar-heavy and on the hot path of a wall-clock-budgeted search.
        self.dlist = self.dist.tolist()

        self.timers: dict[str, float] = {"destroy": 0.0, "repair": 0.0}
        self.op_calls: dict[str, int] = {}
        # Repairs abandoned because no feasible packing was reachable. Always
        # 0 for unit demands; a high count on a CVRP instance means the
        # search is wasting iterations and the operators need attention.
        self.repair_dead_ends = 0

    def total_demand(self) -> float:
        return float(self.demands[self.customers].sum())

    def min_routes(self) -> int:
        return max(1, int(np.ceil(self.total_demand() / self.capacity)))


def route_cost(route, ctx) -> float:
    """Directed cost of ``[depot] + route + [depot]``. Empty routes cost 0."""
    if not route:
        return 0.0
    d, depot = ctx.dlist, ctx.depot
    total = d[depot][route[0]] + d[route[-1]][depot]
    for i in range(len(route) - 1):
        total += d[route[i]][route[i + 1]]
    return total


class SbrpState:
    """An ALNS state: a set of routes plus the customers awaiting reinsertion."""

    __slots__ = ("routes", "unassigned", "ctx", "_obj", "snapshot")

    def __init__(self, routes, unassigned, ctx, obj=None):
        self.routes = routes
        self.unassigned = unassigned
        self.ctx = ctx
        self._obj = obj
        # Routes as they stood before the last destroy, used to recover if a
        # repair dead-ends (possible only with non-unit demands, where
        # aggregate capacity does not imply a feasible packing).
        self.snapshot = None

    def copy(self) -> "SbrpState":
        out = SbrpState([r[:] for r in self.routes], list(self.unassigned),
                        self.ctx, self._obj)
        out.snapshot = self.snapshot
        return out

    def objective(self) -> float:
        if self._obj is None:
            self._obj = sum(route_cost(r, self.ctx) for r in self.routes)
        return self._obj

    def loads(self) -> np.ndarray:
        dem = self.ctx.demands
        return np.array([float(dem[r].sum()) if r else 0.0 for r in self.routes])

    def export_routes(self) -> list[list[int]]:
        """Routes in the harness's ``[depot, c1, ..., depot]`` form, empties dropped."""
        depot = self.ctx.depot
        return [[depot] + [int(c) for c in r] + [depot] for r in self.routes if r]


# --------------------------------------------------------------------------
# Route construction (initial solution)
# --------------------------------------------------------------------------

def nearest_neighbour_route(members, ctx) -> list[int]:
    """Greedy nearest-neighbour tour out of the depot over ``members``."""
    remaining = list(int(c) for c in members)
    if not remaining:
        return []
    d = ctx.dlist
    route = []
    cur = ctx.depot
    while remaining:
        nxt = min(remaining, key=lambda c: d[cur][c])
        remaining.remove(nxt)
        route.append(nxt)
        cur = nxt
    return route


def two_opt(route, ctx, max_passes: int = 2) -> list[int]:
    """2-opt over a depot-anchored route.

    Distances are asymmetric, so reversing a segment flips the direction of every
    arc inside it. The move's gain is therefore not a two-arc difference and each
    candidate is scored by recomputing the tour. Routes here hold at most
    ``capacity`` stops, so the extra cost is immaterial.
    """
    if len(route) < 3:
        return route
    best = route[:]
    best_cost = route_cost(best, ctx)
    for _ in range(max_passes):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = route_cost(cand, ctx)
                if cost < best_cost - 1e-9:
                    best, best_cost, improved = cand, cost, True
        if not improved:
            break
    return best


def _pack(ctx, n_routes: int, strategy: str):
    """Place every customer into ``n_routes`` capacity-feasible routes, or None.

    Customers are seeded in decreasing demand order: this is first-fit-decreasing,
    and packing the large items first is what keeps a near-full instance feasible.
    With unit demands the demand key is constant, so the order degenerates to
    farthest-from-depot first and the behaviour is unchanged.
    """
    d, depot, dem = ctx.dlist, ctx.depot, ctx.demands
    order = sorted((int(c) for c in ctx.customers),
                   key=lambda c: (-dem[c], -d[depot][c]))
    routes: list[list[int]] = [[] for _ in range(n_routes)]
    loads = [0.0] * n_routes
    for c in order:
        best_ri, best_key = None, float("inf")
        for ri in range(n_routes):
            slack = ctx.capacity - loads[ri] - dem[c]
            if slack < -1e-9:
                continue
            # "cost" keeps routes short; "tight" is classic best-fit and packs
            # harder, which is what rescues a near-full instance.
            key = d[routes[ri][-1] if routes[ri] else depot][c] if strategy == "cost" else slack
            if key < best_key:
                best_ri, best_key = ri, key
        if best_ri is None:
            return None
        routes[best_ri].append(c)
        loads[best_ri] += float(dem[c])
    return routes


def greedy_initial_routes(ctx, n_routes: int) -> list[list[int]]:
    """Demand-aware construction, used when demands are not unit.

    Tries a cost-driven placement first and falls back to pure best-fit, which
    packs more aggressively at the expense of longer routes. The search that
    follows recovers the length; infeasibility it cannot recover from.
    """
    for strategy in ("cost", "tight"):
        routes = _pack(ctx, n_routes, strategy)
        if routes is not None:
            return routes
    raise ValueError(
        f"no capacity-feasible packing into {n_routes} routes x capacity "
        f"{ctx.capacity} for total demand {ctx.total_demand()}"
    )


def polish_routes(routes, ctx, max_passes: int = 2) -> list[list[int]]:
    return [two_opt(r, ctx, max_passes) for r in routes]


# --------------------------------------------------------------------------
# Insertion machinery (hot path)
# --------------------------------------------------------------------------

# Sentinel for a capacity-infeasible insertion. Finite so that regret
# differences stay well-defined, huge so it never wins an argmin.
_BLOCKED = 1e18
_BLOCKED_HALF = _BLOCKED / 2.0


def _positions(route, depot):
    """Predecessor/successor arrays for every insertion slot of a route.

    For ``route = [a, b]`` the slots are (depot,a), (a,b), (b,depot); an empty
    route degenerates to the single slot (depot, depot), which is what lets a
    route that was emptied by a destroy operator be refilled.
    """
    p = np.empty(len(route) + 1, dtype=np.int64)
    n = np.empty(len(route) + 1, dtype=np.int64)
    p[0] = depot
    n[-1] = depot
    if route:
        arr = np.asarray(route, dtype=np.int64)
        p[1:] = arr
        n[:-1] = arr
    return p, n


def _best_insertion(route, U, ctx):
    """Cheapest insertion cost and slot of every customer in ``U`` into ``route``.

    Vectorised over slots *and* candidates at once: the naive form of this, a
    Python loop over positions per customer, is the single change that makes the
    search roughly two orders of magnitude slower.
    """
    D, depot = ctx.dist, ctx.depot
    p, n = _positions(route, depot)
    # (slots, |U|): d[prev, u] + d[u, next] - d[prev, next]
    costs = D[np.ix_(p, U)] + D[np.ix_(U, n)].T - D[p, n][:, None]
    slot = costs.argmin(axis=0)
    return costs[slot, np.arange(len(U))], slot


def _repair(state: SbrpState, rng, mode: str) -> SbrpState:
    ctx = state.ctx
    if not state.unassigned:
        return state

    U = np.array(state.unassigned, dtype=np.int64)
    rng.shuffle(U)  # breaks ties between equal-cost insertions
    routes = state.routes
    k = len(routes)
    loads = state.loads()
    dem_u = ctx.demands[U]

    cost = np.empty((k, len(U)))
    slot = np.empty((k, len(U)), dtype=np.int64)
    for ri in range(k):
        cost[ri], slot[ri] = _best_insertion(routes[ri], U, ctx)

    pending = np.ones(len(U), dtype=bool)
    while pending.any():
        feasible = (loads[:, None] + dem_u[None, :]) <= ctx.capacity + 1e-9
        # A large finite sentinel rather than inf: the regret differences below
        # would otherwise evaluate inf - inf and yield NaN.
        masked = np.where(feasible, cost, _BLOCKED)
        masked[:, ~pending] = _BLOCKED

        if mode == "greedy":
            # Cheapest insertion overall, except that a customer down to a single
            # feasible route is placed first: deferring it is what strands it.
            n_feasible = np.where(masked < _BLOCKED_HALF, 1, 0).sum(axis=0)
            critical = pending & (n_feasible == 1)
            search = np.where(critical, masked, _BLOCKED) if critical.any() else masked
            ri, ui = np.unravel_index(np.argmin(search), search.shape)
            ri, ui = int(ri), int(ui)
        else:
            # Regret-k: prefer the customer that would suffer most from waiting.
            # With fewer than k feasible routes the sentinel makes the regret
            # enormous, which is the intended "place this one now" behaviour.
            order = np.sort(masked, axis=0)
            best = order[0]
            kk = min(ctx.params.regret_k, k)
            regret = np.zeros(len(U))
            for t in range(1, kk):
                regret += order[t] - best
            regret[best >= _BLOCKED_HALF] = -np.inf   # nowhere to put it at all
            regret[~pending] = -np.inf
            ui = int(np.argmax(regret))
            ri = int(np.argmin(masked[:, ui]))

        if masked[ri, ui] >= _BLOCKED_HALF:
            # No feasible slot for someone still pending. With unit demands this
            # cannot happen (k*C >= n); with general demands it is a bin-packing
            # dead end, so abandon the move and hand back the pre-destroy
            # solution, which is feasible by construction.
            if state.snapshot is None:
                raise ValueError(
                    "ALNS repair found no capacity-feasible insertion and no "
                    f"snapshot to fall back on; {k} routes x capacity "
                    f"{ctx.capacity} vs total demand {ctx.total_demand()}"
                )
            ctx.repair_dead_ends += 1
            state.routes = [r[:] for r in state.snapshot]
            state.unassigned = []
            state._obj = None
            return state

        cust = int(U[ui])
        routes[ri].insert(int(slot[ri, ui]), cust)
        loads[ri] += dem_u[ui]
        pending[ui] = False
        # Only the touched route's row is stale.
        cost[ri], slot[ri] = _best_insertion(routes[ri], U, ctx)

    state.unassigned = []
    state._obj = None
    return state


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

def _pick_q(state, rng) -> int:
    p = state.ctx.params
    n = int(sum(len(r) for r in state.routes))
    if n <= 1:
        return n
    lo = max(1, int(p.destroy_min_frac * n))
    hi = max(lo, int(p.destroy_max_frac * n))
    hi = min(hi, p.destroy_cap, n - 1)
    lo = min(lo, hi)
    return int(rng.integers(lo, hi + 1))


def _detach(state, victims) -> None:
    targets = set(int(c) for c in victims)
    if not targets:
        return
    state.snapshot = [r[:] for r in state.routes]
    for r in state.routes:
        if any(c in targets for c in r):
            r[:] = [c for c in r if c not in targets]
    state.unassigned.extend(targets)
    state._obj = None


def _biased_pick(rng, ranked, p: float) -> int:
    """Ropke-Pisinger pick: index int(len * U(0,1)^p) into a ranked list."""
    return ranked[int(len(ranked) * (rng.random() ** p))]


def _timed(fn, key):
    def wrapper(state, rng, **kwargs):
        t0 = time.perf_counter()
        out = fn(state, rng, **kwargs)
        ctx = out.ctx
        ctx.timers[key] += time.perf_counter() - t0
        ctx.op_calls[fn.__name__] = ctx.op_calls.get(fn.__name__, 0) + 1
        return out
    wrapper.__name__ = fn.__name__
    return wrapper


def _random_removal(state, rng, **kwargs):
    """Remove q customers drawn uniformly at random."""
    out = state.copy()
    assigned = [c for r in out.routes for c in r]
    q = _pick_q(out, rng)
    if q and assigned:
        victims = rng.choice(np.asarray(assigned, dtype=np.int64),
                             size=min(q, len(assigned)), replace=False)
        _detach(out, victims)
    return out


def _worst_removal(state, rng, **kwargs):
    """Remove the customers whose detour cost is largest, with a greedy bias."""
    out = state.copy()
    ctx = out.ctx
    D, depot = ctx.dist, ctx.depot

    gains, custs = [], []
    for r in out.routes:
        if not r:
            continue
        arr = np.asarray(r, dtype=np.int64)
        p, n = _positions(r, depot)
        prev, nxt = p[:-1], n[1:]
        # Saving from short-circuiting each customer out of its route.
        g = D[prev, arr] + D[arr, nxt] - D[prev, nxt]
        gains.append(g)
        custs.append(arr)
    if not custs:
        return out

    gains = np.concatenate(gains)
    custs = np.concatenate(custs)
    ranked = list(custs[np.argsort(-gains)])  # most expensive first

    q = _pick_q(out, rng)
    victims = []
    for _ in range(q):
        if not ranked:
            break
        pick = _biased_pick(rng, ranked, ctx.params.worst_p)
        ranked.remove(pick)
        victims.append(pick)
    _detach(out, victims)
    return out


def _shaw_removal(state, rng, **kwargs):
    """Shaw/related removal: tear out a spatially coherent group of customers.

    Relatedness is the symmetrised road distance, discounted when two customers
    already share a bus, so the operator opens a genuinely different partition
    rather than reshuffling one route.
    """
    out = state.copy()
    ctx = out.ctx
    sym, params = ctx.sym, ctx.params

    assigned = [c for r in out.routes for c in r]
    if len(assigned) < 2:
        return out
    q = min(_pick_q(out, rng), len(assigned))
    if q <= 0:
        return out

    route_of = {}
    for ri, r in enumerate(out.routes):
        for c in r:
            route_of[c] = ri

    bonus = params.shaw_route_bonus * float(sym[np.ix_(ctx.customers,
                                                       ctx.customers)].mean())
    seed = int(assigned[int(rng.integers(len(assigned)))])
    victims = [seed]
    pool = [c for c in assigned if c != seed]

    while len(victims) < q and pool:
        ref = victims[int(rng.integers(len(victims)))]
        pool_arr = np.asarray(pool, dtype=np.int64)
        rel = sym[ref, pool_arr].astype(float)
        same = np.fromiter((route_of.get(int(c)) == route_of.get(ref)
                            for c in pool_arr), dtype=bool, count=len(pool_arr))
        rel = rel - same * bonus          # same-route pairs count as closer
        ranked = list(pool_arr[np.argsort(rel)])  # most related first
        pick = _biased_pick(rng, ranked, params.shaw_p)
        pool.remove(pick)
        victims.append(int(pick))

    _detach(out, victims)
    return out


def _greedy_repair(state, rng, **kwargs):
    """Reinsert every removed customer at its cheapest feasible slot."""
    return _repair(state, rng, "greedy")


def _regret_repair(state, rng, **kwargs):
    """Reinsert by regret-k: serve first whoever loses most by being deferred."""
    return _repair(state, rng, "regret")


DESTROY_OPERATORS = (
    ("random_removal", _timed(_random_removal, "destroy")),
    ("worst_removal", _timed(_worst_removal, "destroy")),
    ("shaw_removal", _timed(_shaw_removal, "destroy")),
)

REPAIR_OPERATORS = (
    ("greedy_repair", _timed(_greedy_repair, "repair")),
    ("regret_repair", _timed(_regret_repair, "repair")),
)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

@dataclass
class AlnsOutcome:
    routes: list[list[int]]
    objective: float
    init_objective: float
    iterations: int
    destroy_counts: dict
    repair_counts: dict
    timers: dict = field(default_factory=dict)
    repair_dead_ends: int = 0


def run_alns(ctx: AlnsContext, initial_routes, budget_s: float,
             rng) -> AlnsOutcome:
    """Run the adaptive search from ``initial_routes`` for ``budget_s`` seconds."""
    params = ctx.params
    init = SbrpState([list(r) for r in initial_routes], [], ctx)
    init_obj = init.objective()

    engine = ALNS(rng)
    for name, op in DESTROY_OPERATORS:
        engine.add_destroy_operator(op, name=name)
    for name, op in REPAIR_OPERATORS:
        engine.add_repair_operator(op, name=name)

    select = RouletteWheel(list(params.op_scores), params.op_decay,
                           len(DESTROY_OPERATORS), len(REPAIR_OPERATORS))
    accept = SimulatedAnnealing.autofit(
        init_obj, params.sa_worse, params.sa_accept_prob, params.sa_horizon_iters
    )
    result = engine.iterate(init, select, accept, MaxRuntime(max(budget_s, 0.5)))

    best = result.best_state
    stats = result.statistics
    return AlnsOutcome(
        routes=best.export_routes(),
        objective=best.objective(),
        init_objective=init_obj,
        iterations=len(stats.objectives),
        destroy_counts={k: [int(x) for x in v]
                        for k, v in stats.destroy_operator_counts.items()},
        repair_counts={k: [int(x) for x in v]
                       for k, v in stats.repair_operator_counts.items()},
        timers=dict(ctx.timers),
        repair_dead_ends=ctx.repair_dead_ends,
    )

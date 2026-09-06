"""Loader for TSPLIB-format CVRP instances (CVRPLIB / Augerat sets).

Turns a published ``.vrp`` file into the same ``Scenario`` the synthetic
Medellin generator produces, so the ALNS solver runs on external instances
through the identical interface and its absolute solution quality can be
measured against published optima.

Two conventions are reconciled here:

* TSPLIB numbers nodes from 1 and puts the depot first; the harness requires
  the depot at ``origin_index == n``, i.e. last. Nodes are reordered on load.
* ``EDGE_WEIGHT_TYPE : EUC_2D`` means the *rounded* Euclidean distance
  ``nint(sqrt(dx^2 + dy^2))``. Published optima are defined on those integers,
  so rounding is part of the instance, not a detail to skip.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from ..core.scenario import Scenario


@dataclass
class CvrpInstance:
    name: str
    scenario: Scenario
    optimal: float | None      # best-known/optimal value from the COMMENT line
    n_trucks: int | None       # vehicle count declared by the instance
    capacity: float


def _sections(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    header: dict[str, str] = {}
    body: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION") or line in {"DEPOT_SECTION"}:
            current = line
            body[current] = []
            continue
        if ":" in line and current is None:
            key, _, val = line.partition(":")
            header[key.strip().upper()] = val.strip()
            continue
        if current is not None:
            body[current].append(line)
        elif ":" in line:
            key, _, val = line.partition(":")
            header[key.strip().upper()] = val.strip()
    return header, body


def load_cvrp(path: str | Path) -> CvrpInstance:
    path = Path(path)
    header, body = _sections(path.read_text())

    ewt = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if ewt != "EUC_2D":
        raise ValueError(f"{path.name}: unsupported EDGE_WEIGHT_TYPE {ewt!r}")

    dim = int(header["DIMENSION"])
    capacity = float(header["CAPACITY"])

    coords: dict[int, tuple[float, float]] = {}
    for line in body.get("NODE_COORD_SECTION", []):
        parts = line.split()
        coords[int(parts[0])] = (float(parts[1]), float(parts[2]))

    demands: dict[int, float] = {}
    for line in body.get("DEMAND_SECTION", []):
        parts = line.split()
        demands[int(parts[0])] = float(parts[1])

    depot_ids = [int(p) for line in body.get("DEPOT_SECTION", [])
                 for p in line.split()]
    depot_ids = [d for d in depot_ids if d > 0]
    if len(depot_ids) != 1:
        raise ValueError(f"{path.name}: expected exactly one depot, got {depot_ids}")
    depot_id = depot_ids[0]

    if len(coords) != dim:
        raise ValueError(f"{path.name}: {len(coords)} coordinates for DIMENSION {dim}")

    # Depot last, customers in ascending original id.
    order = [i for i in sorted(coords) if i != depot_id] + [depot_id]
    pts = np.array([coords[i] for i in order], dtype=float)

    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.rint(np.sqrt((delta ** 2).sum(-1)))   # TSPLIB EUC_2D rounding
    np.fill_diagonal(dist, 0.0)

    dem = np.array([demands.get(i, 0.0) for i in order], dtype=float)
    dem[-1] = 0.0                                    # depot carries no demand

    comment = header.get("COMMENT", "")
    m_opt = re.search(r"(?:Optimal value|Best value)\s*:?\s*([0-9.]+)",
                      comment, re.I)
    m_trk = re.search(r"(?:No of trucks|trucks)\s*:?\s*(\d+)", comment, re.I)

    scenario = Scenario(
        node_ids=np.array(order),
        x=pts[:, 0], y=pts[:, 1],
        dist_matrix=dist, time_matrix=dist.copy(),
        origin_index=len(order) - 1,
        bus_capacity=capacity,
        graph=nx.MultiDiGraph(),        # unused: these instances are not plotted
        n_children=dim - 1,
        seed=0,
        demands=dem,
    )
    return CvrpInstance(
        name=header.get("NAME", path.stem),
        scenario=scenario,
        optimal=float(m_opt.group(1)) if m_opt else None,
        n_trucks=int(m_trk.group(1)) if m_trk else None,
        capacity=capacity,
    )


def load_directory(directory: str | Path) -> list[CvrpInstance]:
    files = sorted(Path(directory).glob("*.vrp"),
                   key=lambda p: (len(p.stem), p.stem))
    return [load_cvrp(f) for f in files]

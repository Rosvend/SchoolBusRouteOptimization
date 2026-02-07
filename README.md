# School Bus Route Optimization

## Overview

This project explores heuristic methods for optimizing school bus routes under real-world distance constraints.
Given a set of student pickup locations and one or more depot locations, the goal is to construct efficient bus routes that minimize total travel cost while remaining computationally tractable.

---

## Problem Statement

Given:

* A set of student locations ( J = {j_1, j_2, ..., j_n} )
* One or more depot locations ( G )
* A distance function derived from a road network (non-Euclidean)

The objective is to assign students to buses and compute a route for each bus that minimizes total travel distance (or time), subject to practical constraints.

---
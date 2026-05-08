from __future__ import annotations

from abc import ABC, abstractmethod

from .scenario import Scenario
from .solution import Solution


class Solver(ABC):
    name: str = "abstract"

    @abstractmethod
    def solve(self, scenario: Scenario) -> Solution: ...

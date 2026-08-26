"""Shared types and base class for proximal stochastic optimisers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vrpsgd.objective import ObjectiveConfig, objective, sparsity
from vrpsgd.proximal import project_simplex_sparse

FloatArray = NDArray[np.float64]


@dataclass
class OptimizationHistory:
    """Per-iteration trace, recorded every ``log_every`` steps."""

    objective: list[float] = field(default_factory=list)
    sparsity: list[int] = field(default_factory=list)
    elapsed_seconds: list[float] = field(default_factory=list)


@dataclass
class OptimizationResult:
    weights: FloatArray
    history: OptimizationHistory
    algorithm: str
    n_iterations: int
    wall_time_seconds: float

    @property
    def n_nonzero(self) -> int:
        return int(np.sum(np.abs(self.weights) > 1e-6))

    def as_dict(self) -> dict:
        n = len(self.weights)
        return {
            "algorithm": self.algorithm,
            "weights": self.weights.tolist(),
            "n_assets": n,
            "n_nonzero": self.n_nonzero,
            "sparsity_pct": 100.0 * (1 - self.n_nonzero / n),
            "final_objective": self.history.objective[-1] if self.history.objective else None,
            "n_iterations": self.n_iterations,
            "wall_time_seconds": self.wall_time_seconds,
        }


class ProximalStochasticOptimizer(ABC):
    """Common scaffolding for SPGD / Prox-SVRG / Prox-SARAH / Prox-STORM.

    Subclasses implement :meth:`_run` and get objective/sparsity logging,
    simplex projection, timing, and a uniform constructor for free.
    """

    name: str = "base"

    def __init__(self, cfg: ObjectiveConfig, seed: int = 42, log_every: int = 1):
        self.cfg = cfg
        self.seed = seed
        self.log_every = log_every

    @abstractmethod
    def _run(
        self, R: FloatArray, T_max: int, rng: np.random.RandomState
    ) -> tuple[FloatArray, OptimizationHistory, int]:
        """Run the algorithm; return (final_weights, history, n_iterations)."""

    def fit(self, R: FloatArray, T_max: int = 5000) -> OptimizationResult:
        rng = np.random.RandomState(self.seed)
        t0 = time.perf_counter()
        w, history, n_iter = self._run(R, T_max, rng)
        elapsed = time.perf_counter() - t0
        return OptimizationResult(
            weights=w,
            history=history,
            algorithm=self.name,
            n_iterations=n_iter,
            wall_time_seconds=elapsed,
        )

    def _init_weights(self, n: int) -> FloatArray:
        return np.full(n, 1.0 / n)

    def _log(self, history: OptimizationHistory, w: FloatArray, R: FloatArray, t0: float, k: int) -> None:
        if k % self.log_every != 0:
            return
        history.objective.append(objective(w, R, self.cfg))
        history.sparsity.append(sparsity(w))
        history.elapsed_seconds.append(time.perf_counter() - t0)

    @staticmethod
    def _prox_step(w: FloatArray, v: FloatArray, eta: float, lam: float) -> FloatArray:
        from vrpsgd.proximal import soft_threshold

        z = soft_threshold(w - eta * v, eta * lam)
        return project_simplex_sparse(z)

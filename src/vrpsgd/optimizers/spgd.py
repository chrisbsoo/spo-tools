"""Stochastic Proximal Gradient Descent (baseline, diminishing step size)."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from vrpsgd.objective import stochastic_gradient
from vrpsgd.optimizers.base import OptimizationHistory, ProximalStochasticOptimizer

FloatArray = NDArray[np.float64]


class SPGD(ProximalStochasticOptimizer):
    """eta_k = eta0 / sqrt(1+k). O(1/sqrt(k)) convergence for convex problems."""

    name = "SPGD"

    def __init__(self, cfg, seed: int = 42, log_every: int = 1, eta0: float = 0.005):
        super().__init__(cfg, seed, log_every)
        self.eta0 = eta0

    def _run(self, R: FloatArray, T_max: int, rng: np.random.RandomState):
        T, n = R.shape
        w = self._init_weights(n)
        mu = R.mean(axis=0)
        history = OptimizationHistory()
        t0 = time.perf_counter()

        for k in range(T_max):
            eta_k = self.eta0 / np.sqrt(1.0 + k)
            i = rng.randint(T)
            g = stochastic_gradient(w, R[i], mu, self.cfg)
            w = self._prox_step(w, g, eta_k, self.cfg.lam)
            self._log(history, w, R, t0, k)

        return w, history, T_max

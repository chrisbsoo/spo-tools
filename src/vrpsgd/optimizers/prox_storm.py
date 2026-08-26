"""Proximal STORM: momentum-based recursive variance reduction, no periodic full gradients."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from vrpsgd.objective import stochastic_gradient
from vrpsgd.optimizers.base import OptimizationHistory, ProximalStochasticOptimizer

FloatArray = NDArray[np.float64]


class ProxSTORM(ProximalStochasticOptimizer):
    """d_k = grad_i(x_k) + (1-alpha)(d_{k-1} - grad_i(x_{k-1}))."""

    name = "Prox-STORM"

    def __init__(self, cfg, seed: int = 42, log_every: int = 1, eta: float = 0.005, alpha: float = 0.1):
        super().__init__(cfg, seed, log_every)
        self.eta = eta
        self.alpha = alpha

    def _run(self, R: FloatArray, T_max: int, rng: np.random.RandomState):
        T, n = R.shape
        w = self._init_weights(n)
        mu = R.mean(axis=0)
        history = OptimizationHistory()
        t0 = time.perf_counter()

        i0 = rng.randint(T)
        d = stochastic_gradient(w, R[i0], mu, self.cfg)
        w_prev = w.copy()
        w = self._prox_step(w, d, self.eta, self.cfg.lam)
        self._log(history, w, R, t0, 0)

        for k in range(1, T_max):
            i = rng.randint(T)
            g_new = stochastic_gradient(w, R[i], mu, self.cfg)
            g_old = stochastic_gradient(w_prev, R[i], mu, self.cfg)
            d = g_new + (1.0 - self.alpha) * (d - g_old)
            w_prev = w.copy()
            w = self._prox_step(w, d, self.eta, self.cfg.lam)
            self._log(history, w, R, t0, k)

        return w, history, T_max

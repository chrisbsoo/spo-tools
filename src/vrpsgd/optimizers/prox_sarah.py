"""Proximal SARAH: recursive (biased, low-variance) gradient estimator."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from vrpsgd.objective import full_gradient, stochastic_gradient
from vrpsgd.optimizers.base import OptimizationHistory, ProximalStochasticOptimizer

FloatArray = NDArray[np.float64]


class ProxSARAH(ProximalStochasticOptimizer):
    """v_k = v_{k-1} + grad_i(x_k) - grad_i(x_{k-1}) recursive estimator."""

    name = "Prox-SARAH"

    def __init__(self, cfg, seed: int = 42, log_every: int = 1, eta: float = 0.005, inner_loop_len: int | None = None):
        super().__init__(cfg, seed, log_every)
        self.eta = eta
        self.inner_loop_len = inner_loop_len

    def _run(self, R: FloatArray, T_max: int, rng: np.random.RandomState):
        T, n = R.shape
        m = self.inner_loop_len or T
        w = self._init_weights(n)
        mu = R.mean(axis=0)
        history = OptimizationHistory()
        t0 = time.perf_counter()

        total_iters = 0
        while total_iters < T_max:
            v = full_gradient(w, R, self.cfg)
            w_prev = w.copy()
            w = self._prox_step(w, v, self.eta, self.cfg.lam)
            total_iters += 1
            self._log(history, w, R, t0, total_iters)

            for _ in range(m - 1):
                if total_iters >= T_max:
                    break
                i = rng.randint(T)
                g_new = stochastic_gradient(w, R[i], mu, self.cfg)
                g_old = stochastic_gradient(w_prev, R[i], mu, self.cfg)
                v = v + (g_new - g_old)
                w_prev = w.copy()
                w = self._prox_step(w, v, self.eta, self.cfg.lam)
                total_iters += 1
                self._log(history, w, R, t0, total_iters)

        return w, history, total_iters

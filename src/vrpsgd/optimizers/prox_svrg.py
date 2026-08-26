"""Proximal SVRG: anchor-and-correct variance-reduced gradient method."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from vrpsgd.objective import full_gradient, stochastic_gradient
from vrpsgd.optimizers.base import OptimizationHistory, ProximalStochasticOptimizer

FloatArray = NDArray[np.float64]


class ProxSVRG(ProximalStochasticOptimizer):
    """Snapshot a full gradient every ``inner_loop_len`` steps (default: one epoch)."""

    name = "Prox-SVRG"

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
            x_snap = w.copy()
            grad_snap_full = full_gradient(x_snap, R, self.cfg)

            for _ in range(m):
                if total_iters >= T_max:
                    break
                i = rng.randint(T)
                g_w = stochastic_gradient(w, R[i], mu, self.cfg)
                g_snap = stochastic_gradient(x_snap, R[i], mu, self.cfg)
                v = g_w - g_snap + grad_snap_full
                w = self._prox_step(w, v, self.eta, self.cfg.lam)
                total_iters += 1
                self._log(history, w, R, t0, total_iters)

        return w, history, total_iters

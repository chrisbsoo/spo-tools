"""Objective functions and gradients for sparse portfolio optimisation.

The thesis (MATH30009) derives a mean-variance-style objective

    f(w) = 0.5 * w^T Sigma w - gamma * mu^T w

but the reference implementation instead used the *second-moment* matrix
R^T R / T in place of the covariance matrix Sigma = Cov(R), and a sign
convention under which larger gamma means *more* return-seeking rather
than more risk-averse. Both are reproduced here for backward compatibility
(``mode="second_moment"``) and fixed in the default (``mode="covariance"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ObjectiveMode = Literal["covariance", "second_moment"]


@dataclass(frozen=True)
class ObjectiveConfig:
    """Hyperparameters shared by every optimiser in this package."""

    gamma: float = 2.0
    """Risk-aversion parameter. Larger gamma = more weight on variance penalty."""

    lam: float = 0.01
    """L1 regularisation strength controlling sparsity."""

    mode: ObjectiveMode = "covariance"
    """'covariance' (thesis-correct mean-variance) or 'second_moment' (legacy)."""


def _quad_term(R: FloatArray, mode: ObjectiveMode) -> FloatArray:
    """Return the T x n matrix used to build the quadratic (risk) term."""
    if mode == "covariance":
        return R - R.mean(axis=0, keepdims=True)
    return R  # legacy second-moment behaviour


def stochastic_gradient(w: FloatArray, r_t: FloatArray, mu: FloatArray, cfg: ObjectiveConfig) -> FloatArray:
    """Single-sample gradient estimate of the smooth part at day t."""
    if cfg.mode == "covariance":
        centred = r_t - mu
        return centred * centred.dot(w) - cfg.gamma * mu
    return (r_t.dot(w) - cfg.gamma) * r_t


def full_gradient(w: FloatArray, R: FloatArray, cfg: ObjectiveConfig) -> FloatArray:
    """Full-batch gradient of the smooth part, averaged over all T samples."""
    mu = R.mean(axis=0)
    Q = _quad_term(R, cfg.mode)
    Qw = Q.dot(w)
    quad_grad = Q.T.dot(Qw) / len(R)
    return quad_grad - cfg.gamma * mu


def objective(w: FloatArray, R: FloatArray, cfg: ObjectiveConfig) -> float:
    """Full composite objective f(w) + lambda ||w||_1 (to be minimised)."""
    mu = R.mean(axis=0)
    Q = _quad_term(R, cfg.mode)
    Qw = Q.dot(w)
    f = 0.5 * float(np.mean(Qw ** 2)) - cfg.gamma * float(mu.dot(w))
    return f + cfg.lam * float(np.linalg.norm(w, 1))


def sparsity(w: FloatArray, tol: float = 1e-6) -> int:
    """Number of effectively-zero weights."""
    return int(np.sum(np.abs(w) < tol))

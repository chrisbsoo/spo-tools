"""Proximal and projection operators for L1-regularised simplex-constrained optimisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def soft_threshold(v: FloatArray, threshold: float) -> FloatArray:
    """Proximal operator of ``threshold * ||.||_1``: elementwise soft-thresholding.

    prox_{t||.||_1}(v) = sign(v) * max(|v| - t, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def project_simplex(w: FloatArray) -> FloatArray:
    """Euclidean projection of ``w`` onto the probability simplex {w: sum(w)=1, w>=0}.

    Sorting-based algorithm, Duchi et al. (2008), O(n log n).
    """
    n = w.shape[0]
    u = np.sort(w)[::-1]
    cumsum = np.cumsum(u)
    idx = np.arange(1, n + 1)
    cond = u + (1.0 - cumsum) / idx > 0
    if not np.any(cond):
        # Defensive fallback only: cond[0] = u[0] + (1 - cumsum[0]) = 1 > 0
        # always holds, so this branch is mathematically unreachable. Kept
        # in case of future floating-point edge cases at that boundary.
        return np.full(n, 1.0 / n)  # pragma: no cover
    rho = np.max(np.where(cond)[0])
    theta = (cumsum[rho] - 1.0) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def project_simplex_sparse(w: FloatArray) -> FloatArray:
    """Sparsity-preserving simplex projection.

    Standard simplex projection can reallocate mass onto coordinates that were
    just zeroed by soft-thresholding, silently undoing the L1 penalty's sparsity
    effect. This variant restricts the projection to the current support so that
    zeroed weights stay zero.
    """
    support = w > 0
    if not np.any(support):
        return np.full(w.shape[0], 1.0 / w.shape[0])
    out = np.zeros_like(w)
    out[support] = project_simplex(w[support])
    return out

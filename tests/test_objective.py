"""Correctness tests for the smooth objective and its analytic gradient.

The most important test in this file is the finite-difference check: it
verifies the analytic ``full_gradient`` matches a numerical approximation
of the true gradient of ``objective``'s smooth part, for both objective
modes. This is the kind of test that's easy to skip and catches real bugs
(e.g. a sign error or wrong matrix would break every optimiser silently).
"""

from __future__ import annotations

import numpy as np
import pytest

from vrpsgd.objective import ObjectiveConfig, full_gradient, objective, sparsity


def _smooth_part(w, R, cfg):
    """Recompute just the smooth term (objective minus the L1 penalty)."""
    return objective(w, R, cfg) - cfg.lam * float(np.linalg.norm(w, 1))


@pytest.mark.parametrize("mode", ["covariance", "second_moment"])
def test_full_gradient_matches_finite_differences(mode):
    rng = np.random.RandomState(0)
    T, n = 200, 8
    R = rng.randn(T, n) * 0.01
    cfg = ObjectiveConfig(gamma=1.5, lam=0.0, mode=mode)  # lam=0: isolate smooth part
    w = rng.dirichlet(np.ones(n))  # a valid, generic simplex point

    analytic = full_gradient(w, R, cfg)

    eps = 1e-6
    numeric = np.zeros(n)
    for i in range(n):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        numeric[i] = (_smooth_part(w_plus, R, cfg) - _smooth_part(w_minus, R, cfg)) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_objective_uniform_portfolio_matches_hand_computation():
    R = np.array([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.02]])
    w = np.array([0.5, 0.5])
    cfg = ObjectiveConfig(gamma=1.0, lam=0.1, mode="covariance")

    mu = R.mean(axis=0)
    centred = R - mu
    quad = float(np.mean((centred @ w) ** 2)) * 0.5
    expected = quad - 1.0 * float(mu @ w) + 0.1 * float(np.abs(w).sum())

    assert objective(w, R, cfg) == pytest.approx(expected, abs=1e-10)


def test_sparsity_counts_near_zero_entries():
    w = np.array([1e-8, 0.5, -1e-9, 0.5])
    assert sparsity(w) == 2

from __future__ import annotations

import numpy as np
import pytest

from vrpsgd.objective import ObjectiveConfig, objective
from vrpsgd.optimizers import REGISTRY


@pytest.fixture
def synthetic_returns():
    """A small, deterministic synthetic return matrix with a clear 'best' asset,
    so a working optimiser should concentrate weight there.
    """
    rng = np.random.RandomState(0)
    T, n = 400, 12
    R = rng.randn(T, n) * 0.01
    R[:, 0] += 0.003  # asset 0 has a clearly higher mean return
    return R


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_optimizer_output_is_a_valid_portfolio(name, synthetic_returns):
    cfg = ObjectiveConfig(gamma=2.0, lam=0.01)
    optimizer = REGISTRY[name](cfg, seed=42)
    result = optimizer.fit(synthetic_returns, T_max=300)

    w = result.weights
    assert w.shape == (synthetic_returns.shape[1],)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_optimizer_reduces_objective_from_uniform_start(name, synthetic_returns):
    cfg = ObjectiveConfig(gamma=2.0, lam=0.01)
    n = synthetic_returns.shape[1]
    uniform_obj = objective(np.full(n, 1 / n), synthetic_returns, cfg)

    optimizer = REGISTRY[name](cfg, seed=42)
    result = optimizer.fit(synthetic_returns, T_max=500)

    assert result.history.objective[-1] < uniform_obj


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_optimizer_is_deterministic_given_seed(name, synthetic_returns):
    cfg = ObjectiveConfig(gamma=2.0, lam=0.01)
    r1 = REGISTRY[name](cfg, seed=7).fit(synthetic_returns, T_max=200)
    r2 = REGISTRY[name](cfg, seed=7).fit(synthetic_returns, T_max=200)
    np.testing.assert_allclose(r1.weights, r2.weights)


def test_variance_reduced_methods_beat_spgd_within_fixed_budget(synthetic_returns):
    """Regression test capturing the thesis's headline empirical claim:
    variance-reduced methods reach a lower objective than plain SPGD for a
    fixed, small iteration budget on a well-conditioned synthetic problem.
    """
    cfg = ObjectiveConfig(gamma=2.0, lam=0.01)
    budget = 400

    spgd_obj = REGISTRY["spgd"](cfg, seed=1).fit(synthetic_returns, T_max=budget).history.objective[-1]
    svrg_obj = REGISTRY["prox-svrg"](cfg, seed=1).fit(synthetic_returns, T_max=budget).history.objective[-1]
    sarah_obj = REGISTRY["prox-sarah"](cfg, seed=1).fit(synthetic_returns, T_max=budget).history.objective[-1]

    assert svrg_obj <= spgd_obj
    assert sarah_obj <= spgd_obj

import numpy as np
import pytest

from vrpsgd.proximal import project_simplex, project_simplex_sparse, soft_threshold


def test_soft_threshold_zeros_small_values():
    v = np.array([0.5, -0.05, 0.2, -0.3])
    out = soft_threshold(v, threshold=0.1)
    assert out[1] == pytest.approx(0.0)  # |0.05| < 0.1 -> zeroed
    assert out[0] == pytest.approx(0.4)  # 0.5 - 0.1
    assert out[3] == pytest.approx(-0.2)  # -(0.3 - 0.1)


def test_soft_threshold_zero_threshold_is_identity():
    v = np.array([1.0, -2.0, 0.0, 3.5])
    np.testing.assert_allclose(soft_threshold(v, 0.0), v)


@pytest.mark.parametrize("seed", range(10))
def test_project_simplex_produces_valid_distribution(seed):
    rng = np.random.RandomState(seed)
    w = rng.randn(20)
    out = project_simplex(w)
    assert np.all(out >= -1e-12)
    assert out.sum() == pytest.approx(1.0, abs=1e-8)


def test_project_simplex_fixed_point_for_valid_input():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    out = project_simplex(w)
    np.testing.assert_allclose(out, w, atol=1e-8)


def test_project_simplex_handles_all_negative_input():
    # Note: the "no coordinate satisfies cond -> return uniform" fallback in
    # project_simplex is defensive but mathematically unreachable, because
    # cond[0] = u[0] + (1 - cumsum[0]) = u[0] + 1 - u[0] = 1 > 0 always holds.
    # So even all-negative input is projected normally, not routed to uniform.
    w = np.array([-1.0, -2.0, -0.5])
    out = project_simplex(w)
    assert np.all(out >= -1e-12)
    assert out.sum() == pytest.approx(1.0, abs=1e-8)



def test_project_simplex_sparse_preserves_zeros():
    # After soft-thresholding two coords are exactly zero; sparse projection
    # must not reassign mass to them (unlike plain project_simplex).
    w = np.array([0.6, 0.0, 0.3, 0.0])
    out = project_simplex_sparse(w)
    assert out[1] == 0.0
    assert out[3] == 0.0
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out >= 0)


def test_project_simplex_sparse_all_zero_returns_uniform():
    w = np.zeros(5)
    out = project_simplex_sparse(w)
    np.testing.assert_allclose(out, np.full(5, 0.2))

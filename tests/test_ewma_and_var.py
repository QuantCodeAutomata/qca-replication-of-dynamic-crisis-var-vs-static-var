"""Unit tests for EWMA estimator and VaR simulation engine."""
from __future__ import annotations

import numpy as np
import pytest

from src.ewma import cov_to_corr, ewma_covariance, ewma_covariance_path
from src.manifold import equicorrelation_matrix, geodesic_distance
from src.var_engine import (
    average_pairwise_correlation,
    scenario_diagnostics,
    simulate_dynamic_var,
    simulate_static_var,
)


def _synthetic_returns(n: int = 5, T: int = 1500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + np.eye(n) * 0.01
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((T, n))
    return (Z @ L.T) * 0.01


def test_ewma_covariance_shape_and_symmetry():
    R = _synthetic_returns()
    S = ewma_covariance(R)
    assert S.shape == (R.shape[1], R.shape[1])
    assert np.allclose(S, S.T)
    eig = np.linalg.eigvalsh(S)
    assert eig.min() > 0


def test_ewma_covariance_path_length():
    R = _synthetic_returns()
    path = ewma_covariance_path(R, burn_in=60)
    assert len(path) == len(R) - 60
    for S in path[:3]:
        assert S.shape == (R.shape[1], R.shape[1])
        assert np.allclose(S, S.T)


def test_ewma_covariance_includes_last_return():
    """The terminal snapshot must update with R[-1], not stop at R[-2]."""
    lam = 0.8
    burn_in = 3
    R = np.array(
        [
            [0.01, -0.01],
            [0.02, 0.00],
            [-0.01, 0.01],
            [0.00, 0.02],
            [0.25, -0.30],  # deliberately dominant final observation
        ]
    )
    expected = np.cov(R[:burn_in].T, ddof=0)
    for t in range(burn_in, len(R)):
        r = R[t].reshape(-1, 1)
        expected = lam * expected + (1.0 - lam) * (r @ r.T)

    actual = ewma_covariance(R, lam=lam, burn_in=burn_in)

    assert np.allclose(actual, expected)


def test_ewma_path_terminal_matches_snapshot():
    R = _synthetic_returns(T=100)

    path = ewma_covariance_path(R, lam=0.94, burn_in=20)
    snapshot = ewma_covariance(R, lam=0.94, burn_in=20)

    assert len(path) == 80
    assert np.allclose(path[-1], snapshot)


def test_cov_to_corr_diagonal_unity():
    R = _synthetic_returns()
    S = ewma_covariance(R)
    sigma, C = cov_to_corr(S)
    assert np.allclose(np.diag(C), 1.0, atol=1e-12)
    assert (sigma > 0).all()


def test_static_var_positive_and_es_ge_var():
    n = 6
    C = equicorrelation_matrix(n, 0.3)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    out = simulate_static_var(C, sigma, w, T=30, n_paths=2000, seed=0)
    assert out.var > 0
    assert out.es >= out.var - 1e-12


def test_dynamic_var_requires_calibrated_bridge_sigma():
    n = 3
    C = equicorrelation_matrix(n, 0.3)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)

    with pytest.raises(ValueError, match="sigma_bridge must be supplied"):
        simulate_dynamic_var(C, sigma, w, n_paths=1)


def test_dynamic_var_matches_static_when_bridge_target_equals_start():
    n = 5
    rho = 0.3
    C = equicorrelation_matrix(n, rho)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    static = simulate_static_var(C, sigma, w, T=30, n_paths=4000, seed=42)
    # If start == crisis regime, bridge collapses drift; only diffusion noise remains.
    dyn = simulate_dynamic_var(
        C, sigma, w, rho_crisis=rho, T=30, sigma_bridge=1e-6,
        n_paths=4000, seed=42,
    )
    # With almost-zero diffusion, static and dynamic VaR should be within 10% of each other.
    assert abs(dyn.var - static.var) / static.var < 0.10


def test_dynamic_var_exceeds_static_when_moving_toward_crisis():
    n = 8
    C = equicorrelation_matrix(n, 0.2)  # low starting correlation
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    static = simulate_static_var(C, sigma, w, T=30, n_paths=6000, seed=1)
    dyn = simulate_dynamic_var(
        C, sigma, w, rho_crisis=0.9, T=30, sigma_bridge=0.11,
        n_paths=6000, seed=2,
    )
    assert dyn.var > static.var  # correlation synchronization ⇒ fatter tail


def test_scenario_diagnostics_geometry_consistent():
    C = equicorrelation_matrix(10, 0.25)
    d = scenario_diagnostics(C, rho_crisis=0.9)
    assert d["geodesic_distance"] > 0
    assert 0.24 < d["avg_corr_initial"] < 0.26


def test_average_pairwise_correlation_matches_manual():
    C = np.array([[1.0, 0.5, 0.2],
                  [0.5, 1.0, 0.4],
                  [0.2, 0.4, 1.0]])
    assert average_pairwise_correlation(C) == pytest.approx((0.5 + 0.2 + 0.4) / 3)


def test_ewma_covariance_burn_in_effect():
    R = _synthetic_returns(T=80)
    S1 = ewma_covariance(R, burn_in=20)
    S2 = ewma_covariance(R, burn_in=60)
    # Different burn-ins yield different EWMA covariances when history is short.
    assert not np.allclose(S1, S2, atol=1e-10)
    for S in (S1, S2):
        assert np.linalg.eigvalsh(S).min() > 0


def test_simulate_static_var_accepts_single_asset():
    C = np.array([[1.0]])
    sigma = np.array([0.02])
    w = np.array([1.0])
    out = simulate_static_var(C, sigma, w, T=30, n_paths=2000, seed=0)
    assert out.var > 0


def test_simulate_var_zero_positions_zero_loss():
    n = 5
    C = equicorrelation_matrix(n, 0.4)
    sigma = np.full(n, 0.02)
    w = np.zeros(n)
    out = simulate_static_var(C, sigma, w, T=30, n_paths=500, seed=0)
    assert abs(out.var) < 1e-12
    assert abs(out.es) < 1e-12

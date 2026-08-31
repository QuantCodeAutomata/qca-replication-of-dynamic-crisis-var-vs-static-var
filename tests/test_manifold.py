"""Unit tests for the Cholesky-manifold geometry."""
from __future__ import annotations

import numpy as np
import pytest

from src.manifold import (
    bridge_step,
    cholesky_to_correlation,
    correlation_to_cholesky,
    diffusion_step,
    equicorrelation_matrix,
    exp_map_sphere,
    geodesic_distance,
    log_map_sphere,
    random_correlation_via_hemispheres,
    simulate_bridge_path,
    validate_correlation_matrix,
)


def test_equicorrelation_psd_bounds():
    C = equicorrelation_matrix(10, 0.9)
    eig = np.linalg.eigvalsh(C)
    assert eig.min() > -1e-10
    with pytest.raises(ValueError):
        equicorrelation_matrix(10, 1.0)  # rho must be strictly < 1
    with pytest.raises(ValueError):
        equicorrelation_matrix(10, -1.0)  # violates -1/(n-1)


def test_cholesky_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(5):
        C = random_correlation_via_hemispheres(8, rng)
        L = correlation_to_cholesky(C)
        C2 = cholesky_to_correlation(L)
        assert np.allclose(C, C2, atol=1e-10)
        assert np.all(np.diag(L) >= 0)


def test_row_norms_unit_after_diffusion():
    rng = np.random.default_rng(2)
    C = equicorrelation_matrix(6, 0.3)
    L = correlation_to_cholesky(C)
    for _ in range(50):
        L = diffusion_step(L, dt=1.0, sigma=0.15, rng=rng)
    row_norms = np.linalg.norm(L, axis=1)
    # Row 0 = (1,0,...,0), norm 1; other rows preserved by sphere exp map.
    assert np.allclose(row_norms, 1.0, atol=1e-10)


def test_diffusion_preserves_correlation_structure():
    """After many steps the reconstruction should still be a valid correlation matrix."""
    rng = np.random.default_rng(3)
    C = random_correlation_via_hemispheres(10, rng)
    L = correlation_to_cholesky(C)
    for _ in range(60):
        L = diffusion_step(L, dt=1.0, sigma=0.11, rng=rng)
        C_t = cholesky_to_correlation(L)
        ok, d = validate_correlation_matrix(C_t)
        assert ok, d


def test_bridge_hits_crisis_direction():
    """Average correlation should rise significantly along the bridge."""
    rng = np.random.default_rng(4)
    n = 10
    C0 = equicorrelation_matrix(n, 0.2)
    K = correlation_to_cholesky(equicorrelation_matrix(n, 0.9))
    L0 = correlation_to_cholesky(C0)
    T = 30
    L_path = simulate_bridge_path(L0, K, T, sigma=0.11, rng=rng)
    mask = ~np.eye(n, dtype=bool)
    avg0 = cholesky_to_correlation(L_path[0])[mask].mean()
    avgT = cholesky_to_correlation(L_path[-1])[mask].mean()
    assert avgT > avg0 + 0.2  # meaningful progression toward 0.9


def test_geodesic_distance_zero_for_same_matrix():
    C = equicorrelation_matrix(5, 0.4)
    # arccos near 1 leaks ~sqrt(eps) noise; that's the manifold's Lipschitz bound.
    assert geodesic_distance(C, C) == pytest.approx(0.0, abs=1e-4)


def test_geodesic_distance_symmetric():
    rng = np.random.default_rng(5)
    A = random_correlation_via_hemispheres(6, rng)
    B = random_correlation_via_hemispheres(6, rng)
    assert geodesic_distance(A, B) == pytest.approx(geodesic_distance(B, A), rel=1e-10)


def test_geodesic_distance_positive_between_distinct():
    C0 = equicorrelation_matrix(8, 0.1)
    C1 = equicorrelation_matrix(8, 0.9)
    assert geodesic_distance(C0, C1) > 0.5


def test_log_map_is_tangent():
    """Log map output should be tangent: v · L = 0."""
    rng = np.random.default_rng(6)
    n = 6
    C0 = random_correlation_via_hemispheres(n, rng)
    K = correlation_to_cholesky(equicorrelation_matrix(n, 0.9))
    L = correlation_to_cholesky(C0)
    for i in range(1, n):
        v, theta = log_map_sphere(L[i, : i + 1], K[i, : i + 1])
        assert abs(np.dot(v, L[i, : i + 1])) < 1e-8
        assert theta >= 0


def test_exp_map_preserves_unit_norm():
    rng = np.random.default_rng(7)
    L = np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0])
    v = rng.standard_normal(3) * 0.2
    # project to tangent
    v = v - np.dot(v, L) * L
    L_new = exp_map_sphere(L, v)
    assert np.isclose(np.linalg.norm(L_new), 1.0, atol=1e-12)


def test_bridge_step_produces_valid_L():
    rng = np.random.default_rng(8)
    n = 8
    L = correlation_to_cholesky(random_correlation_via_hemispheres(n, rng))
    K = correlation_to_cholesky(equicorrelation_matrix(n, 0.9))
    L1 = bridge_step(L, K, t=0.0, T=30.0, dt=1.0, sigma=0.11, rng=rng)
    ok, d = validate_correlation_matrix(cholesky_to_correlation(L1))
    assert ok, d


def test_random_correlation_via_hemispheres_is_valid():
    rng = np.random.default_rng(9)
    for _ in range(20):
        C = random_correlation_via_hemispheres(10, rng)
        ok, _ = validate_correlation_matrix(C)
        assert ok


def test_geodesic_distance_bounded_by_pi_sqrt_n():
    """Each row angle is at most pi, so total distance ≤ pi * sqrt(n-1)."""
    rng = np.random.default_rng(10)
    n = 10
    for _ in range(5):
        A = random_correlation_via_hemispheres(n, rng)
        B = random_correlation_via_hemispheres(n, rng)
        d = geodesic_distance(A, B)
        assert 0 <= d <= np.pi * np.sqrt(n - 1) + 1e-8

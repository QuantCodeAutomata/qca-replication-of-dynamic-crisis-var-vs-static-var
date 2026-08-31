"""Cholesky-manifold geometry for correlation matrices.

Custom implementation — Context7 confirmed no library equivalent exists for
row-wise Brownian bridges on the Cholesky manifold of correlation matrices
as defined in the paper. `geomstats` provides sphere geometry primitives, but
the lower-triangular Cholesky assembly and the row-wise SDE are paper-specific.

Every correlation matrix ``C`` with unit diagonal can be written as
``C = L L^T`` where ``L`` is lower-triangular, row ``i`` has unit norm and
lies on the upper hemisphere ``S_+^{i-1}``. Row 0 is fixed at ``(1,0,...,0)``.

References
----------
* Paper Section 3–4 (bridge SDE, log map, exp map).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Numerical safeguards — used consistently across the manifold pipeline.
_CLIP_EPS = 1.0 - 1e-12
_SMALL_ANGLE = 1e-8
_SMALL_NORM = 1e-12


def correlation_to_cholesky(C: np.ndarray) -> np.ndarray:
    """Return the lower-triangular Cholesky factor with positive diagonal.

    Parameters
    ----------
    C : (n, n) correlation matrix (symmetric, unit diagonal, PSD).

    Returns
    -------
    L : (n, n) lower-triangular float64 matrix with ``C == L @ L.T``.
    """
    if not np.allclose(C, C.T, atol=1e-8):
        C = 0.5 * (C + C.T)
    # tiny jitter only if strictly needed (log the fact upstream)
    try:
        L = np.linalg.cholesky(C.astype(np.float64))
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.eye(C.shape[0])
        L = np.linalg.cholesky(C.astype(np.float64) + jitter)
    return L


def cholesky_to_correlation(L: np.ndarray) -> np.ndarray:
    """Reconstruct the correlation matrix from its Cholesky factor."""
    return L @ L.T


def equicorrelation_matrix(n: int, rho: float) -> np.ndarray:
    """Build the equicorrelation matrix with off-diagonal ``rho``.

    Enforces the PSD condition ``-1/(n-1) <= rho < 1``.
    """
    if not (-1.0 / (n - 1) <= rho < 1.0):
        raise ValueError(
            f"rho={rho} violates equicorrelation PSD bound [-1/(n-1)={-1.0/(n-1):.4f}, 1)."
        )
    C = np.full((n, n), rho, dtype=np.float64)
    np.fill_diagonal(C, 1.0)
    return C


def _project_tangent(L_row: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Project ``Z`` onto the tangent space of the sphere at ``L_row``.

    ``pi_{T_{L_i}}(Z) = Z - (L_i . Z) L_i``.
    """
    return Z - np.dot(L_row, Z) * L_row


def log_map_sphere(L_row: np.ndarray, K_row: np.ndarray) -> Tuple[np.ndarray, float]:
    """Sphere logarithmic map from ``L_row`` toward ``K_row``.

    ``Log_{L}(K) = (theta / sin(theta)) (K - L cos(theta))`` with
    ``theta = arccos(clip(L . K, -1, 1))``.

    Returns the tangent vector and ``theta``. Uses a first-order tangent
    projection for very small angles to avoid division by ``sin(theta) ~ 0``.
    """
    dot = float(np.clip(np.dot(L_row, K_row), -_CLIP_EPS, _CLIP_EPS))
    theta = float(np.arccos(dot))
    if theta < _SMALL_ANGLE:
        # tangent projection of (K - L); for coincident points this is ~0
        return _project_tangent(L_row, K_row - L_row), theta
    return (theta / np.sin(theta)) * (K_row - L_row * np.cos(theta)), theta


def exp_map_sphere(L_row: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sphere exponential map: ``exp_L(v) = L cos(|v|) + (v/|v|) sin(|v|)``.

    For ``|v|`` numerically zero we return ``L`` unchanged (renormalized).
    """
    norm_v = float(np.linalg.norm(v))
    if norm_v < _SMALL_NORM:
        out = L_row
    else:
        out = L_row * np.cos(norm_v) + (v / norm_v) * np.sin(norm_v)
    # renormalize to combat floating-point drift
    n = float(np.linalg.norm(out))
    if n > 0:
        out = out / n
    return out


def geodesic_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Cholesky-manifold geodesic distance between two correlation matrices.

    ``d(C, D) = sqrt(sum_i arccos^2(L_i . K_i))`` per the paper's definition.
    """
    L = correlation_to_cholesky(C1)
    K = correlation_to_cholesky(C2)
    n = L.shape[0]
    s = 0.0
    for i in range(1, n):  # row 0 is always (1,0,...,0), contributes 0
        Li = L[i, : i + 1]
        Ki = K[i, : i + 1]
        # both rows unit norm — clip and arccos
        dot = float(np.clip(np.dot(Li, Ki) / (np.linalg.norm(Li) * np.linalg.norm(Ki)),
                            -_CLIP_EPS, _CLIP_EPS))
        s += np.arccos(dot) ** 2
    return float(np.sqrt(s))


def bridge_step(
    L: np.ndarray,
    K: np.ndarray,
    t: float,
    T: float,
    dt: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Euler-Maruyama step of the row-wise Brownian bridge on Chol_n.

    SDE (paper): ``dL_i = Log_{L_i}(K_i) / (T - t) dt + sigma (I - L_i L_i^T) dW_i``.

    Row 0 is fixed at ``(1, 0, ..., 0)`` and is never updated.
    """
    n = L.shape[0]
    L_new = L.copy()
    remaining = max(T - t, dt)  # avoid division by 0 at the last step
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n):
        Li = L[i, : i + 1]
        Ki = K[i, : i + 1]
        drift_vec, _ = log_map_sphere(Li, Ki)
        drift = drift_vec * (dt / remaining)
        Z = rng.standard_normal(i + 1)
        diffusion = sigma * _project_tangent(Li, Z) * sqrt_dt
        v = drift + diffusion
        L_new[i, : i + 1] = exp_map_sphere(Li, v)
    return L_new


def diffusion_step(
    L: np.ndarray,
    dt: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Euler-Maruyama step of pure manifold diffusion (no drift).

    SDE: ``dL_i = sigma (I - L_i L_i^T) dW_i``. Used in the exp_2 sanity check.
    """
    n = L.shape[0]
    L_new = L.copy()
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n):
        Li = L[i, : i + 1]
        Z = rng.standard_normal(i + 1)
        v = sigma * _project_tangent(Li, Z) * sqrt_dt
        L_new[i, : i + 1] = exp_map_sphere(Li, v)
    return L_new


def simulate_bridge_path(
    L0: np.ndarray,
    K: np.ndarray,
    T: int,
    sigma: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> np.ndarray:
    """Simulate a full bridge path L_0 ... L_T on the Cholesky manifold.

    Returns an array ``L_path`` of shape ``(T + 1, n, n)``.
    """
    n = L0.shape[0]
    L_path = np.zeros((T + 1, n, n))
    L_path[0] = L0
    L = L0.copy()
    for k in range(T):
        t = k * dt
        L = bridge_step(L, K, t, T * dt, dt, sigma, rng)
        L_path[k + 1] = L
    return L_path


def random_correlation_via_hemispheres(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a valid random correlation matrix by drawing rows on hemispheres.

    Row 0 = (1, 0, ..., 0). For ``i >= 1`` draw a standard normal in R^{i+1},
    normalize to the unit sphere, and enforce the last-nonzero-positive
    convention so the diagonal of ``L`` is positive. This yields a valid
    lower-triangular Cholesky factor of a correlation matrix.
    """
    L = np.zeros((n, n))
    L[0, 0] = 1.0
    for i in range(1, n):
        z = rng.standard_normal(i + 1)
        z /= np.linalg.norm(z)
        # ensure L[i, i] > 0 (positive-diagonal Cholesky convention)
        if z[-1] < 0:
            z = -z
        L[i, : i + 1] = z
    return L @ L.T


def validate_correlation_matrix(
    C: np.ndarray, atol: float = 1e-8
) -> Tuple[bool, dict]:
    """Return (is_valid, diagnostics) for a candidate correlation matrix."""
    n = C.shape[0]
    diag_dev = float(np.max(np.abs(np.diag(C) - 1.0)))
    sym_err = float(np.max(np.abs(C - C.T)))
    try:
        eigvals = np.linalg.eigvalsh(0.5 * (C + C.T))
        min_eig = float(np.min(eigvals))
    except np.linalg.LinAlgError:
        min_eig = float("nan")
    ok = (diag_dev < atol) and (sym_err < atol) and (min_eig > -1e-10)
    return ok, {
        "n": n,
        "diag_dev": diag_dev,
        "symmetry_err": sym_err,
        "min_eig": min_eig,
    }

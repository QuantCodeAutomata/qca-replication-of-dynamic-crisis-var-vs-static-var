"""Static vs dynamic (crisis-bridge) VaR/ES Monte-Carlo engine.

The dynamic engine uses row-wise Brownian-bridge dynamics on the Cholesky
manifold as defined in the paper. Positions and volatilities are held fixed
over the horizon; only the correlation structure evolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.manifold import (
    bridge_step,
    cholesky_to_correlation,
    correlation_to_cholesky,
    equicorrelation_matrix,
    geodesic_distance,
)


@dataclass
class VaRResult:
    """Container for the outputs of a VaR simulation."""
    var: float                       # 95% VaR as a positive loss fraction (of gross notional)
    es: float                        # 95% ES (average loss beyond the VaR level)
    losses: np.ndarray               # (n_paths,) portfolio loss samples
    correlation_paths: Optional[np.ndarray] = None  # (n_paths, T+1, n, n) if kept
    avg_corr_paths: Optional[np.ndarray] = None     # (n_paths, T+1)


def _mvn_sample(
    C: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray
) -> np.ndarray:
    """Return one Gaussian return vector for covariance ``diag(sigma) C diag(sigma)``.

    ``epsilon`` is a pre-drawn ``(n,)`` standard-normal vector — using pre-drawn
    innovations makes common-random-number comparisons trivial (needed for
    exp_5 finite differences).
    """
    L = correlation_to_cholesky(C)
    return sigma * (L @ epsilon)


def simulate_static_var(
    C_start: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    T: int = 30,
    n_paths: int = 50_000,
    alpha: float = 0.95,
    seed: int = 0,
) -> VaRResult:
    """Static-correlation VaR: correlations frozen at ``C_start`` for T days."""
    rng = np.random.default_rng(seed)
    n = C_start.shape[0]
    L_static = correlation_to_cholesky(C_start)
    # (T, n_paths, n) innovations
    eps = rng.standard_normal((T, n_paths, n))
    # daily returns for each path/day: sigma * (L @ eps_t^T)
    daily = np.einsum("i,ij,tpj->tpi", sigma, L_static, eps)
    # 30-day arithmetic-return aggregate (small approx to compounded log-return,
    # but consistent with the paper's zero-mean Gaussian daily model)
    port_daily = daily @ weights                    # (T, n_paths)
    port_total = port_daily.sum(axis=0)             # (n_paths,)
    losses = -port_total
    var, es = _var_es(losses, alpha)
    return VaRResult(var=var, es=es, losses=losses)


def simulate_dynamic_var(
    C_start: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    rho_crisis: float = 0.9,
    T: int = 30,
    dt: float = 1.0,
    sigma_bridge: float = 0.11,
    n_paths: int = 50_000,
    alpha: float = 0.95,
    seed: int = 0,
    store_paths: bool = False,
) -> VaRResult:
    """Dynamic (crisis-bridge) VaR.

    For each MC path, simulate a fresh Brownian-bridge correlation trajectory
    ``L_0 -> ... -> L_T`` toward the crisis equicorrelation matrix (rho=0.9),
    then draw one Gaussian return per day using ``Sigma_t = diag(sigma) C_t diag(sigma)``.
    """
    n = C_start.shape[0]
    C_crisis = equicorrelation_matrix(n, rho_crisis)
    L0 = correlation_to_cholesky(C_start)
    K = correlation_to_cholesky(C_crisis)

    rng = np.random.default_rng(seed)
    losses = np.zeros(n_paths)
    avg_corr = np.zeros((n_paths, T + 1)) if store_paths else None
    all_paths = np.zeros((n_paths, T + 1, n, n)) if store_paths else None

    off_mask = ~np.eye(n, dtype=bool)

    for p in range(n_paths):
        L = L0.copy()
        port_total = 0.0
        if store_paths:
            C0 = cholesky_to_correlation(L)
            all_paths[p, 0] = C0
            avg_corr[p, 0] = C0[off_mask].mean()
        for k in range(T):
            t = k * dt
            L = bridge_step(L, K, t, T * dt, dt, sigma_bridge, rng)
            C_t = cholesky_to_correlation(L)
            eps = rng.standard_normal(n)
            r = sigma * (correlation_to_cholesky(C_t) @ eps)
            port_total += float(r @ weights)
            if store_paths:
                all_paths[p, k + 1] = C_t
                avg_corr[p, k + 1] = C_t[off_mask].mean()
        losses[p] = -port_total

    var, es = _var_es(losses, alpha)
    return VaRResult(
        var=var, es=es, losses=losses,
        correlation_paths=all_paths, avg_corr_paths=avg_corr,
    )


def _var_es(losses: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Empirical VaR and ES at level ``alpha`` (positive numbers = loss)."""
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def average_pairwise_correlation(C: np.ndarray) -> float:
    """Mean of the off-diagonal entries of ``C``."""
    n = C.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(C[mask].mean())


def scenario_diagnostics(
    C_start: np.ndarray, rho_crisis: float = 0.9
) -> dict:
    """Compute geodesic distance to crisis + initial average correlation."""
    n = C_start.shape[0]
    C_crisis = equicorrelation_matrix(n, rho_crisis)
    return {
        "avg_corr_initial": average_pairwise_correlation(C_start),
        "geodesic_distance": geodesic_distance(C_start, C_crisis),
    }

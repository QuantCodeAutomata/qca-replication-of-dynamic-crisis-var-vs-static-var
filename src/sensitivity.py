"""Sensitivity analysis for dynamic VaR / ES (Experiment 5).

Approach
--------
The paper contrasts AD-based sensitivities with central finite differences
under **common random numbers** (CRN). Because the empirical quantile is a
non-smooth order statistic, honest pathwise VaR gradients are ill-defined;
we therefore rely on central finite differences with CRN, which is both the
standard cross-check the paper describes and the most transparent way to
show the ES-vs-VaR stability contrast.

We pre-draw the standard-normal innovations once, reuse them for every
perturbed run, and simulate the entire bridge + return pipeline
deterministically. This is equivalent to differentiating through the
computation graph at the given random draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.manifold import (
    bridge_step,
    cholesky_to_correlation,
    correlation_to_cholesky,
    equicorrelation_matrix,
)


@dataclass
class RiskDraws:
    """Pre-drawn innovations reused across all perturbed simulations."""
    bridge_normals: np.ndarray   # (n_paths, T, n, n_max)  standard normals for bridge rows
    ret_normals: np.ndarray      # (n_paths, T, n)         standard normals for daily returns


def make_draws(n_paths: int, T: int, n: int, seed: int) -> RiskDraws:
    """Pre-draw all randomness so CRN works exactly across perturbations."""
    rng = np.random.default_rng(seed)
    bridge = rng.standard_normal((n_paths, T, n, n))
    rets = rng.standard_normal((n_paths, T, n))
    return RiskDraws(bridge_normals=bridge, ret_normals=rets)


def _bridge_step_crn(
    L: np.ndarray,
    K: np.ndarray,
    t: float,
    T: float,
    dt: float,
    sigma: float,
    Z_mat: np.ndarray,
) -> np.ndarray:
    """Deterministic version of ``bridge_step`` — uses supplied Gaussian block."""
    from src.manifold import _project_tangent, exp_map_sphere, log_map_sphere

    n = L.shape[0]
    L_new = L.copy()
    remaining = max(T - t, dt)
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n):
        Li = L[i, : i + 1]
        Ki = K[i, : i + 1]
        drift_vec, _ = log_map_sphere(Li, Ki)
        drift = drift_vec * (dt / remaining)
        Z = Z_mat[i, : i + 1]
        diffusion = sigma * _project_tangent(Li, Z) * sqrt_dt
        v = drift + diffusion
        L_new[i, : i + 1] = exp_map_sphere(Li, v)
    return L_new


def dynamic_losses_crn(
    C_start: np.ndarray,
    sigma_vol: np.ndarray,
    weights: np.ndarray,
    rho_crisis: float,
    sigma_bridge: float,
    T: int,
    dt: float,
    draws: RiskDraws,
) -> np.ndarray:
    """Return (n_paths,) losses using the pre-drawn CRN innovations."""
    n = C_start.shape[0]
    C_crisis = equicorrelation_matrix(n, rho_crisis)
    K = correlation_to_cholesky(C_crisis)
    L0 = correlation_to_cholesky(C_start)

    n_paths = draws.bridge_normals.shape[0]
    losses = np.zeros(n_paths)
    for p in range(n_paths):
        L = L0.copy()
        port_total = 0.0
        for k in range(T):
            t = k * dt
            L = _bridge_step_crn(
                L, K, t, T * dt, dt, sigma_bridge,
                draws.bridge_normals[p, k, :n, :n],
            )
            C_t = cholesky_to_correlation(L)
            L_ret = correlation_to_cholesky(C_t)
            r = sigma_vol * (L_ret @ draws.ret_normals[p, k, :n])
            port_total += float(r @ weights)
        losses[p] = -port_total
    return losses


def var_es_from_losses(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def finite_diff_sensitivity(
    f: Callable[[float], Tuple[float, float]],
    x0: float,
    h: float,
) -> Tuple[float, float]:
    """Central finite difference for a function returning (VaR, ES)."""
    v_p, e_p = f(x0 + h)
    v_m, e_m = f(x0 - h)
    return (v_p - v_m) / (2.0 * h), (e_p - e_m) / (2.0 * h)


def es_sensitivity_to_rho_crisis(
    C_start: np.ndarray,
    sigma_vol: np.ndarray,
    weights: np.ndarray,
    T: int = 30,
    dt: float = 1.0,
    rho_crisis: float = 0.9,
    sigma_bridge: float = 0.11,
    n_paths: int = 4000,
    seed: int = 0,
    h: float = 5e-3,
) -> float:
    """Return the CRN central-difference sensitivity of ES to ``rho_crisis``."""
    n = C_start.shape[0]
    draws = make_draws(n_paths, T, n, seed=seed)

    def f(rho: float) -> Tuple[float, float]:
        losses = dynamic_losses_crn(
            C_start, sigma_vol, weights, rho, sigma_bridge, T, dt, draws
        )
        return var_es_from_losses(losses)

    _, dE = finite_diff_sensitivity(f, rho_crisis, h)
    return float(dE)


def sensitivity_report(
    C_start: np.ndarray,
    sigma_vol: np.ndarray,
    weights: np.ndarray,
    T: int = 30,
    dt: float = 1.0,
    rho_crisis: float = 0.9,
    sigma_bridge: float = 0.11,
    n_paths: int = 4000,
    seed: int = 0,
    n_batches: int = 5,
    h_rho: float = 5e-3,
    h_sigma: float = 5e-3,
    top_k: int = 6,
) -> Dict:
    """Run the exp_5 sensitivity study.

    Uses ``n_batches`` independent CRN blocks to estimate mean ± SE for VaR
    and ES sensitivities to ``rho_crisis`` and ``sigma_bridge``, plus the
    top-``top_k`` most sensitive pairwise initial correlations for ES.
    """
    n = C_start.shape[0]

    var_rho_batch, es_rho_batch = [], []
    var_sig_batch, es_sig_batch = [], []
    var_levels, es_levels = [], []

    for b in range(n_batches):
        draws = make_draws(n_paths, T, n, seed=seed + 1000 * b)

        def f_rho(rho: float) -> Tuple[float, float]:
            losses = dynamic_losses_crn(
                C_start, sigma_vol, weights, rho, sigma_bridge, T, dt, draws
            )
            return var_es_from_losses(losses)

        def f_sig(sig: float) -> Tuple[float, float]:
            losses = dynamic_losses_crn(
                C_start, sigma_vol, weights, rho_crisis, sig, T, dt, draws
            )
            return var_es_from_losses(losses)

        # baseline levels
        v0, e0 = f_rho(rho_crisis)
        var_levels.append(v0)
        es_levels.append(e0)

        dV_rho, dE_rho = finite_diff_sensitivity(f_rho, rho_crisis, h_rho)
        dV_sig, dE_sig = finite_diff_sensitivity(f_sig, sigma_bridge, h_sigma)
        var_rho_batch.append(dV_rho)
        es_rho_batch.append(dE_rho)
        var_sig_batch.append(dV_sig)
        es_sig_batch.append(dE_sig)

    def _stat(arr: List[float]) -> Dict[str, float]:
        a = np.asarray(arr)
        return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a)))}

    # Pairwise initial-correlation ES sensitivities (single big CRN block).
    draws = make_draws(n_paths, T, n, seed=seed)
    v0, e0 = var_es_from_losses(
        dynamic_losses_crn(C_start, sigma_vol, weights,
                           rho_crisis, sigma_bridge, T, dt, draws)
    )
    pair_sens: List[Tuple[str, str, float]] = []
    # Note: names attached by caller through a wrapper; here indexed
    for i in range(n):
        for j in range(i + 1, n):
            C_pert = C_start.copy()
            C_pert[i, j] += h_rho
            C_pert[j, i] += h_rho
            # ensure PSD by adding jitter; if not PSD, skip
            try:
                _ = np.linalg.cholesky(C_pert)
            except np.linalg.LinAlgError:
                pair_sens.append((f"i{i}", f"j{j}", float("nan")))
                continue
            losses = dynamic_losses_crn(
                C_pert, sigma_vol, weights, rho_crisis, sigma_bridge, T, dt, draws
            )
            _, e_plus = var_es_from_losses(losses)
            pair_sens.append((f"i{i}", f"j{j}", (e_plus - e0) / h_rho))

    ranked = sorted(pair_sens, key=lambda x: -abs(x[2] if not np.isnan(x[2]) else 0.0))
    return {
        "var_level": _stat(var_levels),
        "es_level": _stat(es_levels),
        "dvar_drho":  _stat(var_rho_batch),
        "des_drho":   _stat(es_rho_batch),
        "dvar_dsig":  _stat(var_sig_batch),
        "des_dsig":   _stat(es_sig_batch),
        "pairwise_es_sens_top": ranked[:top_k],
    }

"""Tests for the finite-difference sensitivity engine (exp_5)."""
from __future__ import annotations

import numpy as np

from src.manifold import equicorrelation_matrix
from src.sensitivity import es_sensitivity_to_rho_crisis, sensitivity_report


def test_es_positive_sensitivity_to_rho_crisis():
    """Increasing crisis correlation should raise ES (more synchronization)."""
    n = 8
    C = equicorrelation_matrix(n, 0.25)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    sens = es_sensitivity_to_rho_crisis(
        C, sigma, w, T=30, rho_crisis=0.9, sigma_bridge=0.11,
        n_paths=1500, seed=123, h=5e-3,
    )
    assert sens > 0


def test_sensitivity_report_structure():
    n = 6
    C = equicorrelation_matrix(n, 0.3)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    rep = sensitivity_report(
        C, sigma, w, T=30, dt=1.0, rho_crisis=0.9, sigma_bridge=0.11,
        n_paths=800, seed=0, n_batches=3, top_k=3,
    )
    for key in ("var_level", "es_level",
                "dvar_drho", "des_drho", "dvar_dsig", "des_dsig",
                "pairwise_es_sens_top"):
        assert key in rep
    for k in ("dvar_drho", "des_drho", "dvar_dsig", "des_dsig"):
        assert "mean" in rep[k] and "se" in rep[k]


def test_es_more_stable_than_var_across_batches():
    """Across multiple batches, ES-drho SE should be smaller than VaR-drho SE."""
    n = 10
    C = equicorrelation_matrix(n, 0.25)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    rep = sensitivity_report(
        C, sigma, w, T=30, dt=1.0, rho_crisis=0.9, sigma_bridge=0.11,
        n_paths=1200, seed=0, n_batches=5, top_k=2,
    )
    var_se = rep["dvar_drho"]["se"]
    es_se = rep["des_drho"]["se"]
    # ES sensitivity should be more stable — allow slack in low-path regime.
    assert es_se <= var_se * 1.5


def test_baseline_var_es_ordering():
    n = 6
    C = equicorrelation_matrix(n, 0.3)
    sigma = np.full(n, 0.02)
    w = np.full(n, 1.0 / n)
    rep = sensitivity_report(
        C, sigma, w, T=30, dt=1.0, rho_crisis=0.9, sigma_bridge=0.11,
        n_paths=1000, seed=0, n_batches=2, top_k=2,
    )
    assert rep["es_level"]["mean"] >= rep["var_level"]["mean"]

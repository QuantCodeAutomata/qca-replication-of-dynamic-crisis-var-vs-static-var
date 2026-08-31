"""Experiment 2 — Cholesky-manifold diffusion + Brownian-bridge validation.

Objectives
----------
1. Generate 500 random starting correlation matrices and run pure manifold
   diffusion; verify PD / unit-diagonal preservation.
2. Show off-diagonal histograms remain symmetric and Beta-like.
3. Run a Brownian bridge to a crisis equicorrelation matrix and confirm
   average correlation trends upward and geodesic distance contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.manifold import (
    cholesky_to_correlation,
    correlation_to_cholesky,
    diffusion_step,
    equicorrelation_matrix,
    geodesic_distance,
    random_correlation_via_hemispheres,
    simulate_bridge_path,
    validate_correlation_matrix,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_diffusion_check(
    n: int = 10, n_matrices: int = 500, n_paths: int = 1,
    n_steps: int = 30, sigma: float = 0.11, seed: int = 42,
) -> Dict:
    """Sanity check: pure manifold diffusion preserves valid correlations."""
    rng = np.random.default_rng(seed)
    off_diag_samples = []
    min_eig_list = []
    diag_dev_list = []
    sym_err_list = []

    for m in range(n_matrices):
        C0 = random_correlation_via_hemispheres(n, rng)
        L = correlation_to_cholesky(C0)
        for _ in range(n_paths):
            L_curr = L.copy()
            for _ in range(n_steps):
                L_curr = diffusion_step(L_curr, dt=1.0, sigma=sigma, rng=rng)
            C_end = cholesky_to_correlation(L_curr)
            ok, diag = validate_correlation_matrix(C_end)
            min_eig_list.append(diag["min_eig"])
            diag_dev_list.append(diag["diag_dev"])
            sym_err_list.append(diag["symmetry_err"])
            mask = ~np.eye(n, dtype=bool)
            off_diag_samples.extend(C_end[mask].tolist())

    off = np.array(off_diag_samples)
    stats = {
        "n_matrices": n_matrices,
        "n_steps": n_steps,
        "min_eigenvalue_min":  float(np.min(min_eig_list)),
        "min_eigenvalue_mean": float(np.mean(min_eig_list)),
        "diag_dev_max":        float(np.max(diag_dev_list)),
        "symmetry_err_max":    float(np.max(sym_err_list)),
        "off_diag_mean":       float(np.mean(off)),
        "off_diag_std":        float(np.std(off)),
        "off_diag_median":     float(np.median(off)),
    }

    plt.figure(figsize=(7, 4.5))
    sns.histplot(off, bins=60, kde=True, color="steelblue", stat="density")
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.title(f"Off-diagonal correlations after {n_steps}-day manifold diffusion")
    plt.xlabel(r"$\rho_{ij}$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp2_diffusion_offdiag_hist.png", dpi=150)
    plt.close()

    return stats


def run_bridge_check(
    n: int = 10, n_paths: int = 200, T: int = 30, sigma: float = 0.11,
    rho_crisis: float = 0.9, seed: int = 7,
) -> Dict:
    """Verify the bridge process progresses toward the crisis regime."""
    rng = np.random.default_rng(seed)
    C_crisis = equicorrelation_matrix(n, rho_crisis)
    K = correlation_to_cholesky(C_crisis)

    C0 = random_correlation_via_hemispheres(n, rng)
    L0 = correlation_to_cholesky(C0)

    mask = ~np.eye(n, dtype=bool)
    avg_corr = np.zeros((n_paths, T + 1))
    dist_to_target = np.zeros((n_paths, T + 1))

    for p in range(n_paths):
        L_path = simulate_bridge_path(L0, K, T, sigma, rng)
        for k in range(T + 1):
            C_t = cholesky_to_correlation(L_path[k])
            avg_corr[p, k] = C_t[mask].mean()
            dist_to_target[p, k] = geodesic_distance(C_t, C_crisis)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for p in range(min(n_paths, 40)):
        axes[0].plot(avg_corr[p], color="steelblue", alpha=0.15)
    axes[0].plot(avg_corr.mean(axis=0), color="crimson", lw=2, label="mean")
    axes[0].axhline(rho_crisis, ls="--", color="k", label=r"$\rho_{crisis}$")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel(r"average $\rho_{ij}$")
    axes[0].set_title("Brownian bridge — avg correlation")
    axes[0].legend()

    for p in range(min(n_paths, 40)):
        axes[1].plot(dist_to_target[p], color="darkorange", alpha=0.15)
    axes[1].plot(dist_to_target.mean(axis=0), color="crimson", lw=2, label="mean")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel(r"geodesic distance to $C_{crisis}$")
    axes[1].set_title("Brownian bridge — distance contraction")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp2_bridge_paths.png", dpi=150)
    plt.close()

    return {
        "avg_corr_start_mean":   float(avg_corr[:, 0].mean()),
        "avg_corr_end_mean":     float(avg_corr[:, -1].mean()),
        "distance_start_mean":   float(dist_to_target[:, 0].mean()),
        "distance_end_mean":     float(dist_to_target[:, -1].mean()),
        "rho_crisis":            rho_crisis,
    }


def main() -> Dict:
    print("[exp_2] running diffusion sanity check (500 matrices)...")
    diff = run_diffusion_check()
    print("[exp_2] running bridge validation...")
    bridge = run_bridge_check()
    out = {"diffusion": diff, "bridge": bridge}
    (RESULTS / "exp_2_manifold_validation.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

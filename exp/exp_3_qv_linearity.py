"""Experiment 3 — Empirical linearity of cumulative quadratic variation.

For each portfolio we:
1. Build daily EWMA correlation matrices (lambda = 0.94).
2. Measure geodesic increments d(C_t, C_{t+1}) on the Cholesky manifold.
3. Regress cumulative sum of squared increments on time and report R^2.

Paper reference values (Table): Technology 0.997, Old Economy 0.993,
Commodities 0.995, Financial services 0.998.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PORTFOLIOS, align_returns, portfolio_prices
from src.ewma import cov_to_corr, ewma_covariance_path
from src.manifold import (
    cholesky_dimension,
    geodesic_qv_increments,
    qv_ols_slope,
    sigma_from_qv_slope,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def qv_process(rets: pd.DataFrame, lam: float = 0.94, burn_in: int = 60) -> pd.Series:
    """Return the cumulative quadratic-variation process on the Cholesky manifold."""
    R = rets.to_numpy(dtype=np.float64)
    cov_path = ewma_covariance_path(R, lam=lam, burn_in=burn_in)
    dates = rets.index[burn_in:burn_in + len(cov_path)]
    C_path = np.empty((len(cov_path), R.shape[1], R.shape[1]), dtype=np.float64)
    for k, S in enumerate(cov_path):
        _, C_path[k] = cov_to_corr(S)
    incs = geodesic_qv_increments(C_path)
    qv = np.cumsum(incs)
    return pd.Series(qv, index=dates[1:], name="QV")


def fit_linear(qv: pd.Series) -> Dict:
    y = qv.to_numpy(dtype=np.float64)
    increments = np.empty_like(y)
    increments[0] = y[0]
    increments[1:] = np.diff(y)
    slope, intercept, r_squared = qv_ols_slope(increments)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n_obs": int(len(qv)),
    }


def main() -> pd.DataFrame:
    rows: List[Dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for k, (pname, tickers) in enumerate(PORTFOLIOS.items()):
        print(f"[exp_3] {pname} - loading prices...")
        prices = portfolio_prices(pname)
        rets = align_returns(prices, kind="log")
        print(f"        aligned returns: {len(rets)} rows from "
              f"{rets.index.min().date()} to {rets.index.max().date()}")

        qv = qv_process(rets)
        stats = fit_linear(qv)
        n_assets = int(rets.shape[1])
        stats.update({
            "portfolio": pname,
            "n_assets": n_assets,
            "cholesky_dim": cholesky_dimension(n_assets),
            "sigma_isotropic": sigma_from_qv_slope(stats["slope"], n_assets),
            "start": str(qv.index.min().date()),
            "end":   str(qv.index.max().date()),
        })
        rows.append(stats)

        ax = axes[k]
        ax.plot(qv.index, qv.values, color="steelblue", lw=1.2, label="QV(k)")
        kk = np.arange(1, len(qv) + 1)
        ax.plot(qv.index, stats["intercept"] + stats["slope"] * kk,
                color="crimson", lw=1.2, ls="--",
                label=fr"OLS $R^2$={stats['r_squared']:.4f}")
        ax.set_title(pname)
        ax.set_ylabel("cumulative QV")
        ax.legend()

    plt.suptitle("Cumulative quadratic variation on Chol_n vs time")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp3_qv_linearity.png", dpi=150)
    plt.close()

    df = pd.DataFrame(rows)[["portfolio", "start", "end", "n_obs",
                             "n_assets", "cholesky_dim", "slope",
                             "sigma_isotropic", "intercept", "r_squared"]]
    df.to_csv(RESULTS / "exp_3_qv_regression.csv", index=False)
    (RESULTS / "exp_3_summary.json").write_text(
        json.dumps({"portfolios": df.to_dict(orient="records")}, indent=2))
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

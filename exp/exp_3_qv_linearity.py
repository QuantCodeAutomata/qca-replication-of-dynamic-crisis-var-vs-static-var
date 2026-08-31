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
import statsmodels.api as sm

from src.data import PORTFOLIOS, align_returns, portfolio_prices
from src.ewma import cov_to_corr, ewma_covariance_path
from src.manifold import correlation_to_cholesky

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _daily_geodesic_increment(L_prev: np.ndarray, L_curr: np.ndarray) -> float:
    """Row-wise arccos-based geodesic distance between two Cholesky factors."""
    n = L_prev.shape[0]
    s = 0.0
    for i in range(1, n):
        Li = L_prev[i, : i + 1]
        Ki = L_curr[i, : i + 1]
        dot = float(np.clip(np.dot(Li, Ki) /
                             (np.linalg.norm(Li) * np.linalg.norm(Ki)),
                             -1.0 + 1e-12, 1.0 - 1e-12))
        s += np.arccos(dot) ** 2
    return float(np.sqrt(s))


def qv_process(rets: pd.DataFrame, lam: float = 0.94, burn_in: int = 60) -> pd.Series:
    """Return the cumulative quadratic-variation process on the Cholesky manifold."""
    R = rets.to_numpy(dtype=np.float64)
    cov_path = ewma_covariance_path(R, lam=lam, burn_in=burn_in)
    dates = rets.index[burn_in:burn_in + len(cov_path)]

    L_prev = None
    incs = []
    for k, S in enumerate(cov_path):
        _, C = cov_to_corr(S)
        L = correlation_to_cholesky(C)
        if L_prev is not None:
            incs.append(_daily_geodesic_increment(L_prev, L) ** 2)
        L_prev = L
    incs = np.asarray(incs)
    qv = np.cumsum(incs)
    return pd.Series(qv, index=dates[1:], name="QV")


def fit_linear(qv: pd.Series) -> Dict:
    k = np.arange(1, len(qv) + 1, dtype=np.float64)
    y = qv.to_numpy()
    X = sm.add_constant(k)
    model = sm.OLS(y, X).fit()
    return {
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "r_squared": float(model.rsquared),
        "n_obs": int(len(qv)),
    }


def main() -> pd.DataFrame:
    rows: List[Dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for k, (pname, tickers) in enumerate(PORTFOLIOS.items()):
        print(f"[exp_3] {pname} — loading prices...")
        prices = portfolio_prices(pname)
        rets = align_returns(prices, kind="log")
        print(f"        aligned returns: {len(rets)} rows from "
              f"{rets.index.min().date()} to {rets.index.max().date()}")

        qv = qv_process(rets)
        stats = fit_linear(qv)
        stats.update({"portfolio": pname,
                      "start": str(qv.index.min().date()),
                      "end":   str(qv.index.max().date())})
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
                             "slope", "intercept", "r_squared"]]
    df.to_csv(RESULTS / "exp_3_qv_regression.csv", index=False)
    (RESULTS / "exp_3_summary.json").write_text(
        json.dumps({"portfolios": df.to_dict(orient="records")}, indent=2))
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

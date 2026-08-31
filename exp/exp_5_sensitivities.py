"""Experiment 5 — Sensitivity analysis of dynamic VaR vs ES.

Uses central finite differences under common random numbers (CRN) to
estimate sensitivities to
    * crisis correlation ``rho_crisis``
    * bridge diffusion volatility ``sigma_bridge``
    * pairwise initial correlations ``rho_ij^0``

The paper's Table 7 (Finance / COVID scenario) reports:
    VaR ≈ 7.74%, ES ≈ 9.61%,
    dVaR/drho = 162 ± 304 bps,  dES/drho = 293 ± 15 bps,
    dVaR/dsigma = -12 ± 20 bps, dES/dsigma = -6.4 ± 2.1 bps,
demonstrating the ES-vs-VaR stability contrast.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CRISIS_STARTS, PORTFOLIOS, align_returns, portfolio_prices
from src.ewma import cov_to_corr, ewma_covariance
from src.sensitivity import sensitivity_report

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _initial_state(returns: pd.DataFrame, scenario_date: str):
    mask = returns.index < pd.Timestamp(scenario_date)
    R = returns.loc[mask].to_numpy(dtype=np.float64)
    S = ewma_covariance(R, lam=0.94, burn_in=60)
    return cov_to_corr(S)


def main(portfolio: str = "Finance", crisis: str = "COVID_2020",
         n_paths: int = 1500, n_batches: int = 5, seed: int = 20250101) -> Dict:
    print(f"[exp_5] loading {portfolio} prices...")
    prices = portfolio_prices(portfolio)
    rets = align_returns(prices, kind="log")
    date = CRISIS_STARTS[crisis]
    sigma, C_start = _initial_state(rets, date)
    n = len(sigma)
    w = np.full(n, 1.0 / n)
    tickers = PORTFOLIOS[portfolio]

    print(f"[exp_5] running CRN sensitivity ({n_batches} batches × {n_paths} paths)...")
    report = sensitivity_report(
        C_start, sigma, w, T=30, dt=1.0, rho_crisis=0.9, sigma_bridge=0.11,
        n_paths=n_paths, seed=seed, n_batches=n_batches,
        h_rho=5e-3, h_sigma=5e-3, top_k=6,
    )

    # Attach ticker pair names to the top-k initial-correlation sensitivities.
    def _name(pair):
        i = int(pair[0][1:]); j = int(pair[1][1:])
        return f"{tickers[i]}-{tickers[j]}", pair[2]
    pair_named = [ _name(p) for p in report["pairwise_es_sens_top"] ]
    report["pairwise_es_sens_top_named"] = pair_named
    report["portfolio"] = portfolio
    report["crisis"] = crisis
    report["scenario_date"] = date

    # Compact table for the RESULTS.md consumer.
    rel_var_se = (report["dvar_drho"]["se"] /
                  max(abs(report["dvar_drho"]["mean"]), 1e-12))
    rel_es_se = (report["des_drho"]["se"] /
                 max(abs(report["des_drho"]["mean"]), 1e-12))
    report["relative_se_dvar_drho"] = float(rel_var_se)
    report["relative_se_des_drho"] = float(rel_es_se)
    report["es_more_stable_than_var"] = bool(rel_es_se < rel_var_se)
    report["var_baseline_pct"] = float(100 * report["var_level"]["mean"])
    report["es_baseline_pct"] = float(100 * report["es_level"]["mean"])

    (RESULTS / "exp_5_sensitivity.json").write_text(json.dumps(report, indent=2))

    # Bar chart of VaR vs ES sensitivities with error bars
    labels = ["dX/drho (bps)", "dX/dsigma (bps)"]
    var_vals = [report["dvar_drho"]["mean"] * 100 * 100,
                report["dvar_dsig"]["mean"] * 100 * 100]
    var_ses  = [report["dvar_drho"]["se"] * 100 * 100,
                report["dvar_dsig"]["se"] * 100 * 100]
    es_vals  = [report["des_drho"]["mean"] * 100 * 100,
                report["des_dsig"]["mean"] * 100 * 100]
    es_ses   = [report["des_drho"]["se"] * 100 * 100,
                report["des_dsig"]["se"] * 100 * 100]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    w_bar = 0.35
    ax.bar(x - w_bar/2, var_vals, w_bar, yerr=var_ses, capsize=5,
           color="steelblue", label="VaR")
    ax.bar(x + w_bar/2, es_vals, w_bar, yerr=es_ses, capsize=5,
           color="crimson", label="ES")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("sensitivity (bps per unit)")
    ax.set_title(f"VaR vs ES sensitivities ({portfolio} / {crisis})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp5_sensitivity_bars.png", dpi=150)
    plt.close()

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

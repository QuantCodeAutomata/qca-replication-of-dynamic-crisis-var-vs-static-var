"""Experiment 1 — Dynamic Crisis VaR vs Static VaR across 12 scenarios.

Reproduces the paper's Table-6-style comparison for
  {Technology, Old Economy, Finance, Commodities} x {GFC 2008, Downgrade 2011, COVID 2020}.

Implementation assumptions (paper leaves these unspecified — see RESULTS.md):
* Equal weights normalized to 100% gross notional.
* Log returns.
* EWMA lambda = 0.94, burn-in = 60 observations.
* Missing-data treatment: listwise deletion after the last IPO in the panel.
* Monte Carlo count = 5,000 paths (feasible on CPU; paper suggests up to 50k).
* Random seed fixed = 20250101.
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
import seaborn as sns

from src.data import CRISIS_STARTS, PORTFOLIOS, align_returns, portfolio_prices
from src.ewma import cov_to_corr, ewma_covariance
from src.manifold import equicorrelation_matrix
from src.var_engine import (
    average_pairwise_correlation,
    scenario_diagnostics,
    simulate_dynamic_var,
    simulate_static_var,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_PATHS = 5_000
SEED = 20250101
RHO_CRISIS = 0.9
SIGMA_BRIDGE = 0.11
T = 30


def _initial_state(returns: pd.DataFrame, scenario_date: str, lam: float = 0.94):
    """EWMA-estimated (sigma, C_start) using data strictly before ``scenario_date``."""
    mask = returns.index < pd.Timestamp(scenario_date)
    r = returns.loc[mask].to_numpy(dtype=np.float64)
    if r.shape[0] < 200:
        raise ValueError(f"Too little history before {scenario_date}: {r.shape[0]} rows")
    S = ewma_covariance(r, lam=lam, burn_in=60)
    sigma, C = cov_to_corr(S)
    return sigma, C, r.shape[0]


def run_scenario(portfolio_name: str, crisis_name: str, scenario_date: str,
                 returns: pd.DataFrame) -> Dict:
    """Compute static + dynamic VaR/ES + diagnostics for one scenario."""
    sigma, C_start, n_hist = _initial_state(returns, scenario_date)
    diag = scenario_diagnostics(C_start, RHO_CRISIS)
    n = len(sigma)
    w = np.full(n, 1.0 / n)

    static = simulate_static_var(
        C_start, sigma, w, T=T, n_paths=N_PATHS, alpha=0.95, seed=SEED,
    )
    dynamic = simulate_dynamic_var(
        C_start, sigma, w, rho_crisis=RHO_CRISIS, T=T, dt=1.0,
        sigma_bridge=SIGMA_BRIDGE, n_paths=N_PATHS, alpha=0.95, seed=SEED + 1,
    )

    return {
        "portfolio": portfolio_name,
        "crisis": crisis_name,
        "scenario_date": scenario_date,
        "n_history_days": int(n_hist),
        "avg_corr_initial": diag["avg_corr_initial"],
        "geodesic_distance": diag["geodesic_distance"],
        "var_static_pct":  100 * static.var,
        "es_static_pct":   100 * static.es,
        "var_dynamic_pct": 100 * dynamic.var,
        "es_dynamic_pct":  100 * dynamic.es,
        "var_uplift_pct":  100 * (dynamic.var - static.var),
        "var_uplift_bps":  10_000 * (dynamic.var - static.var),
        "es_uplift_bps":   10_000 * (dynamic.es - static.es),
        "static_losses":   static.losses,
        "dynamic_losses":  dynamic.losses,
    }


def _plot_histograms(rows: List[Dict]) -> None:
    """Save static vs dynamic loss histograms per scenario grid."""
    n = len(rows)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for k, r in enumerate(rows):
        ax = axes[k]
        sns.histplot(r["static_losses"], bins=60, ax=ax, stat="density",
                     color="steelblue", alpha=0.55, label="static")
        sns.histplot(r["dynamic_losses"], bins=60, ax=ax, stat="density",
                     color="crimson", alpha=0.5, label="dynamic")
        ax.axvline(r["var_static_pct"] / 100, ls="--", color="steelblue")
        ax.axvline(r["var_dynamic_pct"] / 100, ls="--", color="crimson")
        ax.set_title(f"{r['portfolio']} — {r['crisis']}")
        ax.set_xlabel("30-day loss")
        ax.legend(fontsize=7)
    for k in range(len(rows), len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp1_loss_histograms.png", dpi=150)
    plt.close()


def main() -> pd.DataFrame:
    rows: List[Dict] = []
    for pname, tickers in PORTFOLIOS.items():
        print(f"[exp_1] loading {pname} ({len(tickers)} names)...")
        prices = portfolio_prices(pname)
        rets = align_returns(prices, kind="log")
        for crisis, date in CRISIS_STARTS.items():
            first_full_row = rets.index.min()
            if pd.Timestamp(date) <= first_full_row + pd.Timedelta(days=250):
                print(f"  [skip] {pname}/{crisis}: pre-IPO or insufficient history "
                      f"(first full obs {first_full_row.date()}).")
                continue
            print(f"  [{pname}/{crisis}] scenario date = {date}")
            row = run_scenario(pname, crisis, date, rets)
            rows.append(row)

    # keep loss arrays only for plotting; drop before saving CSV
    _plot_histograms(rows)
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("static_losses", "dynamic_losses")}
                       for r in rows])
    df = df.sort_values(["crisis", "portfolio"]).reset_index(drop=True)
    df.to_csv(RESULTS / "exp_1_var_table.csv", index=False)

    # Bar chart of VaR uplift
    plt.figure(figsize=(9, 4.5))
    df["label"] = df["portfolio"] + " / " + df["crisis"]
    order = df.sort_values("var_uplift_pct")["label"]
    sns.barplot(data=df, x="var_uplift_pct", y="label", order=order,
                color="steelblue")
    plt.xlabel("Dynamic VaR − Static VaR  (%)")
    plt.ylabel("")
    plt.title("VaR uplift by portfolio × crisis")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp1_uplift_bars.png", dpi=150)
    plt.close()

    # Save JSON summary
    summary = {
        "n_paths": N_PATHS, "T": T, "rho_crisis": RHO_CRISIS,
        "sigma_bridge": SIGMA_BRIDGE, "seed": SEED,
        "n_scenarios": int(len(df)),
        "dynamic_exceeds_static_fraction": float((df["var_uplift_pct"] > 0).mean()),
        "mean_uplift_pct": float(df["var_uplift_pct"].mean()),
        "median_uplift_pct": float(df["var_uplift_pct"].median()),
    }
    (RESULTS / "exp_1_summary.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

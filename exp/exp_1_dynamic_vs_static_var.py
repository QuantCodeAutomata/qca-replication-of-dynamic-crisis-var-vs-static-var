"""Experiment 1 — Dynamic Crisis VaR vs Static VaR across 12 scenarios.

Reproduces the paper's Table-6-style comparison for
  {Technology, Old Economy, Finance, Commodities} x {GFC 2008, Downgrade 2011, COVID 2020}.

Implementation assumptions (paper leaves these unspecified — see RESULTS.md):
* Equal weights normalized to 100% gross notional.
* Log returns.
* EWMA lambda = 0.94, burn-in = 60 observations.
* As-of date = crisis correlation peak minus 30 calendar days (the bridge ends
  at the peak, per the paper's fig. 3); horizon T = 21 trading days (~30
  calendar days).
* sigma_bridge is not a global constant. For each scenario it is recovered
  from the pre-as-of EWMA geodesic-QV slope α on that panel via the isotropic
  BM identity σ = sqrt(2α / (n(n-1))) = sqrt(α / dim(Chol_n)). The paper's
  fig. 4 slopes ~0.11 are the QV rate of the full manifold (dim = 45 at
  n = 10), not the per-coordinate σ; using 0.11 as σ_bridge therefore
  overstates diffusion by ~√45. The dimension-corrected value is ~0.05.
* Missing-data treatment: Massive→Yahoo history fallback; at each scenario,
  assets with fewer than 200 pre-scenario returns are excluded.
* Monte Carlo count = 5,000 paths (feasible on CPU; the paper does not
  specify its VaR path count).
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

from src.data import CRISIS_STARTS, PORTFOLIOS, portfolio_prices, scenario_returns
from src.ewma import cov_to_corr, ewma_covariance
from src.manifold import estimate_bridge_sigma
from src.var_engine import (
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
T = 21
MIN_HISTORY = 200


def _returns_for_scenario(
    prices: pd.DataFrame,
    scenario_date: str,
    min_history: int = MIN_HISTORY,
) -> tuple[pd.DataFrame, List[str]]:
    """Build a no-look-ahead universe with enough history for one scenario."""
    return scenario_returns(
        prices,
        scenario_date,
        kind="log",
        min_history=min_history,
    )


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
    # Same pre-as-of window as the EWMA snapshot: no look-ahead in σ.
    sigma_bridge, qv_info = estimate_bridge_sigma(
        returns.to_numpy(dtype=np.float64)
    )

    static = simulate_static_var(
        C_start, sigma, w, T=T, n_paths=N_PATHS, alpha=0.95, seed=SEED,
    )
    dynamic = simulate_dynamic_var(
        C_start, sigma, w, rho_crisis=RHO_CRISIS, T=T, dt=1.0,
        sigma_bridge=sigma_bridge, n_paths=N_PATHS, alpha=0.95, seed=SEED + 1,
    )

    return {
        "portfolio": portfolio_name,
        "crisis": crisis_name,
        "scenario_date": scenario_date,
        "n_assets": int(n),
        "assets": ",".join(returns.columns),
        "n_history_days": int(n_hist),
        "cholesky_dim": int(qv_info["cholesky_dim"]),
        "qv_slope": qv_info["qv_slope"],
        "qv_r2": qv_info["qv_r2"],
        "sigma_bridge": sigma_bridge,
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
        ax.set_xlabel("21-trading-day (~30-calendar-day) loss")
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
        for crisis, date in CRISIS_STARTS.items():
            rets, excluded = _returns_for_scenario(prices, date)
            print(f"  [{pname}/{crisis}] scenario date = {date}")
            if excluded:
                print(f"    excluding unavailable assets: {', '.join(excluded)}")
            row = run_scenario(pname, crisis, date, rets)
            print(
                f"    n={row['n_assets']} dim={row['cholesky_dim']} "
                f"QV slope={row['qv_slope']:.4f} -> "
                f"sigma={row['sigma_bridge']:.4f}"
            )
            row["excluded_assets"] = ",".join(excluded)
            rows.append(row)

    # keep loss arrays only for plotting; drop before saving CSV
    _plot_histograms(rows)
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("static_losses", "dynamic_losses")}
                       for r in rows])
    df = df.sort_values(["crisis", "portfolio"]).reset_index(drop=True)
    df.to_csv(RESULTS / "exp_1_var_table.csv", index=False)
    sigma_columns = [
        "portfolio", "crisis", "scenario_date", "n_assets", "cholesky_dim",
        "qv_slope", "qv_r2", "sigma_bridge", "excluded_assets",
    ]
    df[sigma_columns].rename(columns={"scenario_date": "asof"}).to_csv(
        RESULTS / "sigma_from_qv_by_scenario.csv",
        index=False,
    )

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
        "bridge_endpoint": "pinned_to_crisis_target",
        "sigma_rule": "sqrt(qv_slope / cholesky_dim)",
        "seed": SEED,
        "n_scenarios": int(len(df)),
        "mean_sigma_bridge": float(df["sigma_bridge"].mean()),
        "min_sigma_bridge": float(df["sigma_bridge"].min()),
        "max_sigma_bridge": float(df["sigma_bridge"].max()),
        "dynamic_exceeds_static_fraction": float((df["var_uplift_pct"] > 0).mean()),
        "mean_uplift_pct": float(df["var_uplift_pct"].mean()),
        "median_uplift_pct": float(df["var_uplift_pct"].median()),
    }
    (RESULTS / "exp_1_summary.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

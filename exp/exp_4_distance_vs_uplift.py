"""Experiment 4 — Cross-sectional regression of VaR uplift on geodesic distance.

Consumes ``results/exp_1_var_table.csv`` produced by exp_1 and fits
    DeltaVaR (bps) = alpha + beta * d(C_start, C_crisis) + eps

The title of Figure 6 on page 13 reports R^2 = 0.839 for the paper's
12 ten-asset scenario cells. The primary fit here therefore uses the eight
scenario cells that retain all ten assets; the pooled 12-cell fit, which mixes
9- and 10-asset geodesic distances, is reported as a secondary diagnostic.
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
import statsmodels.api as sm

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"


def main() -> Dict:
    src_csv = RESULTS / "exp_1_var_table.csv"
    if not src_csv.exists():
        raise FileNotFoundError(
            "Run exp_1 first — its CSV feeds the distance-vs-uplift regression."
        )
    df = pd.read_csv(src_csv)
    df = df.dropna(subset=["geodesic_distance", "var_uplift_bps"]).copy()

    # The paper's portfolio tables list ten names. This replication shrinks to
    # nine where META/MPC pre-date their IPOs, so use the dimension-consistent
    # ten-asset cells as the primary sample.
    full = df[df["excluded_assets"].isna()] if "excluded_assets" in df else df
    if len(full) < 3:
        raise ValueError("Need at least three full-universe cells for OLS.")

    x = full["geodesic_distance"].to_numpy()
    y = full["var_uplift_bps"].to_numpy()
    model = sm.OLS(y, sm.add_constant(x)).fit()

    pooled_x = df["geodesic_distance"].to_numpy()
    pooled_y = df["var_uplift_bps"].to_numpy()
    pooled_model = sm.OLS(pooled_y, sm.add_constant(pooled_x)).fit()

    stats = {
        "primary_sample": "full_10_asset_cells",
        "n_obs": int(len(full)),
        "slope_bps_per_unit_distance": float(model.params[1]),
        "intercept_bps": float(model.params[0]),
        "r_squared": float(model.rsquared),
        "slope_stderr": float(model.bse[1]),
        "slope_pvalue": float(model.pvalues[1]),
        "pooled_n_obs": int(len(df)),
        "pooled_slope_bps_per_unit_distance": float(pooled_model.params[1]),
        "pooled_intercept_bps": float(pooled_model.params[0]),
        "pooled_r_squared": float(pooled_model.rsquared),
        "pooled_slope_stderr": float(pooled_model.bse[1]),
        "pooled_slope_pvalue": float(pooled_model.pvalues[1]),
    }
    (RESULTS / "exp_4_summary.json").write_text(json.dumps(stats, indent=2))

    plt.figure(figsize=(7.5, 5))
    for pname, grp in df.groupby("portfolio"):
        plt.scatter(grp["geodesic_distance"], grp["var_uplift_bps"],
                    label=pname, s=70)
    xs = np.linspace(pooled_x.min(), pooled_x.max(), 100)
    plt.plot(xs, model.params[0] + model.params[1] * xs, "k-",
             label=fr"10-asset OLS  $R^2$={model.rsquared:.3f}")
    plt.plot(
        xs,
        pooled_model.params[0] + pooled_model.params[1] * xs,
        color="0.5",
        ls="--",
        label=fr"Pooled OLS  $R^2$={pooled_model.rsquared:.3f}",
    )
    plt.xlabel(r"Geodesic distance $d(C_{start}, C_{crisis})$")
    plt.ylabel(r"$\Delta\mathrm{VaR}$  (bps)")
    plt.title("Cross-sectional distance vs. VaR uplift")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp4_distance_vs_uplift.png", dpi=150)
    plt.close()

    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
